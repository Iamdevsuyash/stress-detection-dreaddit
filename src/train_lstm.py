"""BiLSTM + GloVe baseline (the non-transformer reference model of the report).

What this file does
    Trains a bidirectional LSTM classifier on the Dreaddit training split, picks the best epoch on the
    validation split (macro-F1, with early stopping), evaluates it once on the test split and saves the
    run with src.eval.save_run(). With --no_glove the same model starts from random word embeddings;
    comparing the two ("lstm-glove" vs "lstm-random") shows what pretrained word vectors add.

Model
    words -> 100-d embeddings (GloVe 6B, fine-tuned) -> 1-layer BiLSTM (128 units per direction)
    -> [max-pool ; mean-pool] over time -> dropout 0.5 -> linear layer -> 2 class scores

Usage:
    python -m src.train_lstm --glove data/glove/glove.6B.100d.txt --seed 13
    python -m src.train_lstm --seed 13 --no_glove     # random embeddings (ablation)

GloVe: if --glove is missing it is downloaded automatically (~860 MB, once) from the HuggingFace mirror
of the official Stanford zip (huggingface.co/stanfordnlp/glove) and extracted to that path.
"""
import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from src.data import load_splits
from src.eval import save_run
from src.utils import get_device, set_seed

# Tokenizer: runs of lowercase letters/digits/apostrophes form a word; any other non-space character
# (punctuation such as "!" or "?") becomes its own token.
TOKEN_RE = re.compile(r"[a-z0-9']+|[^\sa-z0-9]")
PAD, UNK = 0, 1  # reserved vocabulary ids: padding and unknown word


def tokenize(text: str) -> list[str]:
    """Lowercase the text and split it into word and punctuation tokens."""
    return TOKEN_RE.findall(text.lower())


def build_vocab(texts, min_freq=2):
    """Map every token that occurs at least min_freq times in the TRAINING texts to an integer id.

    Rare words (seen once) are left out and later mapped to <unk>, which keeps the vocabulary small
    and avoids learning embeddings from a single example. Built from train only, so no val/test leak.
    """
    counts = Counter(tok for t in texts for tok in tokenize(t))
    itos = ["<pad>", "<unk>"] + [w for w, c in counts.most_common() if c >= min_freq]
    return {w: i for i, w in enumerate(itos)}


def encode(texts, vocab, max_len):
    """Turn texts into a padded id matrix [n, max_len] plus the true length of each text.

    Texts longer than max_len tokens are cut; shorter ones are padded with PAD (id 0). The lengths
    let the LSTM skip padding (packed sequences) and let pooling ignore padded positions.
    """
    ids = np.full((len(texts), max_len), PAD, dtype=np.int64)
    lengths = np.zeros(len(texts), dtype=np.int64)
    for i, t in enumerate(texts):
        toks = [vocab.get(w, UNK) for w in tokenize(t)][:max_len] or [UNK]  # empty text -> one <unk>
        ids[i, : len(toks)] = toks
        lengths[i] = len(toks)
    return torch.tensor(ids), torch.tensor(lengths)


def ensure_glove(path, dim):
    """Download glove.6B.<dim>d.txt from the HuggingFace mirror of the official Stanford zip if missing
    (nlp.stanford.edu is often unreachable from Kaggle/Colab)."""
    path = Path(path)
    if path.exists():
        return path
    import zipfile

    from huggingface_hub import hf_hub_download

    print(f"{path} not found -> downloading glove.6B.zip (~860 MB) from huggingface.co/stanfordnlp/glove")
    zpath = hf_hub_download("stanfordnlp/glove", "glove.6B.zip")
    member = f"glove.6B.{dim}d.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as zf, zf.open(member) as src, open(path, "wb") as dst:
        while chunk := src.read(1 << 24):  # copy in 16 MB chunks to keep memory low
            dst.write(chunk)
    print(f"extracted {member} to {path}")
    return path


def load_glove(path, vocab, dim):
    """Build the initial embedding matrix [vocab_size, dim] from GloVe.

    Words found in GloVe get their pretrained vector; words not in GloVe get a small random vector
    (normal, std 0.1); the padding row is all zeros. Prints the vocabulary coverage.

    Why random (not zero) for missing words: identical zero vectors would make all unknown words
    indistinguishable at the start; small random vectors let training give each its own meaning.
    The padding row is zero so padding contributes nothing.
    """
    path = ensure_glove(path, dim)
    emb = np.random.normal(0, 0.1, (len(vocab), dim)).astype(np.float32)
    emb[PAD] = 0
    found = 0
    with open(path, encoding="utf8") as f:
        for line in f:  # each line: word followed by dim numbers
            word, *vec = line.rstrip().split(" ")
            if word in vocab and len(vec) == dim:
                emb[vocab[word]] = np.asarray(vec, dtype=np.float32)
                found += 1
    print(f"GloVe coverage: {found}/{len(vocab)} ({100 * found / len(vocab):.1f}%)")
    return torch.tensor(emb)


class BiLSTMClassifier(nn.Module):
    """Embedding -> bidirectional LSTM -> max+mean pooling -> linear layer with 2 outputs.

    Pooling over all time steps (instead of using only the last hidden state) lets the classifier use
    a stress cue wherever it occurs in the segment: max-pooling captures the strongest signal,
    mean-pooling the overall tone.
    """

    def __init__(self, vocab_size, emb_dim, hidden, emb_weights=None, freeze_emb=False, dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        if emb_weights is not None:  # GloVe initialisation; otherwise PyTorch's random init is kept
            self.emb.weight.data.copy_(emb_weights)
            self.emb.weight.requires_grad = not freeze_emb  # we fine-tune the embeddings (freeze_emb=False)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        # Dropout 0.5 (applied to embeddings and to the pooled vector): a strong regulariser, because a
        # model trained from scratch on ~2.5k segments overfits quickly. Switched off by model.eval().
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(4 * hidden, 2)  # [max-pool ; mean-pool] of BiLSTM states

    def forward(self, ids, lengths):
        x = self.drop(self.emb(ids))  # [batch, time, emb_dim]
        # Packing tells the LSTM the real length of each sequence so it does not run over padding.
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=ids.size(1))  # [batch, time, 2*hidden]
        mask = (ids != PAD).unsqueeze(-1)  # True at real tokens, False at padding
        h_max = h.masked_fill(~mask, -1e9).max(dim=1).values  # padding can never win the max
        h_mean = (h * mask).sum(1) / mask.sum(1).clamp(min=1)  # average over real tokens only
        return self.out(self.drop(torch.cat([h_max, h_mean], dim=-1)))  # [batch, 2] class scores (logits)


@torch.no_grad()
def predict(model, ids, lengths, device, bs=256):
    """Return (predicted labels, P(stress)) for all texts; label = 1 if P(stress) >= 0.5."""
    model.eval()  # disables dropout
    probs = []
    for i in range(0, len(ids), bs):
        logits = model(ids[i : i + bs].to(device), lengths[i : i + bs])
        probs.append(torch.softmax(logits, -1)[:, 1].cpu())  # probability of class 1 (stress)
    p = torch.cat(probs).numpy()
    return (p >= 0.5).astype(int), p


def main():
    """Train one BiLSTM run (one seed), pick the best epoch on validation and save its test results."""
    # ---- hyper-parameters (the values in the report's hyper-parameter table) ----
    # None of these were tuned on the test set. Why each value:
    #   emb_dim 100  must match the GloVe file used (glove.6B.100d).
    #   hidden 128   a modest size, kept small because there are only ~2.5k training segments.
    #   max_len 200  tokens (words AND punctuation). Segments average ~98 such tokens; only 46 of 3,553 (1.3%)
    #                are longer than 200, so almost nothing is cut while padding stays cheap.
    #   epochs 20 / patience 4  20 is only a ceiling; early stopping ends training sooner. Patience 4 (not 1)
    #                because val macro-F1 on just 283 segments is noisy from epoch to epoch.
    #   lr 1e-3      Adam's standard default learning rate.
    ap = argparse.ArgumentParser()
    ap.add_argument("--glove", default="data/glove/glove.6B.100d.txt")
    ap.add_argument("--no_glove", action="store_true")
    ap.add_argument("--emb_dim", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--max_len", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    # ---- data: shared split, vocabulary from train only, texts -> padded id matrices ----
    set_seed(args.seed)
    device = get_device()
    train, val, test = load_splits()
    vocab = build_vocab(train["text"])
    Xtr, Ltr = encode(train["text"], vocab, args.max_len)
    Xva, Lva = encode(val["text"], vocab, args.max_len)
    Xte, Lte = encode(test["text"], vocab, args.max_len)
    ytr = torch.tensor(train["label"].values)

    # ---- model, optimiser (Adam) and loss (cross-entropy over the 2 classes) ----
    emb = None if args.no_glove else load_glove(args.glove, vocab, args.emb_dim)
    model = BiLSTMClassifier(len(vocab), args.emb_dim, args.hidden, emb).to(device)
    # Only parameters with requires_grad are handed to the optimiser (matters if embeddings were frozen).
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # ---- training loop with early stopping on validation macro-F1 ----
    best_f1, best_state, bad = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr))  # new random order of training examples every epoch
        total = 0.0
        for i in range(0, len(perm), args.batch_size):
            b = perm[i : i + args.batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[b].to(device), Ltr[b]), ytr[b].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # avoid exploding gradients in the LSTM
            opt.step()
            total += loss.item() * len(b)
        val_pred, _ = predict(model, Xva, Lva, device)
        val_f1 = f1_score(val["label"], val_pred, average="macro")
        print(f"epoch {epoch:2d} loss={total / len(Xtr):.4f} val_f1_macro={val_f1:.4f}")
        if val_f1 > best_f1:  # new best epoch: remember its weights
            best_f1, bad = val_f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:  # no improvement: stop after `patience` such epochs in a row
            bad += 1
            if bad >= args.patience:
                print("early stopping")
                break

    # ---- restore the best epoch, evaluate ONCE on the test set and save the run ----
    model.load_state_dict(best_state)
    pred, prob = predict(model, Xte, Lte, device)
    tag = args.tag or ("lstm-random" if args.no_glove else "lstm-glove")
    save_run(tag, args.seed, test, pred, prob, extra={"val_f1_macro": best_f1, **vars(args)})


if __name__ == "__main__":
    main()
