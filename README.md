# Stress Detection in Social Media Posts with Domain-Adapted Language Models

Deep Learning course project (team P24): **Suyash Yadav, Saurabh Bisht, Sarthak Mishra**.

We detect stress in Reddit posts from the **Dreaddit** dataset (Turcan & McKeown, 2019) and ask
whether **mental-health-domain pretraining** (MentalBERT / MentalRoBERTa, Ji et al., 2022; optional
domain-adaptive pretraining) beats general-purpose models. We compare four model families and use
LIME and attention to show which language cues each model relies on.

| Family | Models |
|---|---|
| Baseline | BiLSTM + GloVe-100d (and random-embedding ablation) |
| General LM | BERT-base, RoBERTa-base |
| Domain-adapted LM | MentalBERT, MentalRoBERTa, (stretch) BERT + DAPT on Dreaddit text |
| LLM | Qwen2.5-7B-Instruct zero-shot and 4-shot (next-token Yes/No probability) |

**Research questions**
1. Does mental-health-domain pretraining improve stress detection over general pretraining?
2. How do zero-/few-shot LLMs compare with fine-tuned encoders?
3. Which linguistic cues drive the predictions, and do the general and domain-adapted models differ?

## Results

> Filled in from `results/summary.csv` (mean ± std over 3 seeds, test set, n = 715).

| Model | Accuracy | Macro-F1 | F1 (stress) |
|---|---|---|---|
| LSTM + GloVe | | | |
| BERT-base | | | |
| RoBERTa-base | | | |
| MentalBERT | | | |
| MentalRoBERTa | | | |
| Qwen2.5-7B 0-shot | | | |
| Qwen2.5-7B 4-shot | | | |

## Repository layout

```
src/
  data.py               download Dreaddit, build the shared train/val/test split
  eval.py               metrics, save_run(), summary table, McNemar + paired bootstrap
  train_lstm.py         BiLSTM + GloVe baseline
  train_transformer.py  fine-tune BERT / RoBERTa / MentalBERT / MentalRoBERTa / DAPT checkpoint
  dapt.py               (stretch) continued masked-LM pretraining on in-domain text
  llm_zeroshot.py       zero-/few-shot LLM classification
  explain.py            LIME global cue words, LIME HTML, [CLS]-attention heatmaps
  analysis.py           per-subreddit F1, disagreements, error lists, confusion matrices
scripts/                one runner per work package (see "Reproduce")
notebooks/              Kaggle/Colab runner notebook
data/splits/val_post_ids.txt   the fixed validation posts (committed)
results/preds, results/metrics one CSV + JSON per run (committed; small)
figures/                plots used in the report
report/                 LaTeX sections + references for the Overleaf NLE template
```

## Setup

```bash
git clone https://github.com/Iamdevsuyash/stress-detection-dreaddit.git && cd stress-detection-dreaddit
pip install -r requirements.txt
python -m src.data --stats        # downloads Dreaddit and writes data/splits/*.csv
```

**Gated models.** MentalBERT and MentalRoBERTa need you to accept their terms on HuggingFace
(<https://huggingface.co/mental/mental-bert-base-uncased>). Then `huggingface-cli login`, or set
`HF_TOKEN` (on Kaggle: *Add-ons → Secrets*). They ship only `pytorch_model.bin`, so **torch ≥ 2.6**
is required.

**GPU.** Use Kaggle (T4, ~30 GPU-h/week per account, Internet must be switched on) or Colab.
`notebooks/kaggle_runner.ipynb` clones the repo, installs dependencies and runs one script.

## Reproduce

```bash
bash scripts/run_baselines.sh        # LSTM, 3 seeds                     (Suyash)
bash scripts/run_transformers.sh     # 4 encoders x 3 seeds (+ DAPT=1)   (Saurabh)
bash scripts/run_llm.sh              # LLM 0-shot / 4-shot               (Suyash)
bash scripts/run_explain.sh          # LIME, attention, error analysis   (Sarthak)
python -m src.eval summarize         # -> results/summary.csv
python -m src.eval compare --a bert-base --b mentalbert
```

Every script reads data only through `src.data.load_splits()` and writes results only through
`src.eval.save_run()`, so all numbers are computed the same way. Model selection uses the validation
split; the test set is touched once per run.

## Team and contributions

Each area has a **lead** who makes sure it gets done. Nobody is the only person who understands a part:
every member must be able to explain every model, script, result and design decision.

| Area | Lead | Also worked on it |
|---|---|---|
| Data, split, evaluation, LSTM baseline | Suyash | Saurabh, Sarthak (review) |
| BERT / RoBERTa / MentalBERT / MentalRoBERTa, DAPT, significance tests | Saurabh | Suyash, Sarthak (review) |
| LLM zero-/few-shot | Suyash | Saurabh, Sarthak (review) |
| LIME, attention, error analysis, ethics | Sarthak | Suyash, Saurabh (review) |
| Report | all (each writes and reviews sections outside their own area) | |

How we share knowledge:
- Every pull request is reviewed by **both** other members. Approving means “I can explain this code”.
- Everyone runs every script at least once (`--limit` / `--epochs 1` is enough to see it work).

## Team rules

- **Do not change** `data/splits/val_post_ids.txt` or `SPLIT_SEED`. Seeds are **13, 42, 87**.
- Model tags: `lstm-glove`, `lstm-random`, `bert-base`, `roberta-base`, `mentalbert`,
  `mentalroberta`, `bert-dapt`, `llm-<model>-<k>shot`.
- Work on your own branch (`suyash/…`, `saurabh/…`, `sarthak/…`) and open a PR to `main`.
- After a Kaggle run, download `results/` and `figures/` and commit them. Never commit weights or
  raw data.

## Ethics

Dreaddit consists of public Reddit posts, but the authors are real people discussing abuse, PTSD,
anxiety and poverty. We do not redistribute post text, we do not try to identify users, and we
report only aggregate results. A stress classifier is **not** a diagnostic tool. False negatives and
false positives both carry real costs, and automated monitoring of individuals without consent would
be a misuse of this work. See the Ethics section of the report.

## References

- Turcan & McKeown (2019). *Dreaddit: A Reddit Dataset for Stress Analysis in Social Media.* LOUHI. <https://arxiv.org/abs/1911.00133>
- Ji et al. (2022). *MentalBERT: Publicly Available Pretrained Language Models for Mental Healthcare.* LREC. <https://arxiv.org/abs/2110.15621>
