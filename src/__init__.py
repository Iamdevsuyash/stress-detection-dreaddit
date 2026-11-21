"""Source code of the stress-detection project (Dreaddit).

Pipeline, in the order the scripts are run:
    data.py               download Dreaddit and build the one shared train/val/test split
    train_lstm.py         BiLSTM + GloVe baseline (and random-embedding ablation)
    train_transformer.py  fine-tune BERT / RoBERTa / MentalBERT / MentalRoBERTa       (RQ1)
    llm_zeroshot.py       zero-/few-shot Qwen2.5-7B-Instruct, no training               (RQ2)
    explain.py            LIME cue words and attention heatmaps                         (RQ3)
    analysis.py           per-subreddit F1, disagreements, error lists, confusion matrices
    eval.py               shared metrics, saving of every run, summary table, significance tests
    utils.py              seeding, device selection, project paths
    dapt.py               optional continued pretraining (stretch goal, not used in the report)
"""
