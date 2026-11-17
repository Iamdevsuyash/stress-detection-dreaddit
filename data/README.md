# Data

`python -m src.data` downloads Dreaddit from the authors' site
(<http://www.cs.columbia.edu/~eturcan/data/dreaddit.zip>) into `data/raw/` and writes:

| file | rows | notes |
|---|---|---|
| `splits/train.csv` | 2555 | official train minus validation posts |
| `splits/val.csv` | 283 | 10% of train, stratified by label, **grouped by Reddit post** |
| `splits/test.csv` | 715 | official test set (shares no posts with train) |

Dreaddit labels 5-sentence *segments*; one Reddit post can produce several segments
(2,838 train rows come from 2,343 posts). The validation split therefore keeps all segments of a post
on the same side, as the official test split does. The chosen validation posts are listed in
`splits/val_post_ids.txt`, which is committed so everyone gets an identical split.

Raw data and the split CSVs are git-ignored: we do not redistribute Reddit text.

GloVe (`glove.6B.100d.txt`) is downloaded automatically into `data/glove/` the first time
`train_lstm.py` runs (from the HuggingFace mirror of the official Stanford zip, `stanfordnlp/glove`).
