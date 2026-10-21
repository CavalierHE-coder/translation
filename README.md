# Chinese-to-English Neural Machine Translation (Transformer)

[简体中文](README_zh.md) | English

A compact Chinese → English machine translation project built on PyTorch's `nn.Transformer`. It covers the whole pipeline: data preprocessing, vocabulary building, training, BLEU evaluation, and an interactive command-line translator.

## Features

- Encoder–decoder Transformer with sinusoidal positional encoding
- Character-level tokenization for Chinese, NLTK Treebank word tokenization for English
- Teacher-forced training with padding masks and causal masks
- Greedy autoregressive decoding with batched inference
- Corpus-level BLEU evaluation (NLTK)
- TensorBoard logging of the training loss
- Interactive CLI translation app

## Project Structure

`config.py` sets `ROOT_DIR` to the **parent** of the folder containing the scripts, so the scripts should live in a subfolder (e.g. `src/`):

```
project_root/
├── src/                  # all .py files
│   ├── config.py         # paths, special tokens, hyperparameters
│   ├── tokenizer.py      # ZhTokenizer / EnTokenizer, vocabulary build & load
│   ├── preprocess.py     # read raw data, split, tokenize, save .jsonl
│   ├── dataset.py        # Dataset, collate_fn (padding), DataLoader
│   ├── model.py          # PositionalEncoding + TranslationModel
│   ├── train.py          # training loop, saves best model by training loss
│   ├── evaluate.py       # BLEU evaluation on the test set
│   └── predict.py        # batch inference + interactive CLI
├── data/
│   ├── raw/              # cmn.txt (tab-separated: English \t Chinese [\t ...])
│   └── processed/        # train.jsonl, test.jsonl (generated)
├── models/               # best_model.pt, zh_vocab.txt, en_vocab.txt (generated)
└── logs/                 # TensorBoard logs (generated)
```

> Create `data/raw`, `data/processed`, `models` and `logs` beforehand — the scripts do not create directories automatically.

## Requirements

- Python 3.12+
- PyTorch (a GPU is used automatically if available)
- pandas, scikit-learn, nltk, tqdm, tensorboard

```bash
pip install -r requirements.txt
```

## Data

Place `cmn.txt` (the Tatoeba-style English–Chinese sentence pairs, tab-separated) in `data/raw/`. Only the first two columns are used (`en`, `zh`). The data is split 80% / 20% into train / test with `random_state=42`.

## Usage

Run all commands from inside `src/`.

```bash
# 1. Preprocess: build vocabularies (from the training split only) and encode the data
python preprocess.py

# 2. (Optional) sanity-check the data loaders / model shapes
python dataset.py
python model.py

# 3. Train
python train.py

# 4. Evaluate (BLEU on the test set)
python evaluate.py

# 5. Interactive translation
python predict.py
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir ../logs
```

In the interactive app, type a Chinese sentence and press Enter; enter `q`, `quit` or `exit` to leave.

## Configuration

All settings live in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `BATCH_SIZE` | 64 | Batch size |
| `EPOCHS` | 50 | Number of training epochs |
| `LEARNING_RATE` | 0.001 | Adam learning rate |
| `MAX_SEQ_LEN` | 128 | Max positional-encoding length and max generated tokens |
| `D_MODEL` | 128 | Embedding / hidden size |
| `NUM_HEADS` | 4 | Attention heads |
| `NUM_ENCODER_LAYERS` | 2 | Encoder layers |
| `NUM_DECODER_LAYERS` | 2 | Decoder layers |

## How It Works

1. **Tokenization** – Chinese text is split into single characters; English uses `TreebankWordTokenizer` (and `TreebankWordDetokenizer` for decoding). Vocabularies start with `<pad>`, `<unk>`, `<sos>`, `<eos>`. English targets are wrapped as `<sos> ... <eos>`.
2. **Training** – For each target sequence of length T+1, the decoder input is `targets[:, :-1]` and the label is `targets[:, 1:]`. Cross-entropy ignores `<pad>`. The checkpoint with the lowest average epoch loss is saved to `models/best_model.pt`.
3. **Inference** – The encoder runs once; the decoder starts from `<sos>` and generates one token at a time (argmax) until every sentence emits `<eos>` or `MAX_SEQ_LEN` is reached.
4. **Evaluation** – References have `<sos>`/`<eos>` stripped, and `corpus_bleu` is computed over token-id sequences.

## Notes & Limitations

- Model selection uses **training** loss; there is no separate validation set, so the model may overfit with 50 epochs.
- Decoding is greedy (no beam search).
- Character-level Chinese and a word-level English vocabulary (no subwords) means rare words become `<unk>`.
- `torch.load` in `evaluate.py` / `predict.py` loads the weights on the default device; pass `map_location` if you train on GPU and run on CPU.
- `BLEU` is computed with NLTK's default 4-gram weights without smoothing, so scores on short sentences can be low or zero.

## License

MIT
