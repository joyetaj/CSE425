# CSE425 — Unsupervised Neural Network for Multi-Genre Music Generation

Course project for **CSE425 (Neural Networks)**, BRAC University.

The project builds an unsupervised symbolic-music (MIDI) generation pipeline around four models of increasing capability:

1. **LSTM Autoencoder** — learns compact representations of music.
2. **Variational Autoencoder (VAE)** — samples and interpolates in a regularized latent space.
3. **Transformer decoder** — autoregressive generation over long horizons.
4. **RLHF stage** — aligns the pretrained generator using a reward function.

Training data is the [MAESTRO](https://magenta.tensorflow.org/datasets/maestro) MIDI corpus (not included; download it separately). Full details are in the [report](report/final_report.pdf).

## Repository structure

| Folder | Contents |
|--------|----------|
| [`preprocessing/`](preprocessing) | `midi_parser.py` (MIDI → piano roll), `tokenizer.py` (MIDI → event tokens) |
| [`Models/`](Models) | `autoencoder.py`, `vae.py`, `transformer.py` |
| [`Training/`](Training) | `config.py`, `train_ae.py`, `train_vae.py`, `train_transformer.py`, `rlhf.py`, `generate_all.py`, `complete_run.py` |
| [`Evaluation/`](Evaluation) | `metrics.py` — evaluation metrics for generated music |
| [`Generation/`](Generation) | `baselines.py` — baseline generators for comparison |
| [`notebooks/`](notebooks) | `demo_context.ipynb` — demo notebook (uses a small synthetic stand-in corpus) |
| [`report/`](report) | Final report (PDF) and LaTeX source |

## Setup

```bash
pip install -r requirements.txt
```

## Known limitations

- **Imports and paths:** Some scripts use `import src.config` and `from models...`, which do not match this folder layout. The demo notebook works around this by adding the project folders to `sys.path`; the notebook is the easiest way to run the pipeline.
- **Dataset path:** `Training/config.py` sets `RAW_MIDI_DIR` to a Windows path from the author's machine. Change it to wherever you downloaded MAESTRO.
- **"Multi-genre":** MAESTRO is almost entirely classical piano, so "genre" here means composer/style conditioning within classical music (see the report).
