# FYP: Fake News Detection via LLM-Generated Synthetic Data

Implementation of the methodology in Chapter 3. Trains three model families
(LR/SVM, CNN, BERT) under three data compositions and evaluates cross-domain
generalization (train on ISOT, test on LIAR).

## Requirements

- Python 3.10+
- RTX 4060 (8GB VRAM) or similar recommended for CNN/BERT training; everything
  also runs on CPU, just slower

## Setup

```bash
# 1. Create a virtual environment
python -m venv venv
# Windows:  venv\Scripts\activate
# Linux:    source venv/bin/activate

# 2. Install PyTorch with CUDA (check https://pytorch.org for your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Install everything else
pip install -r requirements.txt

# 4. Set your OpenAI key (only needed if regenerating synthetic data)
# Windows:  set OPENAI_API_KEY=sk-...
# Linux:    export OPENAI_API_KEY=sk-...
# Never commit .env or share it -- it holds your real API key.
```

## Data download (manual)

Put these in `data/raw/`:

- **ISOT**: `True.csv` and `Fake.csv` from
  https://www.uvic.ca/ecs/ece/isot/datasets/fake-news/index.php
  (or the Kaggle mirror "ISOT Fake News Dataset")
- **LIAR**: `train.tsv`, `test.tsv`, `valid.tsv` from
  https://huggingface.co/datasets/liar (or the original UCSB release)
- **WELFake**: `WELFake_Dataset.csv` from the Kaggle dataset "WELFake"

`data/synthetic/` (the LLM-generated corpus) and `data/processed/` (the built
training/test splits) are already included — no need to regenerate them.
`models/` (all trained checkpoints) and `results/` (all scores and the report)
are also included, so the project can be inspected without running anything.

## View the results

Open `results_report.html` in any browser. It is fully self-contained --
no server, no data files needed.

## Running the pipeline from scratch (optional)

Only needed if you want to regenerate everything yourself. Each step depends
on the ones before it.

```bash
# --- Core pipeline ---
python src/load_data.py
python src/eda.py
python src/generate_synthetic_fake.py --n 500
python src/generate_synthetic_real.py --n 1000
python src/build_datasets.py core
python src/build_datasets.py test-sets
python src/evaluate.py leakage
python src/build_datasets.py controls
python src/train.py --model all
python src/evaluate.py master

# --- Cross-domain checks (WELFake) ---
python src/evaluate.py cross-target --dataset welfake --comp real_real half_synthetic full_synthetic synthetic_real_only both_synthetic
python src/evaluate.py cross-target --dataset welfake_clean --comp real_real half_synthetic full_synthetic synthetic_real_only both_synthetic

# --- Length-confound control ---
python src/generate_synthetic_fake.py --n 500 --lengths short medium long
python src/build_datasets.py core
python src/train.py --model all --dataset synthetic_length_controlled
python src/evaluate.py cross-target --dataset welfake_clean --comp full_synthetic synthetic_length_controlled

# --- Authorship-shortcut control ---
python src/evaluate.py edit-distance
python src/generate_synthetic_real.py --n 500 --symmetric --max_retries 4
python src/generate_synthetic_fake.py --n 500 --symmetric --max_retries 4
python src/build_datasets.py controls
python src/train.py --model all --dataset synthetic_real_only_matched both_synthetic_matched
python src/evaluate.py cross-target --dataset welfake_clean --comp both_synthetic both_synthetic_matched

# --- Statistical validation ---
python src/evaluate.py significance --dataset liar
python src/evaluate.py significance --dataset welfake_clean

# --- Cross-validation, LR/SVM only ---
python src/train.py --model lr_svm --cv 5 --dataset real_real half_synthetic full_synthetic style_robust

# --- Synthetic-fraction sweep ---
python src/build_datasets.py sweep
python src/run_swap_sweep_experiment.py

# --- Seed-stability check ---
python src/run_multiseed_robustness.py

# --- Style/tone-attack robustness ---
python src/generate_style.py attack
python src/eval_style_robustness.py
python src/generate_style.py counter-training
python src/build_datasets.py style-robust
python src/train.py --model all --dataset style_robust
python src/generate_style.py attack-reverse
python src/eval_style_robustness.py --pair reverse

# --- Diverse-source synthetic data ---
python src/generate_synthetic_fake_liar.py --n 200
python src/build_datasets.py multisource
python src/train.py --model all --dataset multisource

# --- Rebuild the report (run after any experiment that changes results/) ---
python src/detector.py export
python src/detector.py verify
python src/build_report.py
```
