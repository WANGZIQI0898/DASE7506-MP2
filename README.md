# DASE7506 MP2 Submission

This repository contains the selected V2 candidate selector, its frozen
checkpoint, the supplied data, reproducible training and evaluation scripts,
and the final experimental report.

## Final result

- Validation accuracy: 93.6090% (747 / 798)
- Non-unanimous validation accuracy: 87.8282% (369 / 419)
- Official test accuracy: 81.37044967880085% (380 / 467)
- Inference assets: 156,915 bytes
- Measured prediction plus scoring time: 0.99 seconds on four CPU threads

## Environment

Use Python 3.12 from the `code` directory:

```bash
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install -r requirements-training.txt
python check_data.py
```

The V2 selector itself requires NumPy. The training script additionally uses
the dependencies listed in `requirements-training.txt`.

## Reproduce training

```bash
cd code
python experiments/train_ensemble_v2.py
python experiments/train_set_mlp.py
python experiments/build_v2_bundle.py
```

The first script trains the listwise linear component. The second trains the
four-candidate set encoder. The third assembles the frozen V2 checkpoint in
`runs/verifier_v2`.

## Evaluate validation

```bash
cd code
python predict.py --method submission_v2.py --checkpoint runs/verifier_v2 \
  --split validation --output runs/validation-predictions.json
python score_predictions.py --split validation \
  --predictions runs/validation-predictions.json \
  --output runs/validation-score.json
```

## Evaluate test

```bash
cd code
python predict.py --method submission_v2.py --checkpoint runs/verifier_v2 \
  --split test --output runs/test-predictions.json
python score_predictions.py --split test \
  --predictions runs/test-predictions.json \
  --output runs/test-score.json
```

The scorer's `accuracy` field should be
`81.37044967880085` for the frozen test predictions.

## Checkpoint

The selected checkpoint directory is:

```text
code/runs/verifier_v2/
```

It contains:

```text
config.json
linear_weights.npz
set_encoder_weights.npz
```

The checkpoint stores all learned weights and feature-normalization statistics
needed for offline CPU inference. It does not open question IDs, reference
answers, or external data during prediction.

## Report

The six-page report is available at:

```text
report/MP2_Experimental_Report.pdf
```


