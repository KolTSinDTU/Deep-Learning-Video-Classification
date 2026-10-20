# Training review — 19 October 2026

The clearest opportunity is to reuse the features the per-frame model has
already learned when training late fusion. The other video models are still
learning slowly, so test optimization changes before making them larger.

## What the saved runs show

All four main runs completed 20 epochs at 112 pixels, seed 42, using the
original 500-video training and 120-video validation splits.
Smoke tests and the earlier two-epoch per-frame trial are excluded below.

| Model | Final training accuracy | Best validation video accuracy | Best epoch | Updates per epoch |
| --- | ---: | ---: | ---: | ---: |
| Per-frame | 99.1% of frames | 91.7% | 19 | 625 |
| Late fusion | 47.2% of videos | 40.8% | 20 | 250 |
| Early fusion | 36.0% of videos | 30.8% | 20 | 250 |
| 3D CNN | 63.0% of videos | 49.2% | 19 | 500 |

The per-frame training metric and video metrics use different units. Do not
treat their differences as a directly comparable generalization gap.
The fusion models have low training accuracy as well as low validation accuracy;
their curves do not establish that insufficient regularization is the main
problem. Their loss is still falling at epoch 20.

Twenty epochs is also a different optimization budget for each architecture:
the per-frame model processes 5,000 labeled frames in batches of 8, whereas
late and early fusion process 500 videos in batches of 2. Record optimizer
updates and runtime alongside epoch counts. More updates alone are not
guaranteed to close the gap because the training losses differ too.

## Reuse the trained encoder in late fusion

The two models have exactly the same parameter shapes. Per-frame evaluation
averages frame probabilities; late fusion averages features before the linear
classifier, which is equivalent to averaging frame logits.

Loading the best per-frame checkpoint into late fusion and evaluating all
120 validation videos gave **91.7% accuracy and 0.4934 video cross-entropy**,
with zero additional optimizer steps. The source is the epoch-19 checkpoint
from the 20-epoch per-frame run. This is evidence that useful features already
exist and that the 40.8% result does not establish a limitation of mean fusion.

The two-epoch fine-tuning trial then produced:

| Stage | Validation accuracy | Video cross-entropy |
| --- | ---: | ---: |
| Loaded per-frame weights, before fine-tuning | 91.7% | 0.4934 |
| Fine-tuning epoch 1 | 91.7% | 0.4508 |
| Fine-tuning epoch 2 | 90.8% | 0.4717 |

The trainer saved epoch 1 as the best checkpoint. Fine-tuning reduced loss at
that checkpoint without increasing accuracy; the second epoch demonstrates
that more training is not automatically better. These figures do not establish
a gain over the per-frame model's 91.7% accuracy.

The new `training.init_checkpoint` / `--init-checkpoint` option loads only
model parameters, then starts a fresh optimizer. The warm-start configuration
uses a learning rate of `0.0001`, one tenth of the original rate, as an initial
fine-tuning trial. Source checkpoint identity and epoch are saved with the run.
It checks labels, architecture, normalization, and split regime before training.

```powershell
python -m video_classification.train --config configs/tuning/02_late_fusion_warmstart.toml
```

Run from this project with the environment activated. This starts 10 new
fine-tuning epochs from the per-frame checkpoint each time. It does not resume
the latest late-fusion run. Copy the source checkpoint to the cluster and set
its path before using this configuration there.

This experiment benefits from the earlier per-frame training. Report both
stages; it is not an equal-budget, from-scratch architecture comparison.
For the corrected-split experiment, train the source encoder on corrected
training data too. The initializer rejects transfer across split regimes.

## Optional GroupNorm experiment

`model.normalization = "group"` adds eight-group normalization between each
convolution and ReLU, for any of the four models. It normalizes groups of
features within each sample, avoiding dependence on batch statistics.
This makes it a candidate for the current small video batches.
[Group Normalization paper](https://openaccess.thecvf.com/content_ECCV_2018/html/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.html).

A controlled late-fusion trial changed only normalization: same initial
convolution/classifier weights, seed, learning rate, resolution, batch size,
frame sampling, and splits.

| First five epochs | Best validation accuracy | Validation loss at epoch 5 |
| --- | ---: | ---: |
| Original late fusion | 17.5% | 2.1817 |
| Late fusion with GroupNorm | 20.8% | 2.2254 |

The accuracy gain is four validation videos and the epoch-five loss is worse.
One short run is not enough to establish a reliable improvement. Keep this as
an option rather than replacing all baseline configurations.

Four separate configurations under `configs/tuning/` allow further trials.
For example:

```powershell
python -m video_classification.train --config configs/tuning/03_early_fusion_groupnorm.toml
python -m video_classification.train --config configs/tuning/04_cnn3d_groupnorm.toml
```

These start fresh 20-epoch runs at 112 pixels. They do not load pretrained
weights. Compare them against the existing 20-epoch runs before choosing a
recipe, and confirm promising changes with additional seeds. If moving to
224 pixels or changing batch size on the cluster, record that as another
experimental change.

## Evidence and verification

The four original best checkpoints still load. Automated tests cover
normalization gradients with a batch of one, independence from other samples,
unchanged baseline initialization, checkpoint reconstruction, exact weight
transfer, fresh optimizer creation, and rejection of incompatible sources.
All 36 tests pass.

Local figures and exact run paths are in `outputs/model_review_20261019/`.
The recorded run directory for the fine-tuned checkpoint is
`outputs/tuning/02_late_fusion_warmstart_seed42/20261019T154208_367713Z/`.
These are original-split validation results. The assignment's corrected-split
comparison remains necessary; no test-split metrics were used in this review.
