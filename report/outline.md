# Report outline

Use the CVPR template supplied by the course. This file is a planning outline,
not a replacement template or a completed report.

## Requirements from the brief

- One PDF per group, submitted to DTU Learn.
- Three pages including all tables and figures.
- References and the mandatory statement on generative AI use may be outside
  those three pages.
- Cover all tasks in the same report.

## Suggested structure

1. **Data and experimental setup**: task, class mapping, preprocessing, video
   sampling, splits, seeds, optimization, and validation protocol.
2. **2D and 3D approaches (4.1)**: per-frame aggregation, late fusion, early
   fusion, and 3D CNN. Explain the temporal input and aggregation for each.
3. **Leakage comparison (4.2.1)**: describe the overlap issue and compare the same
   selected model on original and corrected data with a controlled protocol.
4. **Dual stream (4.2.2)**: RGB/flow representations, architecture, fusion, and
   comparison against the single-stream model on corrected data.
5. **Discussion**: conclusions supported by measurements, failure cases, and
   limitations. Do not fill in results before experiments have been run.

## Planned evidence

| Experiment | Split regime | Validation accuracy | Final test accuracy | Notes |
| --- | --- | --- | --- | --- |
| Per-frame aggregation | Original | TBD | TBD | |
| Late fusion | Original | TBD | TBD | |
| Early fusion | Original | TBD | TBD | |
| 3D CNN | Original | TBD | TBD | |
| Selected model | Corrected | TBD | TBD | Same model/protocol for comparison |
| Dual stream | Corrected | TBD | TBD | RGB + optical flow |

Consider one compact model-comparison table, learning curves, and a confusion
matrix or illustrative failures if space permits.

## Generative AI use statement — adapt before submission

Record tools used, purposes, affected parts, and how outputs were checked.
Initial entry: Codex assisted with the project directory scaffold, configuration
files, implementation TODOs, exploration notebook, report outline, and the
per-frame CNN with video probability aggregation, data adapter, training command,
validation checkpoint selection, late fusion using the same CNN architecture,
early fusion with stacked RGB channels, the 3D CNN, video-level training,
optional GroupNorm comparison configurations, per-frame-to-late-fusion
checkpoint initialization, training-run analysis, validation-loss early stopping,
separate loss-selected checkpoints, and
automated checks. Revise this record as the work progresses and follow the
course's required wording.
