# Report draft

Verified export: `report-draft.pdf`; editable source: `main.tex`.
The supplied CVPR format is preserved. There is no abstract. Main text,
title, tables, and figures must occupy at most three pages; references and
the AI-use statement start on a separate page. The source flushes floats and
raises a LaTeX error if the main content exceeds three pages.

## Cluster evidence now included

The report uses the four completed runs from the 21 October download under
`../hpc-results-20261021-100431/outputs/`, each configured for a maximum of
50 epochs, seed 42, 224 x 224 RGB input, Adam at 0.001, and validation-loss
early stopping (patience 5, minimum improvement 0.0001).

| Model | Run directory suffix | Selected/completed epochs | Validation accuracy | Loss at selected epoch |
| --- | --- | --- | --- | --- |
| Per-frame | 01_per_frame_seed42/20261021T072135_823976Z | 22/31 | 90.8% | 0.5671 |
| Late fusion | 02_late_fusion_seed42/20261021T073116_763728Z | 46/50 | 75.0% | 0.9692 |
| Early fusion | 03_early_fusion_seed42/20261021T074608_924815Z | 18/18 | 22.5% | 2.1038 |
| 3D CNN | 04_cnn3d_seed42/20261021T074842_549366Z | 22/23 | 32.5% | 1.9499 |

Selection uses maximum validation video accuracy, with lower validation loss
breaking ties. Early stopping instead monitors validation loss. The table's
accuracy and loss both come from the accuracy-selected epoch, not from
separate best epochs. The per-frame minimum loss is 0.4218 at epoch 26,
where accuracy is 88.3%; it must not be paired with the epoch-22 peak accuracy.

`build_assets.py` names every run explicitly and checks each summary against
its full history, including both checkpoint selections, sample counts,
settings, and stopping status. `generated/evidence.json` records run paths,
source-file hashes, metric definitions, selected and completed epochs,
parameter counts, optimizer updates, elapsed time, and device information.
The result table and figure are regenerated from the logs.

The single-panel figure `figures/train_val_loss.pdf` overlays eight curves.
Model identity is encoded by colour; solid lines show training and dashed
lines show validation. Curves are not smoothed or extrapolated after stopping.
For per-frame aggregation, training loss is frame CE while validation is
video NLL after probability averaging; their gap is not directly comparable.
The other models use video CE for both splits. Training losses average batches
during updates; validation uses end-of-epoch weights.

The older 20-epoch-budget cluster jobs, empty failed runs, laptop pilots,
GroupNorm pilot, and warm-start trial are excluded. Some earlier cluster
histories repeat the same-seed trajectories; they are not independent trials.
Old unused pilot assets may remain in the figures/generated directories;
only `train_val_loss.pdf` and `baseline_table.tex` are included by the source.

Metadata counts are 500/120/120, totaling 740; the course brief's stated total
of 720 is inconsistent with those counts. The report uses the split sizes.
Parameter counts including biases are 94,538 per-frame/late fusion,
102,314 early fusion, and 280,586 3D CNN.

## Still needed before submission

1. Complete Task 4.2.1 on the corrected splits using per-frame aggregation,
   selected by original-split validation accuracy. Match the training recipe
   and report actual corrected split counts and measured changes.
2. Complete Task 4.2.2 and document the actual flow encoding, architecture,
   training, and fusion method; compare with the corrected-split RGB baseline.
3. Evaluate held-out test data after freezing model selection. The downloaded
   runs provide training/validation histories, not test measurements.
4. Replace the remaining draft subsections, then update the final conclusion.
5. Confirm author/group details and review the AI-use statement, including
   the author's checks and any other tools used.

## Rebuild and layout checks

From this folder, run `../.venv/Scripts/python.exe build_assets.py`, then
compile `main.tex` using the available LaTeX tools. On other systems use a
Python environment with matplotlib. A normal build is
`latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`.

The verified export is built in the workspace's `tmp/pdfs/report-draft/build/`
with job name `report-draft`, avoiding concurrent automatic builds of
`main.pdf`. It uses pdflatex, BibTeX, and enough repeat passes for references
to settle, then copies the PDF to this folder. If rebuilding the separate
export, run BibTeX from its build directory and point BIBINPUTS/BSTINPUTS to
this report folder; BibTeX disallows writes through parent-relative paths.

Check `REPORT_MAIN_PAGES` in the log and render the PDF after edits. Do not
reduce fonts, margins, or spacing to meet the page limit. The current export
has three main-content pages plus one page for references and AI disclosure.
