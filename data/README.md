# Data placement

Download data separately; large datasets are ignored by Git.

Single-stream training configs use native-resolution PNG caches in
`data/native_frames/original` (or `no_leakage` for corrected splits). The
supplied original JPEGs are 224 x 224, while the source AVI videos are 320 x 240.
Prepare full-resolution frames with FFmpeg/ffprobe available on PATH:

```bash
python -m video_classification.prepare_frames --root data/ucf101/ufc10 --output-root data/native_frames/original --splits train val
```

This creates ten evenly spaced lossless PNGs per video without resizing,
preserves all supplied files, and records sampled indices in `sampling.json`.
It reads only the selected splits; test data requires an explicit `--splits test`.
The original JPEG sampling indices are unknown, so document this resampling
when comparing against earlier runs. Set `dataset.frame_source = "provided"`
to reproduce loading from the original JPEGs. `dataset.image_size = "native"`
alone means no loader resize; it cannot recover detail already absent from a file.

```text
data/
    course/datasets.py       The loader supplied on DTU Learn for 2026
    ucf101/                  Extracted original dataset (preserve its layout)
    ucf101noleakage/          Extracted corrected dataset and optical flow
```

These root names are local conventions. Update the TOML files if your folders
differ. The original data and loader are supplied through Learn. The brief also
lists `/dtu/datasets1/02516` as the location of the corrected dataset and its zip.

Before implementing the adapter, inspect CSV filenames and columns, relative
path resolution, numeric labels, frame order, and the course loader's arguments.
Keep the supplied splits. Audit exact video overlap and subject/group overlap;
use reliable metadata for subject identity rather than assuming filenames prove
identity. The original splits are intentionally used for the first comparison.

For corrected data, inspect flow storage, dtype, scale, temporal alignment, and
whether available files hold vectors or color encodings. Do not assume 10 flow
fields just because there are 10 RGB frames. Apply spatial transforms consistently
across each clip and its paired flow; vector-valued flow needs appropriate vector
updates when flipped or resized.

## Dual-stream flow pairing

The cluster layout is `/dtu/datasets1/02516/ucf101_noleakage`, containing:

```text
metadata/<split>.csv
frames/<split>/<class>/<video>/frame_1.jpg ... frame_10.jpg
flows_png/<split>/<class>/<video>/flow_1_2.png ... flow_9_10.png
```

Use `dataset.frame_source = "provided"` with these precomputed flows. The loader
uses `video_path` in the existing metadata to locate each video's nine flow
files in numeric pair order. No flow computation or additional CSV manifests
are required. The directory is `flows_png` with an underscore, as confirmed by
the cluster listing. Training accesses only train/val. The filename layout is
verified; inspect a PNG or run a smoke test to check the actual channel format
and dimensions. The supported representations are:

- `rgb_png`: actual RGB color visualizations stored as PNG, decoded as three
  channels and scaled by 1/255. Non-RGB PNGs are rejected. Packed-vector PNGs
  are unsupported; RGB mode alone cannot distinguish those from visualizations,
  so verify the encoding before selecting this setting.
- `uv_npy`: floating-point NumPy arrays `[H, W, 2]` containing horizontal and
  vertical displacements in pixels, in that order. Set a finite positive
  `dataset.flow_scale` (for example `1.0` to retain pixel units). This is a fixed
  divisor, not a statistic fitted using validation/test data. Resizing scales
  horizontal vectors by new_width/old_width and vertical vectors by
  new_height/old_height before division. Non-finite arrays are rejected.

Each flow must share the source spatial dimensions of its paired RGB frames.
All frames within a video must have the same dimensions. With native loading,
clips in a batch also need matching dimensions; the default batch size is 1.

### Optional manifests for custom data

For a custom layout, `uv_npy` arrays, or native-cache flow, explicitly set
`dataset.flow_manifest_root` to a folder with `train.csv` and `val.csv`. This
overrides automatic course-layout loading. Each row associates a flow file with
its RGB endpoints:

```csv
video_name,from_frame,to_frame,flow_path
example_video,1,2,flows_png/train/example_class/example_video/flow_1_2.png
example_video,2,3,flows_png/train/example_class/example_video/flow_2_3.png
```

The example class/video names above illustrate the optional manifest schema.
Use video names from the existing split metadata and the actual flow paths
relative to `dataset.root`. Frame ordinals are one-based and correspond to
`frame_1.jpg`, `frame_2.jpg`, etc. Include each consecutive pair exactly once:
1->2 through 9->10 for ten RGB frames. Each video therefore has nine flow fields;
no zero-motion last field is invented. CSV row order is irrelevant. Paths must
stay within the dataset root, and flow files cannot be reused between pairs or
between training and validation. Custom filenames require a verified
flow-generation naming/index convention.

For `frame_source = "native_video"`, set `flow_manifest_root`, use flows computed
for the cached RGB pairs, and add `source_from_index` and `source_to_index`
columns. Their values
must match adjacent entries of `frame_indices_zero_based` in the cache's
`sampling.json`. These indices refer to the original video and are zero-based.
A missing or mismatched source-index manifest is rejected. This check relies on
accurate metadata; it cannot prove that someone computed a flow from the claimed
frames. Supplied JPEGs and a newly sampled native cache are not interchangeable.

Only create/read a test flow manifest when performing final test evaluation;
training reads train/val only. The corrected dataset and flow manifests are
external inputs, excluded from Git. They are not present in this local checkout.
