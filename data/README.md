# Data placement

Download data separately; large datasets are ignored by Git.

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
