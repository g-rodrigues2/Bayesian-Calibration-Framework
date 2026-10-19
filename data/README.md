# Data

Tumor volumes of 4T1 (radiation-sensitive) and RR-4T1 (radiation-resistant) tumors in mice, and the
[89Zr]-CD8 PET standardized uptake values (SUV) of the treated cohorts, from the experiments described in

Song, P. N., DeMellier, C. T., Lynch, S. E., Gallegos, C. A., Mansur, A., Moye, J. A., Mascioni, A., Jia, F.,
Willey, C. D., Sorace, A. G. CD8 immunoPET imaging to stratify response and guide combination
immunotherapy and radiation in triple negative breast cancer. *Breast Cancer Research* (2026),
https://doi.org/10.1186/s13058-026-02286-9.

## `processed/`

One file per mouse with two columns: day and tumor volume (mm³). File names encode the group, cohort,
mouse, treatment days and, for the immunotherapy cohorts, the SUV:

```
{group}_c{cohort}_m{mouse}_t{day}_..._t{day}[_SUV{value}].txt
```

| Group | Mice | Treatment days in the name | Used in |
|---|---|---|---|
| `control_sensitive` | 9 | t0–t5 (not used: no treatment) | stages 1–2 |
| `control_resistant` | 5 | t0–t5 (not used) | stages 1–2 |
| `radiation_sensitive` | 15 | t0–t5: 2 Gy fractions | stages 3–4 |
| `radiation_resistant` | 9 | t0–t5: 2 Gy fractions | stages 3–4 |
| `io_sensitive` | 3 | t0, t3, …, t18: immunotherapy | stage 5 |
| `rad+io_sensitive` | 15 | t0–t5: radiotherapy (immunotherapy on days 9, 12, …, 24) | stage 6 |

The loaders do not parse the treatment days from the names; each stage sets the schedule of the
experiment by default (radiotherapy days 0–5, immunotherapy days 0–18 or 9–24) and `CALIB_DOSE_DAYS`
overrides it.

## `prepare_data.ipynb`

Notebook that converts the spreadsheets of the experimental study (`raw/`, not distributed) into the
files of `processed/`.
