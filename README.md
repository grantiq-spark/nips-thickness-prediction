# Feature interaction engineering for NIPS membrane thickness prediction

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20112096.svg)](https://doi.org/10.5281/zenodo.20112096)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Public reproducibility repository for the revised manuscript:

**"Feature interaction engineering with explainable AI for viscosity-speed coupling in NIPS membrane thickness prediction from pilot manufacturing data"**
(DWT-D-26-00307)

This repository contains the anonymized public dataset, the analysis scripts, the data dictionary, the leakage and validation protocol, and the locked exact-TreeSHAP and GBR-500 performance summary outputs that support the results reported in the manuscript.

## Repository structure

```
.
|-- data/
|   `-- public/
|       `-- Public_760_Anonymized.csv     (760 records; anonymized public dataset)
|-- docs/
|   |-- data_dictionary.csv               (column-by-column description, units, anonymization notes)
|   `-- leakage_and_validation_protocol.md (validation strategies and leakage controls)
|-- results/
|   |-- GBR500_performance_EXACT.csv      (locked GBR-500 random 5-fold CV metrics)
|   `-- SHAP_global_importance_EXACT.csv  (locked TreeSHAP global importance ranking)
|-- src/
|   |-- pdp_ice_analysis.py               (full reproducibility pipeline)
|   `-- figure_style_template.py          (white-background, 300 dpi figure style)
|-- requirements.txt
`-- README.md
```

## Reproducibility workflow

```bash
pip install -r requirements.txt

python src/pdp_ice_analysis.py --input data/public/Public_760_Anonymized.csv --out results --bootstrap 30
```

## Locked exact values reported in the manuscript

| Metric | Value | File |
|---|---|---|
| GBR-500 R^2 (random 5-fold CV) | 0.7355 | results/GBR500_performance_EXACT.csv |
| GBR-500 MAE (random 5-fold CV) | 6.70 um | results/GBR500_performance_EXACT.csv |
| GBR-500 RMSE (random 5-fold CV) | 14.22 um | results/GBR500_performance_EXACT.csv |
| Top SHAP feature: viscosity x speed | 29.16% relative mean abs SHAP | results/SHAP_global_importance_EXACT.csv |

## Data Availability

The anonymized public dataset and analysis scripts required to reproduce the reported model-performance and SHAP summary outputs are provided in this repository.

The public dataset is anonymized and excludes commercially sensitive identifiers; the internal clean dataset used for audit contains controlled metadata not released publicly. Lot identifiers in the public release are replaced by integer codes, product family labels are anonymized to Family_1 through Family_11, and production dates are retained because the temporal validation result reported in the manuscript depends on chronological order.

## License

Source code under MIT License. Dataset under CC BY 4.0 with attribution to UMTR Co., Ltd.

## Citation

When using this dataset or code, please cite the manuscript (DWT-D-26-00307, Park et al., Desalination and Water Treatment, in revision) and the permanent Zenodo archive of this repository:

> DOI: [10.5281/zenodo.20112096](https://doi.org/10.5281/zenodo.20112096) (release tag `v11.2`)
> 
