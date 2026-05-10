# Leakage and validation protocol

## Leakage controls

- All preprocessing steps, including imputation and categorical encoding, are executed inside the cross-validation pipeline.
- Target-derived variables are not used as model inputs.
- Records from the same lot are not split across training and test folds in the lot-grouped validation.
- A forward-chaining validation is performed as a time-series-aware stress test.
- Product-family labels are tested using ablation to distinguish product-identity learning from process-mechanism learning.

## Validation modes to report

1. Random 5-fold CV: retained for direct comparison with the original model results.
2. Lot-grouped 5-fold CV: primary leakage-resistant validation when lot IDs are available.
3. Forward-chaining time split: stress test against temporal leakage from drift, seasonality, raw-material lots, and equipment change.
4. Product-label ablation: tests whether engineered process features remain dominant without the product-family label.

## Manuscript reporting rule

The abstract may report the random 5-fold result only if the Methods and Discussion clearly state that time-aware and lot-grouped validation were also performed. If time-aware performance is materially lower, the time-aware metric should be presented as the conservative deployment metric.
