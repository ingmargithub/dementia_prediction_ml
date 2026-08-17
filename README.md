# Dementia prediction with regional DAWM

Primary analysis: incident dementia as a time-to-event outcome using an unpenalized Cox proportional-hazards model.

Outcome: `DEM_DEMENTIA` with follow-up/censoring time `DEM_TIME_IN_DAYS`. Baseline dementia is excluded.

Two prespecified models use identical cross-validation folds:

1. Base model: clinical, vascular/lifestyle, SVD, WMH, ICV, and baseline cognition.
2. Regional DAWM model: base model + `FrontalDAWM`, `TemporalDAWM`, `ParietalDAWM`, `OccipitalDAWM`.

The primary out-of-sample metrics are Harrell's C-index and IPCW C-index, evaluated using the observed survival follow-up. The main incremental quantity is the change in these metrics after adding regional DAWM.

The final fitted Cox models are also compared using the nested-model likelihood-ratio test. Hazard ratios and confidence intervals for the final regional DAWM model should be extracted from the fitted Cox coefficients as a separate inferential reporting step.
