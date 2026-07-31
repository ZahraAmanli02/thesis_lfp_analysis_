## Main Research Questions

The thesis was organized around three main research questions.

### RQ1 — Diet Classification

Can dietary group be decoded from electrophysiological features extracted from a single recording?

Diet classification was evaluated using feature-specific and estrous-phase-specific machine-learning models. Mouse-level cross-validation was used to prevent recordings from the same animal from appearing in both the training and test datasets.

### RQ2 — Body-Weight Prediction

Can body weight be predicted from electrophysiological features extracted from a single recording?

Regression models were evaluated using the same feature-specific and phase-specific framework applied to diet classification. Body weight was used as the prediction target and was not included as an input feature.

### RQ3 — Relationship Between Diet and Body-Weight Information

Are dietary group and body weight represented by the same electrophysiological features, or by distinct oscillatory patterns?

The significant feature cells identified for diet classification and body-weight prediction were compared across frequency bands, band-power ratios, estrous phases, and recording cables.

This analysis examined whether diet-related information could be explained by body weight alone or whether the two variables were associated with distinct electrophysiological signatures.

## Cross-Cable Reproducibility

Cable 1 and Cable 3 were analysed independently throughout the study.

Cross-cable comparison was not treated as a separate research question. Instead, it was used as a reproducibility framework for all three research questions.

Reproducibility was evaluated based on:

- the sparsity of significant feature cells;
- the frequency bands containing the strongest signals;
- estrous-phase dependence;
- classification and regression performance;
- the separation between diet-related and body-weight-related features.

The exact identity of significant feature cells was not required to be identical across the two cables. Reproducibility was interpreted at the level of the broader biological and analytical pattern.
