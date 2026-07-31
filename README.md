# LFP Analysis of Diet and Body Weight in the Lateral Hypothalamus

This repository contains the data-processing, spectral-analysis, statistical-modelling, machine-learning, and visualization code used for a thesis project investigating oscillatory activity in the lateral hypothalamus of mice exposed to different diets.

The project examines whether electrophysiological activity derived from the lateral hypothalamus contains distinguishable signatures of:

- dietary group: high-fat diet versus control diet;
- body weight;
- estrous-cycle phase;
- recording cable and channel configuration.

The repository also contains processed data, analysis outputs, and numerical source data used to generate the figures presented in the thesis.

## Research Overview

Electrophysiological recordings were collected from the lateral hypothalamus using two independent recording cables, referred to as **Cable 1** and **Cable 3**.

The main analysis pipeline included:

1. preparation of recording information and experimental metadata;
2. filtering according to the study period and quality-control criteria;
3. generation of channel-to-channel differential signals;
4. estimation of power spectral density using a multitaper method;
5. extraction of absolute and relative band-power features;
6. calculation of pairwise band-power ratios;
7. statistical analysis using mixed-effects models;
8. classification of dietary group from individual recordings;
9. prediction of body weight from electrophysiological features;
10. permutation and bootstrap analyses;
11. generation of thesis figures and source-data tables.

All three research questions were investigated separately in Cable 1 and Cable 3 to assess the reproducibility of the findings across independent recording configurations.

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

## Repository Structure

```text
thesis_lfp_analysis_/
│
├── data/
│   └── Input, metadata, intermediate, and processed data
│
├── outputs/
│   └── Statistical results, model outputs, diagnostic plots,
│       bootstrap results, thesis figures, and source-data tables
│
├── scripts/
│   └── Python scripts implementing the complete analysis workflow
│
└── README.md
