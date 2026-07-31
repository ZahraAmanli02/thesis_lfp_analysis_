# LFP Analysis of Diet and Body Weight in the Lateral Hypothalamus

This repository contains the data-processing, spectral-analysis, statistical-modelling, machine-learning, and visualization code used for a thesis project investigating oscillatory activity in the lateral hypothalamus of mice exposed to different diets.

The project examines whether electrophysiological activity derived from the lateral hypothalamus contains distinguishable signatures of:

- dietary group: high-fat diet versus control diet;
- body weight;
- estrous-cycle phase;
- recording cable and channel configuration.

The repository also contains processed data, analysis outputs, and numerical source data used to generate the figures presented in the thesis.

## Research Overview

Electrophysiological recordings were collected from the lateral hypothalamus using two independent recording cables, referred to as **Cable 1** and **Cable 3**. The analytic sample comprises **208 clean recordings from 19 female mice (10 high-fat, 9 control)** across the four estrous phases, pooled at the feature level.

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

Diet classification was evaluated using feature-specific and estrous-phase-specific machine-learning models. Mouse-level resampling was used to prevent recordings from the same animal from appearing in both the training and test datasets.

### RQ2 — Body-Weight Prediction

Can body weight be predicted from electrophysiological features extracted from a single recording?

Regression models were evaluated using the same feature-specific and phase-specific framework applied to diet classification. Body weight was used as the prediction target and was not included as an input feature. A complementary within-mouse analysis tested whether each animal's own feature trajectory tracks its own weight change.

### RQ3 — Relationship Between Diet and Body-Weight Information

Are dietary group and body weight represented by the same electrophysiological features, or by distinct oscillatory patterns?

The significant feature cells identified for diet classification and body-weight prediction were compared across frequency bands, band-power ratios, estrous phases, and recording cables.

This analysis examined whether diet-related information could be explained by body weight alone or whether the two variables were associated with distinct electrophysiological signatures.

## Key Findings

The lateral-hypothalamic LFP carries **two dissociable, phase-conditional signatures** — one for the diet manipulation and one for the resulting weight gain.

- **Diet is decodable under an estrus-conditional delta-family signature (RQ1).** The single feature cell whose 95% confidence interval clears chance is `fast_gamma / delta` (bootstrap-mean balanced accuracy 0.679, 95% CI [0.513, 0.862]), independently corroborated by a significant delta-band mixed-effects effect in Cable 1 (β = −0.135, p_FDR = 0.033).
- **Body weight is not recoverable across mice, but is recoverable within mice (RQ2).** Cross-mouse regression failed in all 84 phase × feature-cell combinations (bootstrap-mean R² between −1.3 and −13), symmetrically across diet groups — a signature of between-mouse baseline-weight variance dominating the within-mouse signal. A within-mouse Spearman analysis recovers a coherent gamma-anchored cluster: four gamma / beta and gamma / gamma ratios reach sign-test significance, with `fast_gamma / beta` rising with weight in all 9 of 9 mice (median r = +0.365, sign-test p = 0.004).
- **The diet and weight signatures do not overlap (RQ3).** No phase × feature-cell combination carries both signals simultaneously, so the diet signal in the LFP is not a downstream consequence of the diet's effect on body weight.

A **methodological takeaway** generalizes beyond this dataset: when between-subject variance in the target is comparable to or larger than the within-subject signal, cross-subject regression cannot recover the effect regardless of model class or sample size — reframing the analysis at the within-subject level is the productive move.

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
```

## Analysis Workflow

The complete analysis workflow is implemented in the `scripts/` directory.

The numbered filenames generally reflect the order of the analysis pipeline. Some scripts represent diagnostic checks, sensitivity analyses, alternative model specifications, bootstrap procedures, or figure-generation utilities and therefore do not need to be executed in every analysis run.

The workflow covers the following main stages:

1. recording inventory and metadata preparation;
2. study-period filtering and quality control;
3. channel-to-channel differential signal generation;
4. multitaper power spectral density estimation;
5. absolute and relative band-power extraction;
6. band-power ratio calculation;
7. mixed-effects modelling and statistical diagnostics;
8. diet classification and body-weight prediction;
9. permutation and bootstrap analyses;
10. figure generation and source-data export.

## Electrophysiological Signal

The input to the spectral and machine-learning analyses is the **channel-to-channel differential signal**, calculated as the difference between two neighbouring recording channels.

This differential configuration was used to reduce common signal components and emphasize locally varying electrophysiological activity.

For each recording, power spectral density was estimated using a multitaper approach. The main spectral range considered in the analysis was **1–140 Hz**.

## Frequency Bands

The following frequency bands were defined in the feature-extraction pipeline:

| Band | Frequency range |
|---|---:|
| Delta | 1–4 Hz |
| Theta | 4–10 Hz |
| Beta | 15–30 Hz |
| Low gamma | 30–60 Hz |
| High gamma | 60–100 Hz |
| Fast gamma | 100–140 Hz |

The lower boundary of each frequency band is inclusive, whereas the upper boundary is exclusive.

The interval between **10 and 15 Hz** was intentionally not assigned to a separate frequency band. However, this interval remained part of the total **1–140 Hz** spectral range used in the calculation of relative power.

## Extracted Features

The main electrophysiological features include:

- absolute frequency-band power;
- relative frequency-band power;
- pairwise band-power ratios;
- selected oscillatory-episode measures.

### Absolute Band Power

Absolute band power represents the mean spectral power within a defined frequency band.

### Relative Band Power

Relative band power represents the power of a particular frequency band relative to the total spectral power between 1 and 140 Hz.

### Band-Power Ratios

Pairwise ratios between frequency bands were calculated to assess the balance between different oscillatory components.

Examples include ratios such as:

- high gamma / theta;
- low gamma / delta;
- fast gamma / beta.

## Statistical Analysis

Mixed-effects models were used to test diet-related differences in electrophysiological features while accounting for repeated recordings from the same mouse and relevant experimental covariates.

The general model structure was:

```text
band power ~ diet group × days on diet
             + body weight
             + estrous phase
             + random intercept for mouse
```

The statistical workflow included:

- preliminary data checks;
- multicollinearity assessment;
- mixed-effects model estimation;
- model diagnostics;
- marginal-effect estimation;
- effect-size calculation;
- false-discovery-rate correction;
- Mann–Whitney U sensitivity analysis;
- cable-specific and pooled model variants.

The mixed-effects framework was used because individual mice contributed repeated recordings and because body weight, days on diet, and estrous phase could influence the observed spectral values.

## Machine-Learning Analysis

Machine-learning analyses were performed to determine whether electrophysiological features could predict dietary group or body weight from individual recordings.

The analysis included:

- Random Forest classification and regression;
- support-vector classification and regression (SVM-RBF, primary model);
- Ridge regression as a linear sensitivity analysis;
- a mouse-cluster bootstrap with the mouse as the unit of resampling;
- feature-specific models;
- estrous-phase-specific models;
- permutation-based significance testing;
- bootstrap-based stability analysis.

The **SVM-RBF** classifier and regressor were treated as the primary learners, with the **Random Forest** used as a robustness check; conclusions were stated only where both learners agreed in direction.

### Mouse-Cluster Bootstrap

Because each mouse contributes several recordings that share its genetic background, surgical history, and baseline neurophysiology, a cluster-level bootstrap was used in which the **mouse — not the individual recording — is the resampled unit**.

In each of 1000 iterations, mice were drawn with replacement; all recordings from the drawn mice formed the training set, and the recordings of the mice not drawn formed the out-of-bag (OOB) test set. This prevents recordings from the same animal from appearing in both partitions and avoids inflating apparent generalisation performance.

The point estimate reported for every phase × feature-cell combination is the bootstrap mean of the OOB metric (balanced accuracy for RQ1, R² for RQ2), and its uncertainty is the 95% percentile confidence interval across iterations.

### Diet Classification

Diet classification performance was primarily evaluated using balanced accuracy because the number of observations could differ between diet groups and estrous phases.

### Body-Weight Prediction

Body-weight prediction was evaluated using regression-performance metrics. Body weight was used only as the target variable and was not included among the predictor features.

In addition to the cross-mouse bootstrap regression, a within-mouse analysis computed the Spearman correlation between each feature and each animal's own weight change, with a sign test applied across mice to identify features that track weight change consistently at the individual-animal level.

### Permutation Testing

Permutation testing was used to determine whether model performance was higher than expected by chance.

The target labels or values were repeatedly permuted at the level of the mouse, and the observed model performance was compared with the resulting null distribution.

### Bootstrap Analysis

Bootstrap procedures were used to evaluate the stability of model performance, significant feature cells, and cross-task comparisons across repeated resamples of the dataset.

## Estrous-Phase Analysis

Estrous phase was treated as an experimentally relevant variable rather than only as a nuisance covariate.

The machine-learning analyses were therefore performed separately across estrous phases to evaluate whether the relationship between electrophysiological activity, diet, and body weight depended on reproductive-cycle state.

This phase-specific approach allowed the analysis to identify electrophysiological patterns that might be present during one phase but absent during another.

## Software Requirements

The analyses were developed in Python.

The principal packages used across the repository include:

```text
numpy
pandas
scipy
matplotlib
statsmodels
scikit-learn
```

Additional packages may be required by individual scripts.

## Installation

Clone the repository:

```bash
git clone https://github.com/ZahraAmanli02/thesis_lfp_analysis_.git
cd thesis_lfp_analysis_
```

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate the environment on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Install the main dependencies:

```bash
pip install numpy pandas scipy matplotlib statsmodels scikit-learn
```

## Running the Analysis

The analysis scripts are located in the `scripts/` directory.

They should generally be executed according to their numerical order, beginning with data preparation and continuing through preprocessing, spectral feature extraction, statistical modelling, machine learning, bootstrap analysis, and figure generation.

Subsequent scripts depend on intermediate files generated during earlier stages of the workflow.

Not every script must be executed for every analysis. Some files are used specifically for:

- quality-control inspection;
- diagnostic visualization;
- sensitivity analysis;
- alternative model specifications;
- bootstrap analysis;
- defense or thesis figure generation;
- source-data export.

## Path Configuration

Some scripts may contain local path variables that refer to the computer on which the original analysis was performed.

Before running the analysis on another computer, these paths must be replaced with the local path to the cloned repository.

A portable relative-path configuration can be defined as follows:

```python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
```

Using project-relative paths is recommended to improve reproducibility across operating systems and computers.

## Reproducibility Notes

- Cable 1 and Cable 3 are generally analysed independently.
- Recordings from the same mouse are treated as repeated observations.
- Mouse-level resampling (mouse-cluster bootstrap) is used in predictive analyses.
- Random seeds should be retained where they are specified.
- Model significance is evaluated using permutation or bootstrap procedures where applicable.
- Some scripts require outputs generated by earlier pipeline stages.
- Local file paths may need to be configured before execution.
- Raw electrophysiological recordings may not be included because of their size or applicable data-sharing restrictions.

## Outputs

Generated results are stored in the `outputs/` directory.

These outputs include:

- processed metadata tables;
- quality-control summaries;
- power spectral density results;
- extracted frequency-band features;
- band-power ratio tables;
- mixed-effects model results;
- model diagnostics;
- marginal-effect estimates;
- effect-size tables;
- classification results;
- regression results;
- permutation-test results;
- bootstrap distributions;
- heatmaps and summary figures;
- numerical source data underlying thesis figures.

## Source Data

Numerical source data were exported to provide the individual values underlying the averages and summary statistics shown in the thesis figures.

The source-data files are intended to support transparency and allow the reported figure values to be independently checked.

Depending on the figure, these files may contain:

- individual recording-level values;
- mouse-level information;
- group and phase labels;
- frequency-band features;
- model-performance values;
- bootstrap iterations;
- summary means used in the figures.

## Data Availability

This repository contains processed data and numerical source data used to document the reported analyses.

Large raw electrophysiological recording files may not be included in the public repository because of file-size limitations or institutional data-sharing requirements.

Additional data may be available from the author or the supervising research group upon reasonable request and subject to applicable institutional conditions.

## Academic Context

This repository was created as part of a thesis project at:

**Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)**

The project investigates diet-related and body-weight-related electrophysiological signatures in the lateral hypothalamus, with particular attention to frequency-specific activity, estrous-phase dependence, and reproducibility across independent recording cables.

## Author and Contact

**Zahra Amanli**

Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)

- University email: [zahra.amanli@fau.de](mailto:zahra.amanli@fau.de)
- Personal email: [zahra.amanli.za@gmail.com](mailto:zahra.amanli.za@gmail.com)
- GitHub: [ZahraAmanli02](https://github.com/ZahraAmanli02)

## Citation and Academic Use

This repository was developed for an academic thesis project.

When reusing or referring to the code, processed data, figures, or analysis outputs, please acknowledge the repository author and cite the corresponding thesis where appropriate.
