# Retail Credit Risk Modeling Framework

This repository contains the complete analytical pipeline, technical documentation, and executive deliverables for the redevelopment of a retail credit risk scoring model. 

The project evaluates a baseline Generalized Linear Model (GLM) against a Generalized Linear Mixed Effects Model (GLMM) to explicitly quantify how macro-economic events (the Global Financial Crisis) and geographical variance influence portfolio risk independently of applicant quality.

## Project Structure

### Core Deliverables
* **`Credit_Risk.Rmd` / `Credit_Risk.pdf`**: The comprehensive technical statistical report (31 pages). This document contains the full data preprocessing pipeline, feature selection, comparative model fitting (GLM vs GLMM), diagnostics, and out-of-sample evaluations.
* **`SOAP.Rmd` / `SOAP.pdf`**: The "Summary on a Page" (SOAP). A strictly 2-page condensed executive memorandum outlining strategic business impacts, operational recommendations, and key visualizations.
* **`Presentation.qmd` / `Presentation.pptx`**: The stakeholder presentation slide deck formatted in Microsoft PowerPoint (`pptx`), designed to communicate modeling results, economic drift, and strategic recommendations to senior leadership.

### Data & Infrastructure
* **`data/`**: Benchmark partitions (`benchmark_training_loan_data.csv`, `benchmark_validation_loan_data.csv`, `benchmark_testing_loan_data.csv`), extended dataset (`extendend_version_loan_data.csv`), and the data dictionary (`loan_data_dictionary.xlsx`).
* **`assets/`**: Serialized pre-computed model archives (`glmm_models.rds`, `dharma_results.rds`, and `soap_data.rds`).
  * **`assets/quarto/`**: Centralized image artifacts and slide figures generated during Quarto presentation builds.
* **`scripts/`**: Automated reproduction pipeline (`fit_all_models.R`) for fitting all candidate models from raw data and generating diagnostic archives. Modularized statistical utilities (`utils.R`) providing data preprocessing, factor-level alignment, and discrimination metrics (AUC/Gini).
* **`reference/`**: Assessment rubrics, project requirements, and academic background materials.
* **`agent/`**: Development logs, architectural documentation, and consolidated lessons-learned study.

## Required R Packages & Tools

To successfully compile these documents, the following R packages and tools are required:
* **Data Manipulation**: `dplyr`, `tidyr`, `lubridate`
* **Modeling & Evaluation**: `caret`, `pROC`, `ROCR`, `lme4`, `arm`, `car`
* **Diagnostics & Marginal Effects**: `DHARMa`, `ggeffects`
* **Visualization & Tables**: `ggplot2`, `patchwork`, `knitr`, `kableExtra`, `scales`
* **Presentation Tool**: [Quarto CLI](https://quarto.org) (for rendering `Presentation.pptx`)

## Reproducibility & Execution Order

All predictive outcomes and data partitions are strictly reproducible (`set.seed(42)`).

Pre-computed models in `assets/` allow documents to render rapidly without lengthy GLMM convergence delays. If model files are missing, `Credit_Risk.Rmd` automatically calls `scripts/fit_all_models.R` to reconstruct them from raw data.

**Standard Compilation Workflow:**
1. **Render `Credit_Risk.Rmd`**: Compiles the 31-page technical report and updates `assets/soap_data.rds`.
   ```bash
   Rscript -e 'rmarkdown::render("Credit_Risk.Rmd")'
   ```
2. **Render `SOAP.Rmd`**: Compiles the strictly 2-page executive summary.
   ```bash
   Rscript -e 'rmarkdown::render("SOAP.Rmd")'
   ```
3. **Render `Presentation.qmd`**: Generates the PowerPoint slide deck (`Presentation.pptx`).
   ```bash
   quarto render Presentation.qmd
   ```

To refit all models from scratch without cached artifacts, run:
```bash
Rscript scripts/fit_all_models.R --refit
```
