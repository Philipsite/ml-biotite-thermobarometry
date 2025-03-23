# ml-biotite-thermobarometry
This repository contains the supplementary code to "Calibration, validation, and evaluation of machine learning thermobarometers in metamorphic petrology: an application to biotite and outlook for future strategy" by Hartmeier et al. (2025).

## Contents
The data and figures in the manuscript can be reproduced using the jupyter notebooks in th is repository. The most important results and their corresponding notebook are listed here.

### 00: Data analysis
- [Compositional variation of natural biotite](00_data_analysis/00b_compositional_variation_of_biotites.ipynb) (Figure 1)
- [Compositional variation of natural biotite in P-T space](00_data_analysis/00c_compositional_variation_biotites_PTspace.ipynb) (Figure 3)
- [Compositional variation of biotite generated using PEM in P-T space](00_data_analysis/00c_compositional_variation_biotites_PTspace.ipynb) (Figure 4)

### 01: Fit natural biotite
- Model trained on natural data (used in Tables 2 and 3, and Figures 5, 6, 7, 8, 9 and 10):
    - [k-fold training TiXMg model](01_fit_natural_biotite/AA_k_training_TiXMg.ipynb)
    - [k-fold training MnFMAST model](01_fit_natural_biotite/AA_k_training_MnFMAST.ipynb)
    - [k-fold training MnFMAST + idx model](01_fit_natural_biotite/AA_k_training_MnFMAST_index_minerals.ipynb)
    - [k-fold training MnFMAST v2025Feb model](01_fit_natural_biotite/BB_k_training_MnFMAST_updatedDBFeb2025.ipynb)

- S1 Feature Selection (Figures S1.1-1.2):
    - [Feature engineering](01_fit_natural_biotite/feature_eng_testing.ipynb)
    - [Feature engineering plots](01_fit_natural_biotite/plot_feature_eng_testing.ipynb)
- S2 Architecture and Hyperparameter Tuning (Figures S2.1-2.3):
    - [Hyperparameter tuning](01_fit_natural_biotite/hyperparam_testing.ipynb)
    - [Hyperparameter tuning plots](01_fit_natural_biotite/plot_hyperparam_testing.ipynb)


**Data**:
- Biotite data from mineral assemblage sequences, based on the database of Pattison and Forshaw (2025, in review).
    - [Biotite data set](01_fit_natural_biotite/Metapelite-Database_Bt_CLEAN_2024-02-03.xlsx): Used to calibrate the thermobarometer.
    - [Biotite data set](01_fit_natural_biotite/Metapelite-Database_Bt_CLEAN_InclBt-Na2024-02-16.xlsx), reduced to analyses with measured Na: Used in feature engineering.
    - [Biotite data set](01_fit_natural_biotite/Metapelite-Database_Bt_CLEAN_2025-02-25.xlsx), latest version of Pattison and Forshaw (v2025-February). Used to test whether recent updates to the database have resulted in a significant change in model performance.

- K-fold data. CSVs of 5-fold training and validation splits used during cross-validation.
    - [K-fold biotite data](01_fit_natural_biotite/kfold_datasets): Used to calibrate the thermobarometer.
    - [K-fold biotite data](01_fit_natural_biotite/kfold_datasets_updatedDBFeb2025), latest version of Pattison and Forshaw (v2025-February). Used to test whether recent updates to the database have resulted in a significant change in model performance.

- [Training logs](01_fit_natural_biotite/logs), log files of training and validation performance for all models calibrated. 

- [Saved models](01_fit_natural_biotite/saved_models), trained models saved in tensorflow's SavedModel format.

### 02: Pre-training on biotite generated using phase equilibrium modelling
- Model trained on PEM data (used for the transfer learning):
    - [Pre-training model using ds55 and the solution models of White et al. (2007)](02_pretraining/priormodel_ds55Bt07.ipynb)
    - [Pre-training model using ds55 and the solution models of Tajcmanova et al. (2009)](02_pretraining/priormodel_ds55BtT.ipynb)
    - [Pre-training model using ds62 and the solution models of White et al. (2014)](02_pretraining/priormodel_ds62.ipynb)

- S2 Architecture and Hyperparameter Tuning (Figures S2.4):
    - [Hyperparameter tuning](02_pretraining/hyperparam_testing.ipynb)
    - [Hyperparameter tuning plots](02_pretraining/plot_hyperparam_testing.ipynb)

**Data**:
- [Training logs](02_pretraining/logs), log files of training and validation performance for all models calibrated. 

- [Saved models](02_pretraining/saved_models), trained models saved in tensorflow's SavedModel format.

- Data sets generated using phase equilibrium modelling are available upon request.

### 03: Transfer learning
- Training of the single crystal biotite thermobarometer ("final" model evaluated and applied in the paper):
    - [Final calibration using transfer learning](03_transfer_learning/03a_BtThermobarometer_finetuning.ipynb)
- Model trained using transfer learning (k models for cross-validation):
    - [Transfer learning model using the prior model trained with ds55 and the solution models of White et al. (2007)](03_transfer_learning/k_transfer_ds55Bt07.ipynb)
    - [Transfer learning model using the prior model trained with ds55 and the solution models of Tajcmanova et al. (2009)](03_transfer_learning/k_transfer_ds55BtT.ipynb)
    - [Transfer learning model using the prior model trained with ds62 and the solution models of White et al. (2014)](03_transfer_learning/k_transfer_ds62.ipynb)

- S2 Architecture and Hyperparameter Tuning (Figures S2.5):
    - [Transfer method](03_transfer_learning/transfer_technique.ipynb)
    - [Transfer method plots](03_transfer_learning/plot_transfer_technique.ipynb)

**Data**:
- [Training logs](03_transfer_learning/logs), log files of training and validation performance for all models calibrated. 

- [Saved models](03_transfer_learning/saved_models), trained models saved in tensorflow's SavedModel format.

### 04: Validation & model selection
- K-fold cross-validation using model M1, M2a/b, M3a-c (Table 1):
    - [Model validation](04_model_selection_validation/RMSE_kfold_crossvalidation.ipynb) RMSE (Figure 5) and RMSE for different *P*-/*T*-bins (Figure 6)
    - Additional [model validation](04_model_selection_validation/RMSE_kfold_crossvalidatio_compareDB_2024_2025.ipynb), compare if there is a significant effect of training a model equivalent to M2a and M3c on data from an updated version of the Pattison and Forshaw database (v2025Feb) compared to (v2024Feb).

- Validation using metapelitic sequences:
    - [Validation using sequences](04_model_selection_validation/sequence_validation.ipynb) (Figure 7 and 8)
    - **Data**: [Validation dataset of metapelitic sequences](04_model_selection_validation/validation_data_sequences.xlsx)

- Validation using Monte-Carlo error propagation:
    - [Validation using MC error propagation](04_model_selection_validation/error_propagation_validation.ipynb) (Figure 9 and 10)

### 05: Systematic performance analysis (testing)

- [Performance evaluation](05_systematic_performance_analysis/performance_test_data.ipynb) on the test dataset (Figure 12)
- [Comparison with Ti-in-biotite thermometry](05_systematic_performance_analysis/performance_comparison_to_TiBt.ipynb) (Figure 13)

**Data**:
- [Test dataset](05_systematic_performance_analysis/test_data_metapelitic_biotite.xlsx)
- Compositional maps (hdf5 format) are available upon request.

### 06: Explainability
- [Partial derivatives of thermobarometer](06_explainability/pdv_thermobarometer.ipynb)


## A note on reproducibility
As the initial parameterisation and gradient descent optimisation are stochastic processes, the training of a neural network is not fully reproducible.

Therefore, it is not recommended to re-run the scripts used to train the models, as this will overwrite the original calibration of the neural network used in the work presented here.
The purpose of these scripts is solely to document the training procedure and can be copied as a template to fit other new neural networks.

To experiment with the models calibrated here, they can be loaded from the `saved_models` directories provided.


## Installation / Dependencies

The code in this repo is depending on Keras2 and will no longer work with `keras >3.x`. Therefore `tensorflow` is limited to `v2.15.0` as this was the the [final release before the launch of Keras 3.0](https://github.com/tensorflow/tensorflow/releases/tag/v2.15.0).

This introduces some important hard dependencies itself:
- NumPy is limited to `<2.0`, which then imposes
- Python `<3.12`, tested with `3.11` (recommended).

The full dependencies are specified in the `pyproject.toml` and `poetry.lock` file. Checkout [poetry](https://python-poetry.org/) to make use of the lock file to reproduce the virtual env used to generate all results presented here exactly.

**CAUTION:** The poetry installation of tensorflow, seems to fail silently. The installation using `poetry install` runs without error, but tensorflow will not be installed and cannot be called. As a work around tensorflow was installed afterwards using `pip`.
```zsh
source .venv/bin/activate
pip install tensorflow==2.15
```
followed by (to install the ml_tb project locally)
```zsh
poetry install
```