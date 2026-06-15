# 🏠 Kaggle Housing Price Prediction

My first Kaggle competition project! The goal of this competition is to predict residential home sale prices using the [Ames Housing dataset](https://www.kaggle.com/competitions/home-data-for-ml-course). My goal with this project is to build and compare multiple modelling approaches, starting with a boosted tree model and then revisiting the problem with a linear model to understand the different tradeoffs.

**Competition:** [Housing Prices Competition for Kaggle Learn Users](https://www.kaggle.com/competitions/home-data-for-ml-course)  
**Metric:** Mean Absolute Error (MAE) on predicted sale price

---

## Notebooks

| Notebook | Description |
|---|---|
| [`data-exploration.ipynb`](data-exploration.ipynb) | Missing data analysis, feature distributions, log-transform investigation, and residual-based outlier identification |
| [`bdt-model.ipynb`](bdt-model.ipynb) | XGBoost and LightGBM pipelines with feature engineering, Optuna hyperparameter tuning, and weighted ensemble |
| [`linear-model.ipynb`](linear-model.ipynb) | *(In progress)* Apply Ridge/Lasso regression to the same problem |

---

## Approach

### Data Preprocessing
- Missing values are either filled with a meaningful sentinel (`'NA'` for absent features like no fireplace) or imputed from related columns
- Two anomalous training rows that were identified in [`data-exploration.ipynb`](data-exploration.ipynb) are removed. These represent large Edwards neighbourhood homes sold as partial transactions at prices far below market rate
- Categorical columns are encoded as ordered or unordered pandas `CategoricalDtype`. XGBoost uses these natively via `enable_categorical=True`; ordinal columns are converted to integer codes for LightGBM, which does not respect categorical ordering

### Feature Engineering
The following features are constructed on top of the raw columns:

- **Ratios**: living area to lot size, porch coverage, room spaciousness
- **Products**: overall quality × living area
- **Totals**: combined interior square footage, total bathroom count
- **Ages**: house age and years since last renovation at time of sale
- **Presence flags**: binary indicators for garage and basement
- **Neighbourhood statistics**: median and standard deviation of living area per neighbourhood, fit on training data only to prevent leakage

### Modelling
- Target is `log(SalePrice)`. This reduces the influence of high-value outliers and linearises the price distribution
- Scored with 5-fold cross-validated RMSE on `log(SalePrice)`, consistent with the competition's RMSLE metric
- XGBoost and LightGBM are each tuned independently with [Optuna](https://optuna.org/) (Bayesian TPE optimisation, 50+ trials each)
- The two models are combined as a weighted ensemble: $$\alpha\times\mathrm{XGB} + (1−\alpha)\times\mathrm{LGBM}$$. The optimal $\alpha$ is found by minimising ensemble RMSE on out-of-fold predictions using `scipy.optimize.minimize_scalar`

---

## Results

| Model | CV log-RMSE | Kaggle Score |
|---|---|---|
| XGBoost baseline (default params) | 0.133 | |
| XGBoost + feature engineering | 0.130 | |
| XGBoost + feature engineering + Optuna tuning | 0.112 | 13148.57641 |
| LightGBM + feature engineering + Optuna tuning | 0.116 | |
| XGBoost + LightGBM ensemble (α = 0.857) | 0.111 | 13018.65943|

### Observations
1) Combining XGBoost with LightGBM results in a very minor improvement. This makes sense because both XGB and LGBM are both gradient boosted decision trees trained on the same features. Their predictions are probably highly correlated because they split on similar features and end up making similar errors. To get the best gain from an ensemble model we should consider using models that are fundamentally different like adding a linear model to the final ensemble.

---

## Project Structure

```
housing-prices/
├── input/
│   ├── train.csv.gz          # 1,460 training examples
│   └── test.csv.gz           # 1,459 test examples
├── output/
│   └── xgb-submission.csv    # Kaggle submission file
├── optuna-cache/             # Cached Optuna study (avoids re-running tuning)
├── data-exploration.ipynb
├── bdt-model.ipynb
├── linear-model.ipynb
└── data_description.txt      # Full feature descriptions from Kaggle
```

## Setup

```bash
# create and activate the virtual environment
python -m venv venv-housing
source venv-housing/bin/activate   # or activate-housing-venv

# install dependencies
pip install -r packages.txt
```
