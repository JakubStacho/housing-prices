# 🏠 Kaggle Housing Price Prediction

My first Kaggle competition project! The goal of this competition is to predict residential home sale prices using the [Ames Housing dataset](https://www.kaggle.com/competitions/home-data-for-ml-course). My goal with this project is to build and compare multiple modelling approaches including boosted decision trees, linear regression, and a final multi-model ensemble to understand the different tradeoffs between model families.

**Competition:** [Housing Prices Competition for Kaggle Learn Users](https://www.kaggle.com/competitions/home-data-for-ml-course)  
**Metric:** Mean Absolute Error (MAE) on predicted sale price

---

## Notebooks

| Notebook | Description |
|---|---|
| [`data-exploration.ipynb`](data-exploration.ipynb) | Missing data analysis, feature distributions, log-transform investigation, and residual-based outlier identification |
| [`bdt-model.ipynb`](bdt-model.ipynb) | XGBoost and LightGBM pipelines with feature engineering, Optuna hyperparameter tuning, and weighted BDT ensemble |
| [`linear-model.ipynb`](linear-model.ipynb) | Huber regression pipeline with log-normalisation, mutual information feature selection, and Optuna tuning |
| [`ensemble-model.ipynb`](ensemble-model.ipynb) | Final weighted ensemble combining BDT and linear model predictions |

---

## Approach

### Data Preprocessing
- Missing values are either filled with a meaningful sentinel (`'NA'` for absent features like no fireplace) or imputed from related columns
- Two anomalous training rows that were identified in [`data-exploration.ipynb`](data-exploration.ipynb) are removed. These represent large Edwards neighbourhood homes sold as partial transactions at prices far below market rate
- Categorical columns are encoded as ordered or unordered pandas `CategoricalDtype`. XGBoost uses these natively via `enable_categorical=True`. Ordinal columns are converted to integer codes for LightGBM and the linear model

### Feature Engineering
New features are constructed on top of the base input columns including ratios, products, totals, ages, presence flags, and neighbourhood statistics.

The linear model applies additional preprocessing:
- **Log normalisation** of right-skewed continuous features, guided by QQ correlation analysis to reduce heteroskedasticity. Ratios and products are excluded to avoid creating near-linear dependencies with their log-transformed components
- **MI-based feature selection** to drop uninformative columns
- **Dropping linear combinations**: individual bathroom and floor columns are removed after creating aggregate features (`TotBath`, `TotalSF`, `TotalPorchSF`)

### Modelling
- Target is `log(SalePrice)`. This reduces the influence of high-value outliers and linearises the price distribution
- All models are scored with 5-fold cross-validated RMSE on `log(SalePrice)`, consistent with the competition's RMSLE metric
- **BDT models**: XGBoost and LightGBM are each tuned independently with [Optuna](https://optuna.org/) (Bayesian TPE optimisation, 50 trials each). The two are combined as a weighted ensemble on out-of-fold predictions
- **Linear model**: Huber regression is tuned with Optuna over epsilon and alpha. Huber was chosen over Ridge after comparing both models. Huber has a robust loss function that handles residual outliers better even after log-transforming the target
- **Final ensemble**: all three models (XGBoost, LightGBM, Huber) are combined with weights optimised on out-of-fold predictions

---

## Results

| Model | CV log-RMSE | Kaggle MAE |
|---|---|---|
| XGBoost baseline (default params) | 0.133 | - |
| XGBoost + feature engineering | 0.130 | - |
| XGBoost + feature engineering + Optuna | 0.112 | 13148.58 |
| LightGBM + feature engineering + Optuna | 0.116 | - |
| XGBoost + LightGBM ensemble | 0.111 | 13018.66 |
| Huber regression + Optuna | 0.114 | 13977.83 |
| XGBoost + LightGBM + Huber ensemble | 0.108 | **12830.34** |

---

## Lessons Learned

1. **Similar models don't ensemble well.** Combining XGBoost with LightGBM gave only a marginal improvement in CV RMSE (0.112 → 0.111) because both are gradient-boosted decision trees that split on similar features and make similar errors. The real ensemble gain came from adding a fundamentally different model family (linear regression), which dropped CV RMSE from 0.111 to 0.108.

2. **Huber over Ridge for robust linear regression.** I initially built the linear model with Ridge, then swapped to Huber regression which uses L1 loss on large residuals. Even though the target is already log-transformed (which compresses outliers), Huber still outperformed Ridge (0.114 vs 0.116 CV RMSE). The optimal epsilon ≈ 1.95 suggests only a few residuals are treated as outliers, but that small correction seems to matter.

3. **Log-transforming engineered features can backfire.** Applying log normalisation to ratio and product features (like `QualArea = OverallQual × GrLivArea`) when their source columns are also log-transformed creates near-linear dependencies: `log(QualArea) ≈ log(OverallQual) + log(GrLivArea)`. This collapses the interaction information that the feature was designed to capture. The fix was to only log-transform the raw source columns.

4. **Low marginal importance ≠ low conditional contribution.** Features like `LivLotRatio` had low mutual information and low Pearson correlation with `SalePrice`, but removing them made the Ridge/Huber model worse. They act as variance suppressors that absorb the covariance between their component features (GrLivArea, LotArea) and let Ridge/Huber fit the remaining signal more cleanly.

5. **LGBM gets zero weight in the final ensemble.** With XGBoost and Huber already in the mix, LightGBM's optimal weight dropped to 0. Its predictions are too correlated with XGBoost to add value once a genuinely different model (Huber) is present. This reinforces the lesson that ensemble diversity matters more than individual model strength.

---

## Project Structure

```
housing-prices/
├── input/
│   ├── train.csv.gz          # 1,460 training examples
│   └── test.csv.gz           # 1,459 test examples
├── output/
│   ├── ensemble-input-predictions/  # Saved OOF and test predictions per model
│   ├── bdt-submission.csv           # BDT-only Kaggle submission
│   ├── huber-submission.csv         # Huber-only Kaggle submission
│   └── ensemble-submission.csv      # Final ensemble Kaggle submission
├── optuna-cache/             # Cached Optuna studies (avoids re-running tuning)
├── data-exploration.ipynb
├── bdt-model.ipynb
├── linear-model.ipynb
├── ensemble-model.ipynb
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
