# Measured results summary

Dataset: UCI Appliances Energy Prediction.

- Raw dimensions: 19,735 rows × 29 columns
- Missing values: 0
- Duplicate rows: 0
- Duplicate timestamps: 0
- Valid next-interval examples: 19,734
- Training samples: 15,787
- Test samples: 3,947
- Input features: 13
- Learning rate: 0.03
- Iterations: 2,500
- Standardized training MSE: 1.000000 initially → 0.417419 after the final iteration
- Loss reduction: 58.26%

| Model | MAE (Wh) | RMSE (Wh) | R² |
|---|---:|---:|---:|
| Training-mean baseline | 52.6861 | 91.0535 | -0.000328 |
| Manual NumPy gradient descent | 26.4950 | 61.3267 | 0.546216 |
| scikit-learn LinearRegression | 26.4955 | 61.3270 | 0.546213 |

Results apply only to this chronological test split and do not establish performance for other buildings or households.
