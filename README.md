# EnergyWise: Electricity Consumption Prediction and Cost Scenarios

A beginner-friendly AI/ML and calculus project that predicts appliance energy consumption for the next 10-minute interval using multivariable linear regression trained with manually implemented NumPy batch gradient descent. Includes exploratory data analysis, evaluation against a mean baseline and scikit-learn, and a Streamlit dashboard.

## Measured results

| Model | MAE (Wh) | RMSE (Wh) | R² |
|---|---:|---:|---:|
| Training-mean baseline | 52.686 | 91.053 | -0.000328 |
| Manual NumPy gradient descent | 26.495 | 61.327 | 0.546216 |
| scikit-learn LinearRegression | 26.495 | 61.327 | 0.546213 |

Results use a chronological 80/20 split of 19,734 valid next-interval examples (15,787 train; 3,947 test). These are dataset-specific results, not guarantees for other buildings.

## Quick start

Requires Python 3.10+. Install dependencies and download the UCI dataset:

    python -m venv .venv
    # Linux/macOS: source .venv/bin/activate
    # Windows PowerShell: .venv\\Scripts\\Activate.ps1
    python -m pip install -r requirements.txt
    python src/download_data.py
    python src/inspect_dataset.py
    python src/eda.py
    python src/model.py
    streamlit run app.py

Run tests with:

    python -m unittest discover -s tests -v

## Project contents

- app.py — interactive Streamlit predictor and illustrative cost scenarios.
- src/model.py — time alignment, manual gradient descent, baselines, metrics and plots.
- src/eda.py — descriptive analysis and visualisations.
- src/inspect_dataset.py — data quality checks.
- src/download_data.py — dataset downloader.
- tests/test_model.py — numerical-gradient, convergence and timestamp-alignment tests.
- outputs/metrics.json and outputs/RESULTS_SUMMARY.md — measured evaluation results.
- reports/Viva_Questions_and_Answers.md — viva preparation.
- reports/Project_Report_Final_With_Results.pdf and .docx — final report files.
- reports/Project_Presentation_Final.pptx — presentation.

Binary report, presentation and chart artifacts are provided separately with the project package where not present in this GitHub source upload. Large dataset files are intentionally not committed; download them with the script.

## Method

Input features at time t predict Appliances at t + 10 minutes. Consecutive timestamps not exactly 10 minutes apart are excluded. The split is chronological and scaling is fitted on training data only. For standardized inputs Z and target y, prediction is y_hat = Z @ w + b and mean-squared-error gradients are dw = (2/n) Z.T @ (y_hat - y), db = 2 mean(y_hat - y). Parameters are updated in the negative-gradient direction.

## Cost estimate and limitations

Predicted Wh are divided by 1,000, then multiplied by a user-entered illustrative tariff in ₹/kWh. This is a simplified interval estimate, not an official electricity bill or guaranteed saving. The source dataset describes one building in Belgium for a limited period; performance is not validated for Indian households.

## Dataset attribution

Candanedo, “Appliances Energy Prediction,” UCI Machine Learning Repository, DOI: 10.24432/C5VC8G. Dataset page: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction. Check the official repository for current metadata and licence conditions (CC BY 4.0 listed there).

## Before submission

Fill in student names, USNs, faculty guide and academic year; run the tests and dashboard locally; verify citations; and update AI_TOOL_USAGE_REPORT_TEMPLATE.md to reflect your team's actual use and modifications.