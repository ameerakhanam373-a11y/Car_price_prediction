# Car Price Estimator 

This project follows the supplied brief: clean the car data, perform EDA, fit a multiple linear regression model, report holdout metrics, and offer predictions through Gradio.

## What I built

Created a prediction app with Gradio so users can enter data and receive a model prediction through a simple interface.

## Files

- `Car Prices Data.csv` — supplied source data
- `car_price_estimator.ipynb` — Google Colab/Jupyter walkthrough with cleaning, EDA, modeling, and app launch
- `app.py` — standalone Gradio estimator
- `requirements.txt` — Python dependencies

## Run in Google Colab

1. Upload `car_price_estimator.ipynb` and `Car Prices Data.csv` to Colab.
2. Run the notebook top to bottom. It installs the dependencies, cleans and explores the data, evaluates the regression model, and launches the Gradio interface.

## Run locally

```bash
python -m pip install -r requirements.txt
python app.py
```

Open the local Gradio URL printed in the terminal. The model imputes missing predictors within a scikit-learn pipeline, one-hot encodes brand and fuel type, and trains only on rows with valid positive prices. A Tukey IQR fence excludes extreme target outliers. The holdout metrics are reported in the app. The dataset contains only about 60 rows, so predictions are a learning demonstration rather than reliable market valuations.

The Gradio interface has a teal and violet palette, a dedicated estimate card, and compact holdout metric cards.

## Project summary

This project demonstrates how to connect a machine learning model to an interactive web interface. Add the model’s purpose, input features, and evaluation result here so readers know what it predicts and how well it performed.

## Cleaning choices

- Trim and standardize brand/fuel capitalization; map the obvious `Hyundia` typo to `Hyundai`.
- Parse mileage strings into numeric kilometres, including malformed two-digit thousands groups such as `21,14 km` as 21,140 km.
- Convert unparseable prices (including `ten lakh`) to missing. Zero prices are treated as invalid.
- Keep missing feature values for median/mode imputation inside the modeling pipeline, avoiding leakage from test data.
- Exclude rows with missing target from training and remove target outliers using the 1.5×IQR rule.
