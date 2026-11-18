"""Car price estimator: data cleaning, regression pipeline, and Gradio UI."""

from pathlib import Path
import re

import gradio as gr
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path(__file__).with_name("Car Prices Data.csv")
FEATURES = ["Car_Age", "Mileage", "Brand", "Fuel_Type"]
TARGET = "Selling_Price"


def number(value):
    """Convert numbers containing separators/units to float; invalid values -> NaN."""
    if pd.isna(value):
        return float("nan")
    raw = str(value).strip()
    # Repair common two-digit thousands groups such as "21,14 km" -> 21,140.
    grouped = re.fullmatch(r"\s*(\d{1,3}),(\d{2})\s*(?:km)?\s*", raw, re.I)
    if grouped:
        return float(grouped.group(1) + grouped.group(2) + "0")
    found = re.sub(r"[^0-9.\-]", "", raw.replace(",", ""))
    try:
        return float(found) if found else float("nan")
    except ValueError:
        return float("nan")


def load_and_clean(path=DATA_PATH):
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip()
    data["Car_Age"] = pd.to_numeric(data["Car_Age"], errors="coerce")
    data["Mileage"] = data["Mileage"].map(number)
    data[TARGET] = data[TARGET].map(number)
    # Keep categorical columns as ordinary object dtype: sklearn imputers handle
    # np.nan reliably, while pandas' nullable pd.NA can cause ambiguous booleans.
    data["Brand"] = data["Brand"].astype(object).str.strip().str.title()
    data["Fuel_Type"] = data["Fuel_Type"].astype(object).str.strip().str.title()
    # Correct the single obvious typo after case/space normalization.
    data["Brand"] = data["Brand"].replace({"Hyundia": "Hyundai"})
    data.loc[data[TARGET] <= 0, TARGET] = float("nan")
    data.loc[data["Mileage"] < 0, "Mileage"] = float("nan")
    return data


def build_model(data):
    # A missing target cannot be learned. An IQR fence removes the single extreme
    # target outlier while keeping all other valid observations.
    train = data.dropna(subset=[TARGET]).copy()
    q1, q3 = train[TARGET].quantile([0.25, 0.75])
    iqr = q3 - q1
    train = train[train[TARGET].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)]
    numeric = ["Car_Age", "Mileage"]
    categorical = ["Brand", "Fuel_Type"]
    prep = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median"))]), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    model = Pipeline([("preprocess", prep), ("regression", LinearRegression())])
    X_train, X_test, y_train, y_test = train_test_split(
        train[FEATURES], train[TARGET], test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, predicted),
        "RMSE": mean_squared_error(y_test, predicted) ** 0.5,
        "R²": r2_score(y_test, predicted),
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "excluded_target_outliers": len(data.dropna(subset=[TARGET])) - len(train),
    }
    return model, train, metrics


cleaned = load_and_clean()
model, model_data, metrics = build_model(cleaned)
brands = sorted(cleaned["Brand"].dropna().unique().tolist())
fuels = sorted(cleaned["Fuel_Type"].dropna().unique().tolist())


def predict(age, mileage, brand, fuel):
    row = pd.DataFrame([{
        "Car_Age": age, "Mileage": mileage, "Brand": brand, "Fuel_Type": fuel,
    }], columns=FEATURES)
    value = model.predict(row)[0]
    return f"Estimated selling price: {value:,.0f}"


with gr.Blocks(
    title="Drivewise | Car Price Estimator",
    theme=gr.themes.Soft(primary_hue="teal", secondary_hue="violet", neutral_hue="slate"),
    css="""
    .gradio-container {background: linear-gradient(145deg, #f4fbfa 0%, #f7f5ff 52%, #fffaf3 100%);}
    .hero {padding: 1.5rem 1.7rem; border-radius: 22px; color: white;
      background: linear-gradient(110deg, #087f8c, #5865c8 72%, #7a55ad); margin-bottom: 1rem;}
    .hero h1 {color: white !important; margin-bottom: .35rem;}
    .hero p {color: #eafcff; font-size: 1.05rem; margin: 0;}
    .panel {border: 1px solid #dce8eb; border-radius: 18px; background: rgba(255,255,255,.88);}
    .metric {text-align:center; padding:.8rem; border-radius:14px; background:#f0f8f8;}
    .metric h3 {margin:.15rem 0; color:#087f8c;}
    .metric p {margin:0; color:#52616b; font-size:.85rem;}
    .footnote {color:#63717a; font-size:.9rem;}
    """,
) as demo:
    gr.Markdown(
        "# 🚗 Drivewise\nA thoughtful first estimate, based on a car’s age, mileage, brand, and fuel type.",
        elem_classes=["hero"],
    )
    with gr.Row():
        for label, value in [("TRAINING CARS", f"{len(model_data)}"),
                             ("TEST MAE", f"{metrics['MAE']:,.0f}"),
                             ("TEST R²", f"{metrics['R²']:.2f}")]:
            gr.Markdown(f"<div class='metric'><p>{label}</p><h3>{value}</h3></div>")
    with gr.Row():
        with gr.Column(scale=3):
            with gr.Group(elem_classes=["panel"]):
                gr.Markdown("### Tell us about the car\nChoose the closest matching details.")
                age = gr.Slider(0, 20, value=5, step=1, label="Age", info="Years")
                mileage = gr.Number(value=15000, label="Mileage", info="Kilometres")
                with gr.Row():
                    brand = gr.Dropdown(brands, value=brands[0] if brands else None, label="Brand")
                    fuel = gr.Dropdown(fuels, value=fuels[0] if fuels else None, label="Fuel type")
                button = gr.Button("✨  Estimate price", variant="primary")
        with gr.Column(scale=2):
            with gr.Group(elem_classes=["panel"]):
                gr.Markdown("### Your estimate")
                result = gr.Textbox(label="Estimated selling price", interactive=False,
                                    placeholder="Your estimate will appear here")
                gr.Markdown(
                    f"Holdout RMSE: **{metrics['RMSE']:,.0f}**  \n"
                    "A small teaching dataset means this estimate is for learning, not a market valuation.",
                    elem_classes=["footnote"],
                )
    button.click(predict, inputs=[age, mileage, brand, fuel], outputs=result)

if __name__ == "__main__":
    demo.launch()
