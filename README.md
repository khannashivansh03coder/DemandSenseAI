# DemandSense — Hotel Demand Forecasting & OTA Profit Optimisation

## Overview
DemandSense is a hotel analytics prototype that forecasts hotel occupancy and compares estimated profitability across online travel agencies (OTAs) and direct bookings.

## Features
- Occupancy forecasting.
- OTA booking and cancellation prediction using synthetic data.
- Date-wise profit comparison.
- Recommended booking channel based on predicted outcomes and cost assumptions.
- Chronological model evaluation.

## Technology
Python, Pandas, NumPy, Scikit-learn, Streamlit, Matplotlib, Altair and Joblib.

## Project Structure
- `DemandSense_OTA_Trained_Integrated.py` — dashboard.
- `build_ota_model.py` — OTA training script.
- `notebooks/` — project notebooks.
- `data/` — datasets and forecasts.
- `models/` — trained models.
- `reports/` — evaluation results.

## Setup
Install dependencies:

`python3 -m pip install -r requirements.txt`

Run the dashboard:

`python3 -m streamlit run DemandSense_OTA_Trained_Integrated.py`

## Limitations
The OTA models use synthetic demonstration data. Their results do not establish real-world OTA performance.
