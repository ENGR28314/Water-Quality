# AquaClass — Explainable Water Quality Classification

A production-oriented Streamlit starter application for global water-quality screening.

## Features
- Upload a CSV and automatically recognize common aliases for:
  pH, Turbidity, TDS, Dissolved Oxygen, Nitrate, BOD, COD,
  Conductivity, Coliform Indicator, Temperature.
- Classifies every parameter as Good / Moderate / Poor / Very Poor.
- Produces an overall Water-quality classifier.
- Parameter score chart and transparent screening logic.
- World map for uploaded observations using Latitude/Longitude.
- Bundled historical water-issue map and table.
- Optional Random Forest training when a labeled target column is present.
- Export analyzed results to CSV.
- No hard-coded final result for uploaded data: calculations respond to input rows.

## CSV columns
Minimum recommended columns:
Location, Country, Latitude, Longitude, Date, pH, Turbidity, TDS,
Dissolved Oxygen, Nitrate, BOD, COD, Conductivity,
Coliform Indicator, Temperature

Common aliases such as `ph`, `do`, `ec`, `ntu`, `temp`, and `no3` are accepted.

## Run locally
pip install -r requirements.txt
streamlit run app.py

## Deploy to Streamlit Community Cloud
Push all three files to GitHub:
- app.py
- requirements.txt
- historical_water_issues.csv

Then create a Streamlit deployment from the GitHub repository.

## Important scientific note
The default parameter bands are generic screening bands intended for software demonstration.
They are NOT a universal drinking-water, ecological-water, wastewater-discharge, or public-health standard.
For a research/production deployment, replace the defaults with the exact standard selected
for the intended water use and jurisdiction, document units, and preserve the standard/version
in the dataset metadata.

## Scaling to worldwide historical data
The included historical CSV is deliberately a small starter dataset. For a serious global research
edition, connect/import authoritative time-series sources and normalize them into a common schema
with source, date, location, units, methodology, and uncertainty fields. The app's uploader already
supports replacing the starter dataset with larger curated files.
