# AquaClass Global 2.0

A Streamlit application for worldwide water-quality screening and explainable analysis.

## Main capabilities
- Multiple CSV upload
- Automatic recognition of common parameter aliases
- Parameter classification:
  pH, turbidity, TDS, dissolved oxygen, nitrate, BOD, COD,
  conductivity, coliform indicator and temperature
- Good / Moderate / Poor / Very Poor classification
- Overall water-quality classifier
- Per-record transparent parameter score chart
- Global observation map from latitude/longitude
- Country and year filters
- Time-series and observation-coverage charts
- Historical water-issue reference map
- Optional Random Forest model for labeled datasets
- CSV export

## Recommended global data architecture

AquaClass should distinguish:
1. **Observed water-quality measurements** — laboratory/monitoring observations.
2. **Historical water-related events** — spills, floods, drought/water scarcity, contamination events.
3. **Country-level indicators** — useful for context but not interchangeable with water-quality measurements.

For large global research datasets, use authoritative sources and retain:
`source`, `dataset`, `station_id`, `country`, `location`, `latitude`, `longitude`,
`date`, `parameter`, `value`, `unit`, `method`, `quality_flag`.

UNEP GEMS/Water GEMStat is a particularly relevant source: the official GEMStat information
reports tens of millions of measurements, thousands of monitoring stations, hundreds of
parameters, and records extending back to the early 1900s. The public archive should be
downloaded/processed separately rather than bundled into this GitHub repository because of its
size.

## Deployment
1. Upload `app.py`, `requirements.txt`, and `historical_water_issues.csv` to GitHub.
2. Create a Streamlit Community Cloud app from the repository.
3. Set the main file to `app.py`.
4. No API key is required for the core analysis.

## Scientific safeguards
The default thresholds are generic screening bands, not a universal global standard.
For publication or regulatory use, implement a standards layer that records:
- jurisdiction
- intended water use
- parameter
- unit
- threshold
- standard name
- standard version/date

Never combine measurements with incompatible units or analytical methods without normalization.
