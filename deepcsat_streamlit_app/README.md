# DeepCSAT - Local Deployment (Streamlit)

## Setup (one-time)

```
pip install -r requirements.txt
```

If `saved_model/` is missing or you want to retrain on your own machine
(recommended, since it should also fix the earlier NLTK download issue):

```
python build_deployment_artifacts.py
```

This reads `eCommerce_Customer_support_data.csv` (must be in this folder),
replicates the full cleaning/feature-engineering/modeling pipeline from the
ML notebook, and writes everything the app needs into `saved_model/`.

## Run the app

```
python -m streamlit run app.py
```

(Use `python -m streamlit run app.py`, not bare `streamlit run app.py` -
this avoids the PATH issue that's bitten you before on Windows.)

This opens a browser tab where you can enter interaction details (channel,
category, response time, item price, customer remarks, etc.) and get a
predicted CSAT score (1-5) with a probability breakdown, using the exact
same ANN model and preprocessing pipeline built and validated in the ML
notebook.

## Files
- `app.py` - Streamlit UI (thin layer, no business logic)
- `csat_inference.py` - the actual preprocessing + prediction logic (tested
  independently of the UI - see `test_app_logic.py` in the ML working files)
- `build_deployment_artifacts.py` - retrains the model and saves the full
  artifact bundle needed for single-row inference (model + scaler + TF-IDF +
  SVD + medians + outlier bounds + frequency-encoding maps + category lists)
- `saved_model/` - the pre-built artifacts (already included, but you can
  regenerate them by running `build_deployment_artifacts.py`)
