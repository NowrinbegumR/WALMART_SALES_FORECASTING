Project Overview

Purpose: Detailed, Kaggle-ready notebook for weekly retail sales forecasting using ETS, ARIMA, SARIMA, and LSTM.
Notebook: Retail_Sales_Forecasting.ipynb (path: Retail_Sales_Forecasting.ipynb)
Dataset: Expects a CSV (default preferred name Walmart_Store_sales.csv). The notebook auto-discovers CSVs under /kaggle/input/ using find_dataset_path().
How to Run (Kaggle)

Upload: Upload Retail_Sales_Forecasting.ipynb to a Kaggle Notebook or open it in your Kaggle workspace.
Dataset placement: On Kaggle, add the dataset (example used: nowrinbegumr/walmart-store-sales-csv) via
the Notebook > Add data UI; find_dataset_path() will locate it automatically.
Execution order: Run cells top-to-bottom (1 → 9). Do not skip cells; each cell is self-contained and prepares variables for subsequent cells.
Run commands (local dev / optional):
Dependencies

Core Python libs: pandas, numpy, matplotlib, seaborn
Modeling / ML: scikit-learn, statsmodels
Deep learning (LSTM): tensorflow
(imported lazily inside LSTM cells to avoid unnecessary GPU initialization)
Suggested requirements.txt:

pandas>=1.3
numpy>=1.19
matplotlib>=3.4
seaborn>=0.11
scikit-learn>=1.0
statsmodels>=0.13
tensorflow>=2.10
1. Data Understanding: Loads CSV, parses Date (explicit format %d-%m-%Y), aggregates total weekly sales (.asfreq('W-FRI')), fills missing values by time interpolation, and creates train / test splits.
2. EDA: Visualizations (time series, distribution, correlations, top stores) and seasonal decomposition. Helps verify trend/seasonality assumptions.
3. ETS: Exponential smoothing model (level + trend + seasonality). Trains on the first 80% of the weekly series, forecasts the test period, reports metrics.
4. ARIMA: Grid-search over small (p,d,q) values on training data using AIC selection, forecasts test period, reports metrics.
5. SARIMA: Seasonal ARIMA grid search using weekly seasonality (seasonal period default 52 when sufficient data), forecasts test, reports metrics.
6. LSTM: Sliding-window LSTM:
Scales data using MinMaxScaler.
Creates lookback windows (default lookback = 12 weeks).
Imports TensorFlow lazily inside the cell and forces CPU usage to avoid GPU initialization issues (tf.config.set_visible_devices([], 'GPU')).
Trains with early stopping, inverts scaling for evaluation, and plots training history.
7. Model Comparison: Concatenates metrics from ETS, ARIMA, SARIMA, LSTM; ranks models by RMSE and selects best_model_name.
8. Future Sales Forecast: Retrains the selected best model on the full weekly series and forecasts future_steps (default 12 weeks). LSTM retraining uses the same lookback pipeline and iterative forecast loop.
9. Conclusion: Summary and guidance to interpret outputs.

Train / Test Details

Split ratio: First 80% of the aggregated weekly series → train; last 20% → test.
Why: Keeps the temporal order (no leakage). All models are trained only on the training window and forecast the test horizon.
Evaluation metrics: MAE, RMSE, MAPE (implemented as safe_mape to handle zero values).
Best-model selection: The notebook ranks by RMSE on the test set and chooses the top model to produce future forecasts.
Output & Expected Files

Primary outputs (in-notebook):
Plots: time series, decompositions, model forecast overlays, LSTM training history.
Tables: evaluation metrics table
Tables: evaluation metrics table (model_results) and the future_forecast DataFrame for the next 12 weeks.
Optional: Save forecasts to CSV by adding:
Troubleshooting & Notes

Date parsing errors: If pd.to_datetime fails, ensure the Date format in your CSV matches %d-%m-%Y. Modify the format in the Data Understanding cell if needed.
TensorFlow / GPU issues: The notebook forces CPU for LSTM cells to avoid cuInit/cuDNN errors in some environments. If you want GPU:
Remove or comment the tf.config.set_visible_devices([], 'GPU') lines in LSTM and Future

Forecast cells.
Ensure appropriate CUDA/cuDNN versions are installed (matching the tensorflow build).
Missing CSV / path errors: If find_dataset_path() does not find your CSV:
Upload your CSV to the notebook environment or adjust dataset_path manually:
Long grid-searches: ARIMA/SARIMA order searches are limited to small candidate sets to keep runtime reasonable on Kaggle. Expand cautiously—grid size increases runtime exponentially.

Interpreting Results

Metrics table: Lower RMSE/MAE is better; MAPE gives percent error but can be unstable near zero.
Residuals & Diagnostics: For classical models, inspect residuals for autocorrelation and heteroscedasticity. Use ADF test results printed in the ARIMA cell as a stationarity hint.

Next Steps & Variants

Per-store models: To forecast per store rather than aggregated totals, modify aggregation to group by Store and loop over stores to train per-store models (consider hierarchical forecasting approaches if you need consistency).
Hyperparameter tuning: Use pmdarima's auto_arima for automated ARIMA selection or keras-tuner for LSTM hyperparameter search.
Ensembling: Combine ETS/ARIMA/LSTM predictions via weighted average or stacking to potentially improve accuracy.
Deployment: Export the best model and build a small service to serve forecasts (Flask/FastAPI) or save results to cloud storage.

