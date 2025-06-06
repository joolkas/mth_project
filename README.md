# mth_project
Repository for the master thesis project, about the intrusion detection in industrial network.


Check points:
1. Data Collection: Gather sensor values, control signals, and/or network statistics (e.g., bytes/sec, TCP flags) over time.
	1. Fixed sample rate?
	2. Input to the model is raw data from network devices
2. Preprocessing: Normalize, handle missing data, create sequences.
3. Prediction Model: Use TimesFM to forecast future time points.
4. Error Scoring: Compare actual vs. predicted values to compute reconstruction/forecast error.
5. Anomaly Detection: Use thresholds, statistical techniques, or additional ML models to flag anomalies.
Alert & Correlation: Combine with other alerts for final intrusion classification.![image](https://github.com/user-attachments/assets/44a0a70f-a7b8-4a62-9d1f-ff5e7b63b222)
