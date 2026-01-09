# mth_project
Repository for the master thesis project, about the intrusion detection in industrial network.

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Utilities
This project includes several utility tools to help with development:

- **Code Analysis**: See `CODE_ANALYSIS.md` for a comprehensive code review
- **Configuration Management**: Use `config_manager.py` for portable path handling
- **Data Validation**: Use `data_validator.py` to check data quality
- **Project Utilities**: Run `python project_utils.py --diagnostics` to check your setup

For detailed documentation on utilities, see `UTILITIES_README.md` and `SUMMARY.md`.


Check points:
1. Data Collection: Gather sensor values, control signals, and/or network statistics (e.g., bytes/sec, TCP flags) over time.
	1. Fixed sample rate?
	2. Input to the model is raw data from network devices
2. Preprocessing: Normalize, handle missing data, create sequences.
3. Prediction Model: Use TimesFM to forecast future time points.
4. Error Scoring: Compare actual vs. predicted values to compute reconstruction/forecast error.
5. Anomaly Detection: Use thresholds, statistical techniques, or additional ML models to flag anomalies.
Alert & Correlation: Combine with other alerts for final intrusion classification.![image](https://github.com/user-attachments/assets/44a0a70f-a7b8-4a62-9d1f-ff5e7b63b222)

ToC for thesis:

1. Introduction
	a. Overview of OT Cybersecurity Landscape
	b. Role of Machine Learning in Industrial Network Security
	c. Objectives and Scope of the Thesis
2. Theoretical Background
	a. Nature of Industrial Network Data
	b. Industrial Network Diagnostics Techniques
		i. Intrusion Prevention Systems (IPS)
		ii. Early Intrusion Detection Systems (IDS)
	c. Machine Learning and Deep Learning Models for Time Series Analysis
		i. Examination of data dynamics
		ii. Granger Causality for Multivariate Relationships - missing! - should the theory be included, even though it is not implemented?
		iii. Pretrained Models (e.g., TimesFM)
		iv. VAR for multivariate with TimesFM for residuals as enhancement
		v. LSTM for Temporal Forecasting
		vi. CNN for Feature Extraction and Classification
	3. Dataset and Methodology
	a. Data Collection strategy
		3.1.1. Anomaly Definition
		3.1.2. Testbed Infrastructure
		3.1.3. Data Collection Methodology
	b. Data Preprocessing
	c. Conceptual Framework
		3.3.1. Causal Inference Analysis
		3.3.2. Forecasting
		3.3.3. Classification
	d. Model Development
		3.4.1. Forecasting Model
		3.4.2. Classification Model
	e. Challenges and Limitations
4. Experimental Results and Analysis
	a. Evaluation of Forecasting Techniques
		i. TimesFM
		ii. VAR
		iii. Hybrid VAR + TimesFM
		iv. LSTM
	b. Evaluation of Classification Techniques
		i. 1D CNN
	c. Combined Analysis and Comparative Impact
	d. Discussion
5. Future Work and Improvements
	a. Integration into Real-World Systems
	b. Enhancements to Current Models and Framework
	c. Challenges and Limitations
6. Conclusion -> first write what was done, what we couldn’t achieve, future work