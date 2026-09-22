Add POST /predict to main.py per specs/api.md. Load boiler_efficiency_dnn.pt
(a PyTorch nn.Sequential: Linear(n,64)-BatchNorm1d-GELU-Dropout(0.1)-
Linear(64,32)-GELU-Linear(32,1)), boiler_efficiency_scaler.pkl, and
boiler_efficiency_meta.pkl from the models/ folder at startup. One-hot
encode the "unit" field to match meta.pkl's saved "features" list before
scaling. Un-scale the model's output using y_mu and y_sd from meta.pkl
before returning it. Add torch, joblib, scikit-learn to requirements.txt.
