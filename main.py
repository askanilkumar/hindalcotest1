from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import OneHotEncoder

app = FastAPI()

# Load model artifacts at startup
MODEL_PATH = "models/boiler_efficiency_dnn.pt"
SCALER_PATH = "models/boiler_efficiency_scaler.pkl"
META_PATH = "models/boiler_efficiency_meta.pkl"

meta = joblib.load(META_PATH)
scaler = joblib.load(SCALER_PATH)

# Define the model architecture per spec
n_features = len(meta["features"])
model = nn.Sequential(
    nn.Linear(n_features, 64),
    nn.BatchNorm1d(64),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(64, 32),
    nn.GELU(),
    nn.Linear(32, 1)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

y_mu = meta["y_mu"]
y_sd = meta["y_sd"]
features = meta["features"]

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(request: Request):
    try:
        input_data = await request.json()

        unit_value = input_data.pop("unit", None)
        if unit_value is None:
            return JSONResponse(status_code=400, content={"error": "Missing 'unit' field in input"})

        x_dict = {feat: 0 for feat in features}

        for k, v in input_data.items():
            if k in x_dict:
                x_dict[k] = v

        unit_feature_name = f"unit_{unit_value}"
        if unit_feature_name not in x_dict:
            return JSONResponse(status_code=400, content={"error": f"Unknown unit value: {unit_value}"})
        x_dict[unit_feature_name] = 1

        x = [x_dict[feat] for feat in features]

        x_scaled = scaler.transform([x])
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

        with torch.no_grad():
            y_pred_scaled = model(x_tensor).item()

        y_pred = (y_pred_scaled * y_sd) + y_mu

        return {"prediction": y_pred}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})