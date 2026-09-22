# API — POST /predict

Request JSON:
{
  "load_pct": float,
  "excess_o2_pct": float,
  "coal_gcv_kcal_kg": float,
  "coal_moisture_pct": float,
  "coal_ash_pct": float,
  "hrs_since_soot_blow": float,
  "ambient_temp_C": float,
  "flue_gas_temp_C": float,
  "unit": "CPP-1" | "CPP-2" | "CPP-3"
}

Response 200:
{ "predicted_efficiency_pct": float }

Errors:
- 400 — missing required field, names the field
- 422 — value out of physically sensible range (e.g. excess_o2_pct outside 0.5–8.0)
- 500 — model file failed to load
