import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Manganese Exploration System - Mock Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Models ----------

class PointPredictionRequest(BaseModel):
    latitude: float
    longitude: float
    band_8: float = 1500
    band_11: float = 2100
    band_12: float = 1800


class PointPredictionResponse(BaseModel):
    latitude: float
    longitude: float
    prospectivity_score: float
    confidence: str
    risk_level: str
    recommendation: str


class ForecastSummary(BaseModel):
    current_production_kt: float
    projected_demand_2030_mt: float
    import_deficit_pct: float
    self_sufficiency_pct: float
    yearly_forecast: List[dict]


class DepositPoint(BaseModel):
    latitude: float
    longitude: float
    name: str
    grade: str
    status: str


class GridCell(BaseModel):
    latitude: float
    longitude: float
    score: float


# ---------- Ground Truth Deposit Data (129 Balaghat district deposits) ----------

BALAGHAT_CENTER = (21.8000, 80.1800)
np.random.seed(42)

DEPOSIT_NAMES = [
    "Balaghat North", "Balaghat South", "Waraseoni Block A", "Waraseoni Block B",
    "Lamta Deposit", "Katangi East", "Katangi West", "Khairagarh Ridge",
    "Baihar Zone 1", "Baihar Zone 2", "Kirnapur East", "Kirnapur West",
    "Tirodi Main", "Tirodi Extension", "Bhiwapur Cluster", "Ramtek Corridor",
    "Saugor Belt", "Mandla Fold", "Dindori Block", "Chhindwara Seam",
    "Seoni Ridge", "Nagpur South", "Wardha Valley", "Brahmapuri Grid",
    "Gondia Basin", "Bhandara Strip", "Gadchiroli Edge", "Chandrapur Belt",
    "Yavatmal Zone", "Washim Block", "Akola Strip", "Buldhana Edge",
    "Amravati Grid", "Wardha North", "Nagpur Ridge", "Bhandara East",
    "Gondia West", "Seoni North", "Mandla South", "Dindori East",
    "Balaghat Central", "Waraseoni Core", "Lamta North", "Katangi Central",
    "Baihar South", "Kirnapur Central", "Tirodi South", "Bhiwapur East",
    "Ramtek North", "Saugor East", "Mandla North", "Dindori West",
    "Chhindwara East", "Seoni South", "Nagpur West", "Wardha South",
    "Brahmapuri South", "Gondia East", "Bhandara West", "Gadchiroli West",
    "Chandrapur East", "Yavatmal West", "Washim East", "Akola East",
    "Buldhana West", "Amravati East", "Wardha Central", "Nagpur Central",
    "Balaghat East", "Waraseoni South", "Lamta Central", "Katangi South",
    "Baihar Central", "Kirnapur South", "Tirodi Central", "Bhiwapur West",
    "Ramtek South", "Saugor West", "Mandla Central", "Dindori Central",
    "Chhindwara West", "Seoni Central", "Nagpur South-East", "Wardha East",
    "Brahmapuri West", "Gondia Central", "Bhandara Central", "Gadchiroli Central",
    "Chandrapur West", "Yavatmal Central", "Washim West", "Akola West",
    "Buldhana East", "Amravati West", "Wardha West", "Nagpur North-East",
    "Balaghat West", "Waraseoni West", "Lamta West", "Katangi North",
    "Baihar North-West", "Kirnapur North-West", "Tirodi North", "Bhiwapur North",
    "Ramtek West", "Saugor Central", "Mandla West", "Dindori South",
    "Chhindwara South", "Seoni West", "Nagpur North", "Wardha North-East",
    "Brahmapuri North", "Gondia North", "Bhandara North", "Gadchiroli North",
    "Chandrapur North", "Yavatmal North", "Washim North", "Akola North",
    "Buldhana North", "Amravati North", "Wardha South-West", "Nagpur South-West",
    "Balaghat North-East", "Waraseoni North-East", "Lamta South-East",
    "Katangi North-East", "Baihar South-East", "Kirnapur South-East",
    "Tirodi North-East", "Bhiwapur South-East",
]

GRADES = ["High (40-55% Mn)", "Medium (25-40% Mn)", "Low (10-25% Mn)", "Trace (<10% Mn)"]
STATUSES = ["Active Mine", "Exploration", "Prospecting", "Historical", "Depleted"]

deposits = []
for i in range(129):
    angle = np.random.uniform(0, 2 * np.pi)
    dist = np.random.uniform(0.01, 0.12)
    lat = BALAGHAT_CENTER[0] + dist * np.cos(angle)
    lon = BALAGHAT_CENTER[1] + dist * np.sin(angle)
    grade_idx = np.random.choice([0, 0, 0, 1, 1, 2, 3])
    deposits.append({
        "latitude": round(lat, 6),
        "longitude": round(lon, 6),
        "name": DEPOSIT_NAMES[i % len(DEPOSIT_NAMES)] + f" #{i+1}",
        "grade": GRADES[grade_idx],
        "status": np.random.choice(STATUSES, p=[0.3, 0.3, 0.2, 0.1, 0.1]),
    })


# ---------- Grid Prediction Data (for heatmap) ----------

grid_lats = np.linspace(21.70, 21.90, 40)
grid_lons = np.linspace(80.08, 80.28, 40)

grid_predictions = []
for lat in grid_lats:
    for lon in grid_lons:
        dist_center = np.sqrt((lat - BALAGHAT_CENTER[0])**2 + (lon - BALAGHAT_CENTER[1])**2)
        base_score = max(0, 1 - dist_center * 8)
        noise = np.random.normal(0, 0.12)
        score = float(np.clip(base_score + noise, 0, 1))
        grid_predictions.append({
            "latitude": round(float(lat), 6),
            "longitude": round(float(lon), 6),
            "score": round(score, 4),
        })


# ---------- Endpoints ----------

@app.get("/")
def root():
    return {"message": "Manganese Exploration System - Mock Backend Running", "version": "1.0.0"}


@app.get("/api/v1/forecast/summary", response_model=ForecastSummary)
def get_forecast_summary():
    yearly_forecast = []
    for year in range(2024, 2031):
        production = round(3.38 - (year - 2024) * 0.12 + np.random.normal(0, 0.05), 2)
        demand = round(8.97 + (year - 2024) * 0.45 + np.random.normal(0, 0.08), 2)
        yearly_forecast.append({
            "year": year,
            "domestic_production_mt": max(production, 1.5),
            "total_demand_mt": round(demand, 2),
            "import_gap_mt": round(max(demand - production, 0), 2),
        })

    return ForecastSummary(
        current_production_kt=3380,
        projected_demand_2030_mt=12.32,
        import_deficit_pct=62.3,
        self_sufficiency_pct=37.6,
        yearly_forecast=yearly_forecast,
    )


@app.get("/api/v1/spatial/deposits", response_model=List[DepositPoint])
def get_deposits():
    return deposits


@app.get("/api/v1/spatial/grid-predictions", response_model=List[GridCell])
def get_grid_predictions():
    return grid_predictions


@app.post("/api/v1/spatial/predict-point", response_model=PointPredictionResponse)
def predict_point(req: PointPredictionRequest):
    dist_center = np.sqrt(
        (req.latitude - BALAGHAT_CENTER[0])**2 + (req.longitude - BALAGHAT_CENTER[1])**2
    )
    base_score = max(0, 1 - dist_center * 7)
    band_factor = (req.band_8 / 2000 + req.band_11 / 2500 + req.band_12 / 2200) / 3
    score = float(np.clip(base_score * 0.6 + band_factor * 0.4 + np.random.normal(0, 0.05), 0, 1))
    prospectivity = round(score * 100, 1)

    if prospectivity >= 75:
        confidence, risk, rec = "High", "Low", "Favorable mineral indicators found. Recommend exploratory drilling."
    elif prospectivity >= 50:
        confidence, risk, rec = "Medium", "Medium", "Promising zone. Consider ground truthing survey."
    elif prospectivity >= 25:
        confidence, risk, rec = "Low", "High", "Marginal prospectivity. Additional data recommended."
    else:
        confidence, risk, rec = "Very Low", "Very High", "Unfavorable zone. Not recommended at this time."

    return PointPredictionResponse(
        latitude=req.latitude,
        longitude=req.longitude,
        prospectivity_score=prospectivity,
        confidence=confidence,
        risk_level=risk,
        recommendation=rec,
    )


@app.get("/api/v1/health")
def health_check():
    return {"status": "healthy", "deposits_loaded": len(deposits), "grid_cells": len(grid_predictions)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
