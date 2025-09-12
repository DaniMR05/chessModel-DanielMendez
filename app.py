from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from joblib import load
import pathlib
from fastapi.middleware.cors import CORSMiddleware

# Permitir CORS
origins = ["*"]

app = FastAPI(title='Chess Openings Prediction')

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Cargar modelo entrenado
model = load(pathlib.Path('model/openings-v1.joblib'))

# Definir entrada
class InputData(BaseModel):
    num_games: int = 1000
    perf_rating: float = 1800
    avg_player: float = 1700
    draw_pct: float = 25.0   # porcentaje de tablas

# Definir salida
class OutputData(BaseModel):
    predicted_win_pct: float

# Endpoint de predicción
@app.post('/predict', response_model=OutputData)
def predict(data: InputData):
    # Convertir input en array para el modelo
    model_input = np.array([
        data.num_games,
        data.perf_rating,
        data.avg_player,
        data.draw_pct
    ]).reshape(1, -1)

    # Predecir con el modelo de regresión
    result = model.predict(model_input)[0]

    return {'predicted_win_pct': float(result)}
