from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from joblib import dump
import pandas as pd
import pathlib

# ---------------------------
# Cargar dataset
# ---------------------------
df = pd.read_csv(pathlib.Path('data/openings.csv'))

# Limpiar espacios en los nombres de columnas
df.columns = df.columns.str.strip()

# ---------------------------
# Definir target y features
# ---------------------------
y = df.pop('Player Win %')  # Target: porcentaje de victorias
X = df[['Num Games', 'Perf Rating', 'Avg Player', 'Draw %']]  # Features numéricas

# ---------------------------
# Dividir en entrenamiento y prueba
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Entrenar modelo
# ---------------------------
print('Training model...')
clf = RandomForestRegressor(
    n_estimators=10,
    max_depth=2,
    random_state=0
)
clf.fit(X_train, y_train)

# ---------------------------
# Guardar modelo
# ---------------------------
print('Saving model...')
dump(clf, pathlib.Path('model/openings-v1.joblib'))

print('Model trained and saved successfully!')
