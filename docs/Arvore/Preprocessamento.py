# Preprocessamento.py
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Colunas onde zero é inválido fisiologicamente (tratamos como ausente)
COLS_ZERO_AS_NA = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

def preprocessar(df: pd.DataFrame):
    df = df.copy()

    # Substitui zeros por NaN nas colunas específicas
    df[COLS_ZERO_AS_NA] = df[COLS_ZERO_AS_NA].replace(0, pd.NA)

    # Imputação por mediana
    imputer = SimpleImputer(strategy='median')
    df[COLS_ZERO_AS_NA] = imputer.fit_transform(df[COLS_ZERO_AS_NA])

    # Separa X e y
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"].astype(int)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X_scaled, y, imputer, scaler
