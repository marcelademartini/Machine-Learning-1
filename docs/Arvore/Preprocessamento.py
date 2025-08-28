import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def preprocessar_dados(df):
    # Substituir valores inválidos (0) por NaN
    cols_invalidas = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in cols_invalidas:
        df[col] = df[col].replace(0, np.nan)

    # Imputação com mediana
    imputer = SimpleImputer(strategy="median")
    df[cols_invalidas] = imputer.fit_transform(df[cols_invalidas])

    # Normalização (padrão z-score)
    scaler = StandardScaler()
    colunas_numericas = df.drop("Outcome", axis=1).columns
    df[colunas_numericas] = scaler.fit_transform(df[colunas_numericas])

    return df
