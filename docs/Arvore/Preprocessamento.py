import pandas as pd
from sklearn.impute import SimpleImputer

def preprocess_data(df):
    # Substituir zeros por NaN em colunas onde zero não faz sentido
    cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_with_zero] = df[cols_with_zero].replace(0, pd.NA)

    # Imputação: preencher valores ausentes com mediana
    imputer = SimpleImputer(strategy='median')
    df[cols_with_zero] = imputer.fit_transform(df[cols_with_zero])

    # Separar features e target
    x = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    
    return x, y

# Exemplo de uso
# df = pd.read_csv("https://raw.githubusercontent.com/...")
# x, y = preprocess_data(df)
