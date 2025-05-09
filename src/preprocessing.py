import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess_data(path):
    df = pd.read_csv(path)

    # Eliminar filas con valores nulos
    df = df.dropna()

    # Codificar todas las columnas categóricas (tipo object)
    label_encoders = {}
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Separar features y target
    X = df.drop("target", axis=1)
    y = df["target"]

    # Escalar características numéricas
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dividir dataset
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
