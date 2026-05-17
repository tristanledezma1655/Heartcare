import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

class preprocesador:
    def __init__(self, ruta_csv):
        self.ruta = ruta_csv
        self.df = None
        self.df_limpio = None
        self.preprocessor = None

    def cargar_BD(self, dataframe):
        self.df = dataframe.copy()
        print(f"Datos cargados desde la base de datos: {self.df.shape}")

    def cargar_CSV(self):
        try:
            self.df = pd.read_csv(self.ruta, sep=";")
            self.df = self.df.drop("id", axis=1, errors="ignore")
            print("Datos cargados correctamente")
            print(f"Forma del dataset: {self.df.shape}")
        except FileNotFoundError:
            print("No se encontró el archivo CSV en la ruta especificada.")

    # 1. LIMPIEZA DE DATOS ORIGINAL
    def limpiar_datos(self):
        if self.df is None:
            print("Error: No hay datos cargados para limpiar.")
            return

        temp_df = self.df.copy()
        antes = len(temp_df)
        temp_df = temp_df.dropna().drop_duplicates()

        if "age" in temp_df.columns:
            temp_df["age"] = (temp_df["age"] / 365).astype(int)

        temp_df = temp_df[(temp_df["ap_hi"] <= 250) & (temp_df["ap_hi"] >= 40)]
        temp_df = temp_df[(temp_df["ap_lo"] <= 150) & (temp_df["ap_lo"] >= 30)]
        temp_df = temp_df[(temp_df["ap_hi"] > temp_df["ap_lo"])]
        print(f"Los registros despues de filtrar por ap: {temp_df.shape}")

        if "weight" in temp_df.columns and "height" in temp_df.columns:
            temp_df["imc"] = temp_df["weight"] / ((temp_df["height"] / 100) ** 2)

        temp_df = temp_df[(temp_df["weight"] <= 250) & (temp_df["weight"] >= 35)]
        temp_df = temp_df[(temp_df["height"] <= 230) & (temp_df["height"] >= 130)]
        temp_df = temp_df[(temp_df["imc"] <= 55) & (temp_df["imc"] >= 12)]
        print(f"Los registros despues de filtrar por peso y altura: {temp_df.shape}")

        print(f"Registros eliminados: {antes - len(temp_df)}")
        self.df_limpio = temp_df

    # 2. MUESTREO ESTADÍSTICO INTERMEDIO
    def muestrear_datos(self, margen_error=0.03):
        if self.df_limpio is None:
            print("Error: Primero debes ejecutar limpiar_datos() antes de muestrear.")
            return None

        N = len(self.df_limpio)
        Z, P, Q = 1.96, 0.5, 0.5
        E = margen_error

        numerador = (Z**2) * P * Q * N
        denominador = (E**2) * (N - 1) + (Z**2) * P * Q
        n_objetivo = math.ceil(numerador / denominador)
        porcentaje_muestra = n_objetivo / N

        df_muestra, _ = train_test_split(
            self.df_limpio,
            train_size=porcentaje_muestra,
            random_state=42,
            stratify=self.df_limpio['cardio']
        )
        print(f"\n Muestreo Estadístico (95% Confianza / {E*100}% Error):")
        print(f"-> Muestra representativa generada: {df_muestra.shape[0]} registros.")
        return df_muestra

   # 3. CONTEXTO DE TRANSFORMACIÓN CORREGIDO (Sin entrenar los datos antes de tiempo)
    def transformar_datos(self, df_muestreado, target='cardio'):
        colums_ign = [target, "height", "weight"]
        x = df_muestreado.drop(colums_ign, axis=1)
        y = df_muestreado[target]

        # splits en bruto
        x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
        x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

        # Configura escaladores y codificadores
        scaler = StandardScaler()
        codificador = OneHotEncoder(drop="first", sparse_output=False)

        cols_num = ["age", "ap_hi", "ap_lo", "imc"]
        cols_cat = ["gender", "cholesterol", "gluc"]
        cols_bin = ["smoke", "alco", "active"]

        # estructura en la variable de la clase
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', scaler, cols_num),
                ('cat', codificador, cols_cat),
                ('bin', 'passthrough', cols_bin)
            ], remainder="drop"
        )

        # Regresamos los conjuntos limpios pero en bruto.
        return x_train, y_train, x_val, y_val, x_test, y_test

    def obtener_preprocessor(self):
        return self.preprocessor
