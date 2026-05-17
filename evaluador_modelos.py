import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

class EvaluadorModelos:
    def __init__(self, preprocesador_obj):
        """
        Recibe el objeto preprocesador para extraer su ColumnTransformer 
        y dejar armados los Pipelines de forma automatizada.
        """
        self.transformador = preprocesador_obj.obtener_preprocessor()
        
        # Dejamos definidos los 5 modelos estables envueltos en sus Pipelines seguros
        self.modelos = {
            "Regresión Logística": Pipeline([('prep', self.transformador), ('mod', LogisticRegression(max_iter=1000, random_state=42))]),
            "Árbol de Decisión": Pipeline([('prep', self.transformador), ('mod', DecisionTreeClassifier(max_depth=10, random_state=42))]),
            "Random Forest": Pipeline([('prep', self.transformador), ('mod', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))]),
            # Reemplaza la línea de Gradient Boosting por esta:
            "Naive Bayes": Pipeline([('prep', self.transformador), ('mod', GaussianNB())]),
            "k-NN": Pipeline([('prep', self.transformador), ('mod', KNeighborsClassifier(n_neighbors=5))])
        }
        
    def evaluar_validacion_cruzada(self, x_train, y_train):
        """
        Calcula el Cross-Validation sobre los datos en bruto 
        y devuelve una tabla comparativa limpia.
        """
        nombres = []
        medias = []
        desviaciones = []
        
        for nombre, pipeline in self.modelos.items():
            puntuaciones = cross_val_score(pipeline, x_train, y_train, cv=5, scoring='accuracy')
            
            nombres.append(nombre)
            medias.append(puntuaciones.mean())
            desviaciones.append(puntuaciones.std()) # <-- Aquí está en minúscula
            
        # Construimos la tabla de resultados
        df_res = pd.DataFrame({
            'Modelo': nombres,
            'Exactitud Promedio (Accuracy)': medias,
            'Variabilidad (Std)': desviaciones # <-- ¡CÁMBIALA AQUÍ A MINÚSCULA!
        })
        return df_res.sort_values(by='Exactitud Promedio (Accuracy)', ascending=False).reset_index(drop=True)

    def obtener_predicciones_y_metricas(self, nombre_modelo, x_train, y_train, x_val, y_val):
        """
        Entrena un modelo específico y devuelve sus predicciones reales,
        su reporte de texto y los datos necesarios para la matriz de confusión.
        """
        if nombre_modelo not in self.modelos:
            print(f"El modelo '{nombre_modelo}' no existe en este evaluador.")
            return None
            
        pipeline = self.modelos[nombre_modelo]
        
        # Entrenamos el pipeline completo (Transformación + Modelo)
        pipeline.fit(x_train, y_train)
        
        # Predecimos sobre el conjunto de validación en bruto
        preds = pipeline.predict(x_val)
        
        # Extraemos las probabilidades para la curva ROC
        if hasattr(pipeline.named_steps['mod'], "predict_proba"):
            probs = pipeline.predict_proba(x_val)[:, 1]
        else:
            probs = pipeline.decision_function(x_val)
            
        reporte = classification_report(y_val, preds, target_names=['Sano (0)', 'Enfermo (1)'])
        cm = confusion_matrix(y_val, preds)
        
        return preds, probs, reporte, cm
        
    def obtener_todos_los_modelos(self):
        return self.modelos