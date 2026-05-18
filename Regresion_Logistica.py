
import math
import numpy as np

class RegresionLogisticaPura:
    def __init__(self, lr=0.01, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        # X e y deben ser listas o arrays simples de NumPy
        X = np.array(X)
        y = np.array(y)
        
        n_samples = len(X)
        n_features = len(X[0])
        
        # Inicializamos los pesos en cero para cada variable
        self.weights = [0.0] * n_features
        self.bias = 0.0

        # Bucle de épocas (iteraciones globales)
        for _ in range(self.epochs):
            # Pasamos por cada uno de los 360 datos individualmente
            for i in range(n_samples):
                # 1. Calcular Z (Xw + b) usando un bucle for para el producto punto
                z = self.bias
                for j in range(n_features):
                    z += X[i][j] * self.weights[j]
                
                # 2. Aplicar la función Sigmoide manual
                # Limitamos z entre -50 y 50 para evitar errores de desbordamiento en math.exp
                z = max(-50, min(50, z))
                y_pred = 1.0 / (1.0 + math.exp(-z))
                
                # 3. Calcular el error del dato actual
                error = y_pred - y[i]
                
                # 4. Actualizar los pesos uno por uno (Gradiente Descendiente)
                for j in range(n_features):
                    self.weights[j] -= self.lr * error * X[i][j]
                self.bias -= self.lr * error

    def predict(self, X):
        predicciones = []
        for i in range(len(X)):
            z = self.bias
            for j in range(len(X[0])):
                z += X[i][j] * self.weights[j]
            
            z = max(-50, min(50, z))
            y_pred = 1.0 / (1.0 + math.exp(-z))
            
            # Clasificación binaria con umbral de 0.5
            if y_pred >= 0.5:
                predicciones.append(1)
            else:
                predicciones.append(0)
        return predicciones
