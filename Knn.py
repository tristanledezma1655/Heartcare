import math
class KNNPuro:
    def __init__(self, k=5):
        self.k = k
        self.X_train = []
        self.y_train = []

    def fit(self, X, y):
        # En k-NN manual solo guardamos la muestra en memoria
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        predicciones = []
        
        # Para cada paciente en el conjunto de prueba...
        for x_test_individual in X_test:
            distancias = []
            
            # ...calculamos la distancia contra cada uno de los 360 pacientes de entrenamiento
            for idx, x_train_individual in enumerate(self.X_train):
                suma_cuadrados = 0.0
                
                # Bucle para recorrer cada variable/columna una por una
                for j in range(len(x_test_individual)):
                    suma_cuadrados += (x_test_individual[j] - x_train_individual[j]) ** 2
                
                distancia_euclidiana = math.sqrt(suma_cuadrados)
                # Guardamos la distancia junto con la etiqueta (sano/enfermo) de ese vecino
                distancias.append((distancia_euclidiana, self.y_train[idx]))
            
            # Ordenamos las distancias de menor a mayor (Fuerza Bruta pura)
            distancias.sort(key=lambda x: x[0])
            
            # Tomamos los K vecinos más cercanos
            k_vecinos = distancias[:self.k]
            
            # Contamos los votos (cuántos enfermos y cuántos sanos)
            votos_enfermos = sum(1 for d, etiqueta in k_vecinos if etiqueta == 1)
            votos_sanos = self.k - votos_enfermos
            
            # El voto mayoritario gana
            if votos_enfermos > votos_sanos:
                predicciones.append(1)
            else:
                predicciones.append(0)
                
        return predicciones
