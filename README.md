#  HeartCare: Análisis Predictivo de Riesgo Cardiovascular

Este proyecto realiza un análisis integral de un dataset médico de 70,000 registros para predecir el riesgo cardiovascular, comparando la eficiencia de implementaciones manuales frente a librerías optimizadas.

##  Descripción del Dataset
Se utiliza el **Cardiovascular Disease Dataset** (Kaggle), que contiene información clínica y de estilo de vida.

### Categorización de Variables:
| Categoría | Descripción | Ejemplos |
| :--- | :--- | :--- |
| **Objetiva** | Datos fisiológicos directos | Edad, Peso, Género |
| **Examen** | Resultados de pruebas clínicas | Presión arterial, Colesterol |
| **Subjetiva** | Información proporcionada por el paciente | Tabaquismo, Actividad física |

##  Ingeniería de Variables y Preprocesamiento
* **Transformación Temporal:** Conversión de edad (días) a años.
* **Nuevas Métricas:** Cálculo del **IMC** (Índice de Masa Corporal) a partir de peso y altura.
* **Limpieza:** Eliminación de valores atípicos fisiológicamente imposibles (ej. presiones arteriales extremas).
* **Escalado:** Aplicación de `StandardScaler` y `One-Hot Encoding` mediante Pipelines.

##  Modelado y Algoritmos
Se evaluaron 5 modelos, destacando la comparativa entre:
1. **Regresión Logística:** Modelo base para interpretación clínica.
2. **XGBoost:** Modelo final optimizado mediante Boosting para capturar relaciones no lineales.

### Análisis de Complejidad Computacional
Se incluyó un estudio comparativo de eficiencia:
* **Manual:** $O(N \cdot M)$ - Sensible al overhead de Python.
* **Optimizado (Sklearn/XGBoost):** $O(N)$ o $O(T \log N)$ gracias a la vectorización y paralelismo.

##  Hallazgos Clave
* **Variable Crítica:** La **Presión Arterial Sistólica** aporta más del 50% de la importancia relativa para la predicción.
* **Métrica Estrella:** Se priorizó el **Recall (91%)** sobre la Precisión para minimizar los Falsos Negativos en un entorno médico.
* **Arquitectura:** Estructura basada en Programación Orientada a Objetos (POO) para facilitar la escalabilidad.

##  Tecnologías
`Python 3.x` | `Google Colab` | `Pandas` | `Scikit-Learn` | `XGBoost` | `Seaborn`

##  Instrucciones de Ejecución

Para replicar este análisis en **Google Colab**, sigue estos pasos:

1. **Configuración del entorno:** Ejecuta las dos primeras celdas del notebook para clonar automáticamente el repositorio y posicionarte en el directorio de trabajo.
2. **Ignorar secciones de desarrollo:** Las celdas marcadas para *cambios y commits* están deshabilitadas, ya que corresponden a la etapa de desarrollo y finalización del código.
3. **Flujo de ejecución:** Puedes ejecutar el resto del notebook de forma secuencial para observar el preprocesamiento, el entrenamiento y la comparativa de modelos.
