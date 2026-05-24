Heartcare: Análisis de Riesgo Cardiovascular
Este proyecto realiza un análisis integral de un dataset médico para predecir el riesgo cardiovascular, comparando la eficiencia de implementaciones manuales frente a librerías optimizadas y analizando la importancia de las variables clínicas.

Info del Dataset
Se selecciono un dataset de Kaggle que trata de enfermedades cardiovasculares.
Este dataset se encuentra como ("Cardiovascular Disease dataset") en la url que se presenta a continuacion: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
Fue seleccionado debido a que estan bastante limpios, ordenados, tiene una buena cantidad de registros,
ademas de ser un tema de interes del grupo al querer saber si podria aportar en algo a la salud publica desde mi carrera.
Razon simplificada: Se Selecciono por capricho del estudiante que redacta este documento.

En este podemos observar que presenta 11 features, 1 target y 70,000 registros medicos.
Presenta variables de los tipos int, categorical y binary, no presenta strings.
Ademas entas variables las clasifican en 3 dependiendo de la procedencia de los datos.
Estas categorias son:


Objetivo : información objetiva                                  #Variable tomada en el lugar normalmmente
Examen : resultados del examen médico                            #Variable tomada mediante examenes medicos XD
Subjetivo : información proporcionada por el paciente.           #Como su nombre indica la proporciona el paciente y se espera que no mintiera en algo tan importante

Esta clasificacion no cambia mucho lo que es el entrenamiento ni el proceso de manejar los datos
solo son consideraciones que se deben tomar a la hora de interpretar los resultados o trabajarlos.
Ademas es mas profesional XD.

Las variables que encontramos en este documento son:

Edad | Característica objetiva | edad | int (días)
Altura | Característica objetivo | altura | int (cm) |
Peso | Característica del objetivo | peso | flotación (kg) |
Género | Característica objetiva | género | código categórico |
Presión arterial sistólica | Característica de examen | ap_hi | int |
Presión arterial diastólica | Característica de examen | ap_lo | int |
Colesterol | Característica del examen | colesterol | 1: normal, 2: por encima de lo normal, 3: muy por encima de lo normal |
Glucosa | Característica del examen | gluc | 1: normal, 2: por encima de lo normal, 3: muy por encima de lo normal |
Fumar | Característica subjetiva | humo | binario |
Consumo de alcohol | Característica subjetiva | alco | binario |
Actividad física | Característica subjetiva | activo | binario |
Presencia o ausencia de enfermedad cardiovascular | Variable objetivo | cardio | binario |

Características Principales
Ingenieria de variables: Se modifico la varible edad para que estuviera en años y se creo una nueva variable IMC a partir de peso y altura eliminando asi estas dos para dejar solo IMC
Análisis Exploratorio (EDA): Visualización de relaciones complejas mediante matrices de dispersión (Pairplots) e histogramas de densidad.
Modelado Predictivo: Uso de Lasso Regression para selección de variables y XGBoost para clasificación de alta precisión.
Estudio de Complejidad: Comparativa asintótica entre algoritmos manuales (O(N \cdot M)) y optimizados mediante vectorización y estructuras de datos avanzadas (O(N)  O(T \log N)).
Estructura: Se siguio una estructura aproximada a POO con algunas areas de mejora notables pero en su mayoria las partes importantes van por clases

Hallazgos Clave
Variable Crítica: La presión arterial sistólica (ap_hi) se identificó como el predictor con mayor ganancia de información (superior al 50% de importancia relativa).
Modelos: Se comprobo que el uso de mmodelos predictivos podria hacer un cambio en el ambito medico mientras se trabaje a fondo y se pula
Modelo a implementar: Se encontro hasta el momento que modelos de prediccion no parammetricos como XGBoost dan un mejor resultado en cuanto a detectar a la gran parte de los enfermos con minimos cambios
Eficiencia: Las implementaciones de Sklearn demostraron una estabilidad superior frente al ruido de ejecución gracias a la vectorización de bajo nivel.🛠️ 

Tecnologías Utilizadas
Lenguaje: Python 3.x
Entorno: Google Colab
Librerías: * Pandas & NumPy (Manejo de datos)Seaborn & Matplotlib (Visualización)Scikit-Learn (Lasso, k-NN, Preprocesamiento)XGBoost (Clasificación avanzada)

Resultados de Complejidad
El análisis asintótico incluido en el notebook Main.ipynb muestra cómo el tiempo de ejecución escala según el tamaño del dataset (N):Manual: Sensible al overhead de Python en muestras pequeñas.
Optimizado: Comportamiento estable que sigue las curvas teóricas de complejidad computacional.
