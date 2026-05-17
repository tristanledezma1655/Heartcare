import mysql.connector
import pandas as pd
from mysql.connector import Error

class gestorDB:
  # Por defecto lo dejamos en 'local' como querías
  def __init__(self, entorno='local'):
    self.entorno = entorno.lower()

    # El INTERRUPTOR corregido
    if self.entorno == 'local':
      # AQUÍ VAN TUS DATOS DE TU COMPUTADORA (XAMPP / Workbench)
      self.config = {
        "host": "localhost",
        "user": "root",
        "password": "",
        "database": "tu_base_local", # <-- Reemplaza por el nombre real de tu BD en XAMPP
        "port": 3306
      }
    else:
      # AQUÍ VAN LOS DATOS DE LA NUBE (filess.io)
      self.config = {
        "host": "2ps03y.h.filess.io",
        "user": "Heartcare_instantmud",
        "password": "f226c276a2e6c7aac3dc37870d32a781de3dd29d",
        "database": "Heartcare_instantmud",
        "port": 61002 
      }

  def obtener_datos(self, vista = "dataset_completo"):
    query = f"SELECT * FROM {vista}"
    connection = None
    try:
      connection = mysql.connector.connect(**self.config)
      df = pd.read_sql(query, connection)
      print(f"✅ Datos cargados correctamente desde [{self.entorno.upper()}] - Vista: {vista}")
      return df
    except Error as e:
      print(f"❌ Error al conectar a la base de datos ({self.entorno.upper()}): {e}")
      return None
    finally:
      if connection and connection.is_connected():
        connection.close()
