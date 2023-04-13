from __future__ import absolute_import

import argparse
import logging
import os
import csv
import random
import pickle

from datetime import datetime

import tensorflow as tf
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.io import ReadFromText #fichero de texto a colección
from apache_beam.io import WriteToText #colección a fichero de texto
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions, DirectOptions

from core.config import Config

settings = Config()

class preprocess(beam.DoFn):
    """clase que contiene los métodos para  aplicar transformaciones y limpieza a tus datos. Por ejemplo, podrías eliminar duplicados, 
    filtrar datos irrelevantes, convertir formatos de fecha, agrupar datos, etc. El resultado de esta fase sería un conjunto de datos 
    limpios y listos para ser utilizados en el entrenamiento """

    def process(self, element):
        pass
      
class scaler(beam.DoFn):
    """esta clasee realiza el escalado y tiene que almacenar el scaler para procesar los datos de test, para ello la clase setup crea una carpeta para almacenar el 
    scaler.pkl"""
    def __init__(self):
        self.scaler = None
        self.model_dir = None    

    def setup(self, work_dir):
        """creamos la carpeta donde se almacenaran los modelos"""
        # Obtenemos la fecha actual en formato AAAA-MM-DD
        today = datetime.today().strftime('%Y-%m-%d')

      # Creamos el directorio del modelo en una carpeta con la fecha actual en el nombre
        self.model_dir = os.path.join(work_dir, "data/model_" + today)

        # Comprobamos si ya existe la carpeta para esa fecha
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir) #si no existe la carpeta la creamos

        # Configuramos donde guardar el modelo
        run_config = tf.estimator.RunConfig() 
        run_config = run_config.replace(model_dir=self.model_dir)

        # Nos permite seguir el entrenamiento cada 10 steps
        run_config = run_config.replace(save_summary_steps=10)

    def create_scaler(self, data):
        # introduciomos el codigo para el scalado
        ##############################################
        ##############################################

        #almacenamos el scaler en el directorio
        pickle.dump(
        self.scaler,
        tf.io.gfile.GFile(os.path.join(self.model_dir, settings.SCALER_MODEL), mode="wb"),
        protocol=0,
       )
        return None #aqui retornamos la data scalada
    
    def transform_with_scaler(self,data):
        return None #aqui retornamos la data scalada
        
    def process(self, data, mode):
        if mode == 'train':
            scaler_data = self.create_scaler(data)
        else:
            scaler_data = self.transform_with_scaler(data)
        yield scaler_data

def split_dataset(element, num_partitions, eval_percent):
    """usa el valor de eval_percent y la función random.uniform para decidir a qué partición enviar cada elemento."""
    if random.uniform(0, 100) < eval_percent:
        return 1
    else:
        return 0
#######################################################################################################################      
######################################Configuramos y creamos el Pipeline###############################################
#######################################################################################################################

def run (arg=None, save_main_session= True):
    """método de entrada al pipeline, con save_main_session= True indica que se deben guardar los objetos del espacio 
    de nombres global de la función principal (__main__) en una caché, para que estén disponibles para su reutilización
    en los procesos secundarios que se ejecuten en la pipeline"""

    parser = argparse.ArgumentParser() #Creamos una instacia al parseador de argumentos

    parser.add_argument( #añado argumento y lo guardo en la variable work_dir. Directorio de trabajo
        "--work-dir", dest="work_dir", required=True, help="Working directory",
  )

    parser.add_argument( # añado argumento. Fichero de entrada
        "--input", dest="input", required=True, help="Input dataset in work dir",
    )
    parser.add_argument( # añado argumento. Fichero de salida
        "--output",
        dest="output",
        required=True,
        help="Output path to store transformed data in work dir",
    )
    parser.add_argument( # añado argumento. Indica si estamos entrenando o evaluando
        "--mode",
        dest="mode",
        required=True,
        choices=["train", "test"],
        help="Type of output to store transformed data",
    )

    known_args, pipeline_args = parser.parse_known_args(arg) #recogemos los argumentos enviados por consola y los almacenamos

    # Añadimos la configuración de la pipeline
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session #permite usar dependencias externas como nltk
    pipeline_options.view_as(DirectOptions).direct_num_workers = 0 #número de workers. 0 para que coja el número máximo que tiene la máquina

    #Construimos la pipeline

    with beam.Pipeline(options=pipeline_options) as p:
        #Leemos el csv con los datos a traves de una pCollection en este punto se pueden meter mas metodos del preprocesado
        data = p | "ReadCSV" >> ReadFromText(
             known_args.input
        )

        if known_args.mode == 'train':
             #si estamos en modo train realizamos las sigueientes trasnformaciones
            transformed_data = (
                data 
                | "PreprocessingData" >> beam.ParDo(preprocess())
                | "ScalingData" >> beam.ParDo(scaler(known_args.work_dir))
            )
            #Vamos a dividir el set en train y validación, para eso nos ayudaremos de la funcion creada split_dataset y la funcion Partition de beam
            eval_percent = 20  #porcentaje de set de evaluación
            train_dataset, eval_dataset = (
            transformed_data
            | "SplitData" >>
              beam.Partition(num_partitions=2, partition_fn=lambda elem, num_partitions: split_dataset(elem, num_partitions, eval_percent))
            | beam.GroupByKey()
            | beam.Partition(fn=lambda index, _: 0 if index == 0 else 1, num_partitions=2) # Reasignamos los índices de las particiones
            )

            #almacenamos los dos set en la carpeta output en formato csv
            train_dataset | "TrainWriteToCSV" >> WriteToText( # escribimos el conjunto de train en un csv
                os.path.join(known_args.output, "train", "part") #part genera las distintas particiones para cada fichero. Es el prefijo del fichero.
            )
            eval_dataset | "EvalWriteToCSV" >> WriteToText( # escribimos el conjunto de eval en un csv
                os.path.join(known_args.output, "eval", "part") #part genera las distintas particiones para cada fichero. Es el prefijo del fichero.
            )
        else: # known_args.mode == "test"
            transformed_data = (
                data 
                | "PreprocessingData" >> beam.ParDo(preprocess())
                | "ScalingData" >> beam.ParDo(scaler(known_args.mode))
            )
            transformed_data | "TestWriteToCSV" >> WriteToText(
                os.path.join(known_args.output, "test", "part") #escribimos de nuevo a un fichero de texto.
            )

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()






