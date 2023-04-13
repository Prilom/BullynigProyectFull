from __future__ import absolute_import

import argparse
import multiprocessing as mp
import logging
import tempfile
import os
import pickle
import gensim
import pandas as pd
import numpy as np
import tensorflow as tf

from datetime import datetime



def train_and_evaluate(
    work_dir, train_df, eval_df, batch_size=1024, epochs=8, steps=1000
    ):
    # Obtenemos la fecha actual en formato AAAA-MM-DD
    today = datetime.today().strftime('%Y-%m-%d')

    # Creamos el directorio del modelo en una carpeta con la fecha actual en el nombre
    model_dir = os.path.join(work_dir, "data/model_" + today)

    # Comprobamos si ya existe un modelo
    if tf.io.gfile.exists(model_dir):
        tf.io.gfile.rmtree(model_dir) #si existe lo eliminamos
    tf.io.gfile.mkdir(model_dir) #creamos un directorio de modelo

    # Configuramos donde guardar el modelo
    run_config = tf.estimator.RunConfig() 
    run_config = run_config.replace(model_dir=model_dir)

    # Nos permite seguir el entrenamiento cada 10 steps
    run_config = run_config.replace(save_summary_steps=10)

    #Configuramos el modelo 

    


