import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    #variables que almacenan los nombre de los modelos y scalados
    SCALER_MODEL= os.environ.get('SCALER_MODEL')

