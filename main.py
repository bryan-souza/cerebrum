import json
import pickle
from typing import List

from typing import Tuple, Union, Type
from pathlib import Path
import numpy as np

from orangecontrib.imageanalytics.image_embedder import ImageEmbedder


# Constants
CEREBRUM_PATH = Path(__file__).resolve().parent
MODELS_PATH   = Path(CEREBRUM_PATH, 'models')
CONFIG_FILE   = Path(CEREBRUM_PATH, 'config.json')


class Identifier(object):
    _instances = {}

    def __new__(cls):
        if cls not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[cls] = instance
        return cls._instances[cls]

    def __init__(self) -> None:
        config = json.load( open(CONFIG_FILE) )

        self.IMG_SIZE         = ( config['img_width'], config['img_height'] )
        self.SECTION_NAMES    = config['section_names']
        self.PLANT_NAMES      = config['plant_names']
        self.MODELS           = {}

        for model in MODELS_PATH.iterdir():
            class_name = str(model.stem)
            self.MODELS.update({ class_name: pickle.load( open(model, 'rb') ) })

    def identify_plant(self, path: Union[str, Path]) -> Tuple[str, float]:
        with ImageEmbedder(model='squeezenet') as emb:
            embeddings = emb([ str(path) ])
            section_pred, _ = self.MODELS['orgaos'].predict(embeddings)
            section = self.SECTION_NAMES[ int(section_pred) ]

            plant_pred, weights = self.MODELS[section].predict(embeddings)
            plant = self.PLANT_NAMES[ int(plant_pred) ]
            accuracy = 100 * np.max( self.softmax(weights) )

            return (plant, accuracy)


    @staticmethod
    def softmax(data: List[float]) -> List[float]:
        return np.exp(data) / np.sum(np.exp(data))
