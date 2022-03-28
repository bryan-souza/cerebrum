import os
import logging
from typing_extensions import Self

# Set logging to error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)



import json
import numpy as np

from typing import Tuple, Union, Any
from pathlib import Path
from tensorflow.nn import softmax
from tensorflow import expand_dims
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array


# Constants
CEREBRUM_PATH = Path(__file__).resolve().parent
MODELS_PATH   = Path(CEREBRUM_PATH, 'models')
CONFIG_FILE   = Path(CEREBRUM_PATH, 'config.json')


class Identifier(object):
    _instances = {}

    def __new__(cls: type[Self]) -> Self:
        if cls not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[cls] = instance
        return cls._instances[cls]

    def __init__(self) -> None:
        config = json.load( open(CONFIG_FILE) )

        self.DEFAULT_IMG_SIZE = ( config['img_width'], config['img_height'] )
        self.IMG_SIZE         = self.DEFAULT_IMG_SIZE
        self.SECTION_NAMES    = config['section_names']
        self.PLANT_NAMES      = config['plant_names']
        self.MODELS           = dict([ model for model in self._load_models() ])

    @staticmethod
    # TODO: Specify return type annotation
    def _load_models():
        for model in MODELS_PATH.iterdir():
            class_name = str(model.stem)
            yield ( class_name, load_model(model) )


    def _identify_section(self, path: str) -> Tuple[str, np.array]:
        model = self.MODELS['section']
        img = load_img( path, target_size=self.IMG_SIZE )
        img_array = img_to_array(img)
        img_array = expand_dims(img_array, 0)

        predictions = model.predict(img_array)
        score = softmax( predictions[0] )
        class_name = self.SECTION_NAMES[ np.argmax(score) ]

        return ( class_name, img_array )


    def identify_plant(self, path: Union[str, Path]) -> Tuple[str, float]:
        class_name, img_array = self._identify_section(path)
        model = self.MODELS[ class_name ]

        predictions = model.predict(img_array)
        score = softmax( predictions[0] )
        plant_name = self.PLANT_NAMES[ np.argmax(score) ]
        accuracy = 100 * np.max(score)

        return ( plant_name, accuracy )
