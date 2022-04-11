from pathlib import Path

import pytest
import numpy as np
from tensorflow.python.framework.ops import EagerTensor

from cerebrum.main import Identifier
from cerebrum.main import CEREBRUM_PATH


identifier = Identifier()
TEST_PHOTOS_DIR = Path(CEREBRUM_PATH, 'fotos', 'testes')

@pytest.fixture
def plant_data():
    return {
        'path'    : Path(TEST_PHOTOS_DIR, 'Caule10LL.jpg'),
        'section' : 'caule',
        'plant'   : 'leucaena_leucocephala'
    }

@pytest.fixture
def all_plants():
    return {
        'Caule10LL': {
            'path'    : Path(TEST_PHOTOS_DIR, 'Caule10LL.jpg'),
            'section' : 'caule',
            'plant'   : 'leucaena_leucocephala'
        },
        'FlorCR20' : {
            'path'    : Path(TEST_PHOTOS_DIR, 'FlorCR20.jpg'),
            'section' : 'flor',
            'plant'   : 'crotalaria_retusa'
        },
        'FolhaCP8' : {
            'path'    : Path(TEST_PHOTOS_DIR, 'FolhaCP8.jpg'),
            'section' : 'folha',
            'plant'   : 'caesalpinia_pulcherrima'
        },
        'FrutoHC1' : {
            'path'    : Path(TEST_PHOTOS_DIR, 'FrutoHC1.jpg'),
            'section' : 'fruto',
            'plant'   : 'hymenaea_coubaril'
        },
        'FrutoHV18' : {
            'path'    : Path(TEST_PHOTOS_DIR, 'FrutoHV18.jpg'),
            'section' : 'fruto',
            'plant'   : 'hymenaea_velutina'
        }
    }

class TestIdentifier:
    def test_singleton(self):
        instance_one = Identifier()
        instance_two = Identifier()

        assert isinstance(instance_one, Identifier)
        assert isinstance(instance_two, Identifier)
        assert instance_one == instance_two

    def test_section_identification(self, plant_data):
        section, img_array = identifier._identify_section( plant_data['path'] )

        assert type(section) == str
        assert type(img_array) == EagerTensor
        assert section == plant_data['section']

    def test_plant_identification(self, plant_data):
        plant, accuracy = identifier.identify_plant( plant_data['path'] )

        assert type(plant) == str
        assert type(accuracy) == np.float64
        assert plant == plant_data['plant']

    def test_ai_accuracy(self, all_plants):
        for metadata in all_plants.values():
            section, _ = identifier._identify_section( metadata['path'] )
            plant, accuracy = identifier.identify_plant( metadata['path'] )

            assert section == metadata['section']
            assert plant == metadata['plant']
            assert accuracy >= 75
