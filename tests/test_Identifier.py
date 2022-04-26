import random
import re

import numpy as np
from tensorflow.python.framework.ops import EagerTensor

from cerebrum.main import Identifier
from .fixtures import plant_data, all_caules, all_flores, all_folhas, all_frutos

identifier = Identifier()


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


class TestCaule:

    def test_caules_identification(self, all_caules):
        for value in all_caules.values():
            plant_name, accuracy = identifier.identify_plant( value['path'] )

            assert type(plant_name) == str
            assert type(accuracy) == np.float64
            assert plant_name == value['plant']


class TestFlor:

    def test_flores_identification(self, all_flores):
        for value in all_flores.values():
            plant_name, accuracy = identifier.identify_plant( value['path'] )

            assert type(plant_name) == str
            assert type(accuracy) == np.float64
            assert plant_name == value['plant']

class TestFolha:

    def test_flores_identification(self, all_folhas):
        for value in all_folhas.values():
            plant_name, accuracy = identifier.identify_plant( value['path'] )

            assert type(plant_name) == str
            assert type(accuracy) == np.float64
            assert plant_name == value['plant']


class TestFruto:

    def test_flores_identification(self, all_frutos):
        for value in all_frutos.values():
            plant_name, accuracy = identifier.identify_plant( value['path'] )

            assert type(plant_name) == str
            assert type(accuracy) == np.float64
            assert plant_name == value['plant']