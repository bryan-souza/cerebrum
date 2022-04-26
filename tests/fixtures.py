import random
import re
from pathlib import Path

import pytest

from cerebrum.main import CEREBRUM_PATH


TEST_PHOTOS_DIR = Path(CEREBRUM_PATH, 'fotos', 'testes')

NAMES_LOOKUP_TABLE = {
    'CP': 'caesalpinia_pulcherrima',
    'CD': 'chamaecrista_desvauxii',
    'CR': 'crotalaria_retusa',
    'HC': 'hymenaea_coubaril',
    'HV': 'hymenaea_velutina',
    'LL': 'leucaena_leucocephala'
}

SECTION_LOOKUP_TABLE = { # Only use with RegEx
    'Caule': 'caule',
    'Flor': 'flor',
    'Folha': 'folha',
    'Fruto': 'fruto'
}


def solve_plant_name( filename: str ):
    for key, value in NAMES_LOOKUP_TABLE.items():
        if key in filename:
            return value
    
    return ''

def solve_plant_section( filename: str ):
    for key, value in SECTION_LOOKUP_TABLE.items():
        if re.search(key, filename, re.IGNORECASE) is not None:
            return value
    
    return ''


@pytest.fixture
def plant_data():
    sections_path = Path(TEST_PHOTOS_DIR, 'orgaos')
    images = [ img for img in sections_path.glob('*.jpg') ]

    with random.choice(images) as image:
        key = image.stem
        return {
            'path':    image,
            'section': solve_plant_section(key),
            'plant':   solve_plant_name(key)
        }


@pytest.fixture
def all_caules():
    caules_path = Path(TEST_PHOTOS_DIR, 'caule')

    out = {}
    for image in caules_path.glob('*.jpg'):
        key = image.stem
        value = {
            'path':    image,
            'section': solve_plant_section(key),
            'plant':   solve_plant_name(key)
        }

        out.update({ key: value })

    return out

@pytest.fixture
def all_flores():
    flores_path = Path(TEST_PHOTOS_DIR, 'flor')

    out = {}
    for image in flores_path.glob('*.jpg'):
        key = image.stem
        value = {
            'path':    image,
            'section': solve_plant_section(key),
            'plant':   solve_plant_name(key)
        }

        out.update({ key: value })

    return out

@pytest.fixture
def all_folhas():
    folhas_path = Path(TEST_PHOTOS_DIR, 'folha')

    out = {}
    for image in folhas_path.glob('*.jpg'):
        key = image.stem
        value = {
            'path':    image,
            'section': solve_plant_section(key),
            'plant':   solve_plant_name(key)
        }

        out.update({ key: value })

    return out

@pytest.fixture
def all_frutos():
    frutos_path = Path(TEST_PHOTOS_DIR, 'fruto')

    out = {}
    for image in frutos_path.glob('*.jpg'):
        key = image.stem
        value = {
            'path':    image,
            'section': solve_plant_section(key),
            'plant':   solve_plant_name(key)
        }

        out.update({ key: value })

    return out