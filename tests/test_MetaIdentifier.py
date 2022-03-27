from cerebrum.main import MetaIdentifier

# Sample class
class Sample(metaclass=MetaIdentifier):
    def __init__(self) -> None:
        self.sample_id = "1337"


class TestMetaIdentifier:
    def test_singleton(self):
        instance_one = Sample()
        instance_two = Sample()

        assert instance_one == instance_two