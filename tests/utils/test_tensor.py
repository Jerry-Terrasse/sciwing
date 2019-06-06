import pytest
import torch
from parsect.utils.tensor import has_tensor


@pytest.fixture
def setup_has_tensor():
    objects = {
        "tensor": torch.rand(2, 5),
        "dict_1": {
            "tensor": torch.randn(2, 5),
            "some_other_thing": "hey there",
            "list_a": "This is a list".split(),
        },
        "dict_2": {"one_thing": 1, "other_thing": "a"},
    }

    return objects


class TestTensorUtils:
    def test_has_tensor_with_tensor_returns_true(self, setup_has_tensor):
        assert has_tensor(setup_has_tensor["tensor"])

    def test_has_tensor_with_dict_returns_true(self, setup_has_tensor):
        assert has_tensor(setup_has_tensor["dict_1"])

    def test_has_tensor_returns_false(self, setup_has_tensor):
        dict_2 = setup_has_tensor["dict_2"]
        assert has_tensor(dict_2) is False