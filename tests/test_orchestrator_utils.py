import torch

from ralo.orchestrator import encode_gradients, extract_gradient_tensors
from ralo.utils import json_to_bytes_list, bytes_list_to_json


def test_gradient_roundtrip_serialization():
    grads = {
        "layer.weight": torch.ones(2, 2),
        "layer.bias": torch.zeros(2),
    }
    payload = {"meta": "test"}
    payload.update(encode_gradients(grads))

    blob = json_to_bytes_list(payload)
    decoded = bytes_list_to_json(blob)
    decoded_grads = extract_gradient_tensors(decoded)

    assert set(decoded_grads.keys()) == set(grads.keys())
    for name, tensor in grads.items():
        assert torch.allclose(decoded_grads[name], tensor)

