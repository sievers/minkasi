import numpy as np

from minkasi.fitting import models


class DummyTOD:
    def __init__(self):
        self.info = {
            "dx": np.zeros((2, 3)),
            "dy": np.zeros((2, 3)),
        }

    def get_data_dims(self):
        return (2, 3)

    def get_empty(self, zero=False):
        if zero:
            return np.zeros((2, 3))
        return np.empty((2, 3))


def test_derivs_from_isobeta_uses_numpy_prod(monkeypatch):
    def deprecated_product(*args, **kwargs):
        raise AssertionError("np.product must not be used")

    monkeypatch.setattr(np, "product", deprecated_product, raising=False)
    calls = []

    def fill_derivs(params, dx, dy, pred, derivs, size):
        calls.append(size)

    monkeypatch.setattr(models, "fill_isobeta_derivs_c", fill_derivs)

    params = np.array([0.0, 0.0, 1.0, 0.5, 2.0])
    derivs, pred = models.derivs_from_isobeta_c(params, DummyTOD())

    assert calls == [6]
    assert derivs.shape == (5, 2, 3)
    assert pred.shape == (2, 3)
