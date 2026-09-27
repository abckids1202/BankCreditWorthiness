from types import SimpleNamespace

import numpy as np

from credit_simulator.explain import _feature_effect_vector


def test_feature_effect_vector_unwraps_calibration_estimators():
    wrapper = SimpleNamespace(calibrated_classifiers_=[SimpleNamespace(estimator=SimpleNamespace(coef_=np.array([[1.0, -2.0]]))), SimpleNamespace(estimator=SimpleNamespace(coef_=np.array([[3.0, -4.0]])))])
    assert np.allclose(_feature_effect_vector(wrapper, 2), [2.0, -3.0])
