from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from credit_simulator.explain import _feature_effect_vector, reason_codes, structured_reasons


def test_feature_effect_vector_unwraps_calibration_estimators():
    wrapper = SimpleNamespace(calibrated_classifiers_=[SimpleNamespace(estimator=SimpleNamespace(coef_=np.array([[1.0, -2.0]]))), SimpleNamespace(estimator=SimpleNamespace(coef_=np.array([[3.0, -4.0]])))])
    assert np.allclose(_feature_effect_vector(wrapper, 2), [2.0, -3.0])


def test_tree_reasons_are_deterministic_local_effects_and_exclude_protected_fields():
    training = pd.DataFrame({"utilization": [0.05, 0.1, 0.8, 0.9, 0.2, 0.7], "late_payment_count": [0, 0, 3, 4, 1, 2]})
    target = np.array([0, 0, 1, 1, 0, 1])
    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingClassifier(max_iter=30, random_state=7))])
    model.fit(training, target)
    applicant = pd.DataFrame({"utilization": [0.85], "late_payment_count": [3]})
    first = structured_reasons(model, applicant, list(training.columns), {"utilization": "Utilization", "late_payment_count": "Late payments"}, top_n=2)
    second = structured_reasons(model, applicant, list(training.columns), {"utilization": "Utilization", "late_payment_count": "Late payments"}, top_n=2)
    assert first == second
    assert len(first) == 2
    assert all(set(reason) >= {"feature", "description", "value", "direction", "importance"} for reason in first)
    assert all(reason["feature"] in applicant.columns for reason in first)
    assert all(reason["direction"] in {"increased_risk", "reduced_risk"} for reason in first)
    assert all(reason["explanation_method"] == "local_feature_ablation" for reason in first)
    assert all("SEX" not in reason["feature"] for reason in first)
    assert len(reason_codes(model, applicant, list(training.columns))) == 2


def test_explanations_reject_protected_features():
    training = pd.DataFrame({"utilization": [0.05, 0.1, 0.8, 0.9], "late_payment_count": [0, 0, 3, 4]})
    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingClassifier(max_iter=10, random_state=7))]).fit(training, [0, 0, 1, 1])
    applicant = training.iloc[[0]].copy()
    with pytest.raises(ValueError, match="Protected attributes"):
        structured_reasons(model, applicant, list(training.columns), protected_features=["utilization"])
