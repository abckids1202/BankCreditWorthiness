from __future__ import annotations

import json
import os
from pathlib import Path

import requests
import streamlit as st
import pandas as pd


st.set_page_config(page_title="Credit Risk Simulator", page_icon="🏦", layout="wide")
st.title("Explainable Credit Risk Simulator")
st.warning("Educational prototype only. This must not be used for real lending decisions.")
api_url = st.sidebar.text_input("API URL", "http://api:8000").rstrip("/")
st.sidebar.caption("Demographic attributes are audit-only and are excluded from scoring.")


def api_request(method: str, path: str, **kwargs):
    headers = dict(kwargs.pop("headers", {}) or {})
    api_key = os.getenv("CREDIT_API_KEY")
    if api_key:
        headers.setdefault("X-API-Key", api_key)
    if headers:
        kwargs["headers"] = headers
    response = requests.request(method, f"{api_url}{path}", timeout=30, **kwargs)
    response.raise_for_status()
    return response.json()


def show_request_error(error):
    if isinstance(error, requests.HTTPError) and error.response is not None:
        st.error(f"API error {error.response.status_code}: {error.response.text}")
    else:
        st.error(f"Could not reach the scoring API: {error}")


defaults = {"LIMIT_BAL": 50000, "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0, "BILL_AMT1": 20000, "BILL_AMT2": 19000, "BILL_AMT3": 18000, "BILL_AMT4": 17000, "BILL_AMT5": 16000, "BILL_AMT6": 15000, "PAY_AMT1": 2000, "PAY_AMT2": 2000, "PAY_AMT3": 2000, "PAY_AMT4": 2000, "PAY_AMT5": 2000, "PAY_AMT6": 2000}
tabs = st.tabs(["Applicant scoring", "Threshold simulator", "Model evidence", "Review queue", "Drift monitoring"])

with tabs[0]:
    st.subheader("Score an educational sample applicant")
    with st.form("applicant"):
        columns = st.columns(3); values = {}
        for index, (name, default) in enumerate(defaults.items()):
            with columns[index % 3]:
                values[name] = st.number_input(name, value=float(default))
        submitted = st.form_submit_button("Score applicant")
    if submitted:
        try:
            st.session_state["applicant"] = values
            st.session_state["prediction"] = api_request("POST", "/predict", json=values)
        except requests.RequestException as exc:
            show_request_error(exc)
    result = st.session_state.get("prediction")
    if result:
        score_col, risk_col, decision_col = st.columns(3)
        score_col.metric("Credit score", result["credit_score"])
        risk_col.metric("Default risk", f"{result.get('default_probability', result['risk_probability']):.1%}")
        decision_col.metric("Recommendation", result["decision"].replace("_", " ").title())
        st.caption(f"Risk band: {result['risk_band']} | Model: {result['model_version']} | Policy: {result.get('policy_version', 'unknown')}")
        st.info(result["rationale"])
        for warning in result.get("warnings", []):
            st.warning(warning)
        st.subheader("Reason codes")
        for reason in result["reason_codes"]:
            st.write(f"• {reason}")
        with st.expander("Structured model explanations"):
            st.json(result.get("explanations", []))
        if result and result.get("decision") != "manual_review":
            st.caption("Only manual_review recommendations can enter the human-review queue.")
        if st.button("Create review case", disabled=not result or result.get("decision") != "manual_review"):
            try:
                case = api_request("POST", "/review-cases", json=st.session_state["applicant"])
                st.success(f"Review case created: {case['case_id']}")
            except requests.RequestException as exc:
                show_request_error(exc)

with tabs[1]:
    st.subheader("Research-only policy simulator")
    st.caption("This changes only the simulation output; it never changes the configured automatic policy.")
    try:
        active_policy = api_request("GET", "/policy/current")
        active_thresholds = active_policy["decision_thresholds"]
        st.caption(f"Active policy: {active_policy['policy_version']} | approve ≤ {active_thresholds['approve_max_risk']:.2f} | decline ≥ {active_thresholds['decline_min_risk']:.2f}")
    except requests.RequestException:
        active_thresholds = {"approve_max_risk": 0.20, "decline_min_risk": 0.45}
        st.warning("Could not load the active policy; simulator defaults are shown.")
    probabilities_text = st.text_area("Predicted probabilities, comma-separated", "0.05, 0.12, 0.28, 0.51, 0.74")
    defaults_text = st.text_input("Optional actual defaults, comma-separated 0/1", "")
    approve = st.slider("Approve at or below", 0.0, 1.0, float(active_thresholds["approve_max_risk"]), 0.01)
    decline = st.slider("Decline at or above", 0.0, 1.0, float(active_thresholds["decline_min_risk"]), 0.01)
    if st.button("Simulate thresholds"):
        try:
            probabilities = [float(value.strip()) for value in probabilities_text.split(",") if value.strip()]
            actual = [int(value.strip()) for value in defaults_text.split(",") if value.strip()] if defaults_text.strip() else None
            st.json(api_request("POST", "/policy/simulate", json={"probabilities": probabilities, "actual_defaults": actual, "approve_max_risk": approve, "decline_min_risk": decline}))
        except (ValueError, requests.RequestException) as exc:
            show_request_error(exc)
with tabs[2]:
    st.subheader("Model evidence")
    report_path = Path("outputs/reports/training_report.json")
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        metric_tab, visual_tab, fairness_tab, calibration_tab, registry_tab = st.tabs(["Metrics", "Visual diagnostics", "Fairness", "Calibration", "Registry"])
        with metric_tab:
            st.json({"selected_model": report.get("selected_model"), "candidate_metrics": report.get("candidate_metrics"), "test_metrics": report.get("test_metrics")})
            if report.get("test_metric_bootstrap"):
                st.subheader("Held-out metric uncertainty")
                st.caption("Bootstrap 95% intervals show sampling uncertainty in the test split; they are not a guarantee of future performance.")
                st.json(report["test_metric_bootstrap"])
        with visual_tab:
            st.caption("These diagnostics describe the held-out test split and the configured policy. They are evidence for learning and review, not lending decisions.")
            image_columns = st.columns(2)
            for column, filename, caption in [
                (image_columns[0], "feature_distributions.png", "Feature distributions"),
                (image_columns[1], "confusion_matrix.png", "Confusion matrix"),
                (image_columns[0], "threshold_comparison.png", "Decision population by threshold"),
                (image_columns[1], "risk_distribution.png", "Predicted-risk distributions"),
            ]:
                image = Path("outputs/reports") / filename
                if image.exists():
                    column.image(str(image), caption=caption, use_container_width=True)
            threshold_path = Path("outputs/reports/threshold_analysis.csv")
            if threshold_path.exists():
                st.subheader("Threshold comparison data")
                st.dataframe(pd.read_csv(threshold_path), use_container_width=True, hide_index=True)
            approval_path = Path("outputs/reports/approval_rate_analysis.csv")
            if approval_path.exists():
                st.subheader("Performance at target approval rates")
                st.caption("Default recall here means the fraction of observed defaults that were not approved in the held-out test split.")
                st.dataframe(pd.read_csv(approval_path), use_container_width=True, hide_index=True)
            importance = report.get("global_feature_importance", [])
            if importance:
                st.subheader("Global feature importance")
                st.caption("Permutation importance measures the change in held-out ROC-AUC when a feature is shuffled. It describes model behavior, not causality.")
                st.dataframe(pd.DataFrame(importance), use_container_width=True, hide_index=True)
        with fairness_tab:
            st.json(report.get("fairness_detailed", report.get("fairness", {})))
            if report.get("fairness_threshold_sensitivity"):
                st.subheader("Fairness threshold sensitivity")
                st.caption("The same group metrics are recomputed at several policy bands; disparities can change when thresholds change.")
                st.json(report["fairness_threshold_sensitivity"])
        with calibration_tab:
            st.json(report.get("calibration", {}))
            image = Path("outputs/reports/calibration_curve.png")
            if image.exists():
                st.image(str(image), caption="Calibration curve")
        with registry_tab:
            st.caption("Read-only model lifecycle view. Promotion and rollback remain explicit CLI operations.")
            try:
                registry = api_request("GET", "/models").get("models", [])
                if registry:
                    st.dataframe(pd.DataFrame(registry), use_container_width=True, hide_index=True)
                else:
                    st.info("No registered model versions found.")
            except requests.RequestException as exc:
                show_request_error(exc)
        with st.expander("Data quality and feature engineering"):
            st.json({"training_config": report.get("training_config"), "data_quality": report.get("data_quality"), "feature_engineering": report.get("feature_engineering")})
    else:
        st.info("Run python scripts/train.py to generate model evidence.")

with tabs[3]:
    st.subheader("Human review queue")
    try:
        cases = api_request("GET", "/review-cases")
        if cases:
            st.dataframe([{key: case.get(key) for key in ("case_id", "created_at", "automatic_decision", "reviewer_decision", "risk_probability", "credit_score")} for case in cases], use_container_width=True)
            case_ids = [case["case_id"] for case in cases]
            selected_id = st.selectbox("Case", case_ids)
            selected = api_request("GET", f"/review-cases/{selected_id}")
            st.json(selected)
            history = api_request("GET", f"/review-cases/{selected_id}/history")
            with st.expander("Review audit history"):
                st.dataframe(history["events"], use_container_width=True)
            with st.form("review_update"):
                decision = st.selectbox("Reviewer decision", ["approved", "declined", "needs_more_information", "escalated"])
                reviewer_id = st.text_input("Reviewer ID", placeholder="e.g. reviewer-demo")
                note = st.text_area("Reviewer note")
                if st.form_submit_button("Save review"):
                    updated = api_request("PATCH", f"/review-cases/{selected_id}", json={"reviewer_decision": decision, "reviewer_note": note, "reviewer_id": reviewer_id or None})
                    st.success(f"Saved {updated['reviewer_decision']}")
        else:
            st.info("No review cases yet.")
    except requests.RequestException as exc:
        show_request_error(exc)

with tabs[4]:
    st.subheader("Input drift monitoring")
    st.caption("Provide reference and current records as JSON arrays. Critical drift should be investigated before automated use.")
    reference_text = st.text_area("Reference records", '[{"LIMIT_BAL": 10000}, {"LIMIT_BAL": 20000}]')
    current_text = st.text_area("Current records", '[{"LIMIT_BAL": 12000}, {"LIMIT_BAL": 25000}]')
    if st.button("Check drift"):
        try:
            result = api_request("POST", "/monitoring/drift", json={"reference_records": json.loads(reference_text), "current_records": json.loads(current_text)})
            st.json(result)
            if result["critical_features"]:
                st.error("Critical drift detected. Investigate before automated use.")
            elif result["warning_features"]:
                st.warning("Warning-level drift detected.")
            else:
                st.success("No material drift detected by the configured PSI thresholds.")
        except (ValueError, requests.RequestException) as exc:
            show_request_error(exc)
    try:
        history = api_request("GET", "/monitoring/drift/history")
        if history:
            st.subheader("Recent drift diagnostics")
            st.dataframe(history, use_container_width=True)
    except requests.RequestException as exc:
        show_request_error(exc)
