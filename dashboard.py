import json
from pathlib import Path

import requests
import streamlit as st


st.set_page_config(page_title="Credit Approval Simulator", page_icon="🏦", layout="wide")
st.title("Explainable Credit Approval Simulator")
st.warning("Educational prototype only. This must not be used for real lending decisions.")
api_url = st.sidebar.text_input("API URL", "http://api:8000")
st.sidebar.caption("Model features intentionally omit demographic attributes; fairness fields are audit-only.")

defaults = {"LIMIT_BAL": 50000, "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0, "BILL_AMT1": 20000, "BILL_AMT2": 19000, "BILL_AMT3": 18000, "BILL_AMT4": 17000, "BILL_AMT5": 16000, "BILL_AMT6": 15000, "PAY_AMT1": 2000, "PAY_AMT2": 2000, "PAY_AMT3": 2000, "PAY_AMT4": 2000, "PAY_AMT5": 2000, "PAY_AMT6": 2000}
with st.form("applicant"):
    columns = st.columns(3)
    values = {}
    for index, (name, default) in enumerate(defaults.items()):
        with columns[index % 3]:
            values[name] = st.number_input(name, value=float(default))
    submitted = st.form_submit_button("Score applicant")
if submitted:
    try:
        response = requests.post(f"{api_url}/predict", json=values, timeout=30)
        response.raise_for_status()
        result = response.json()
        score_col, risk_col = st.columns(2)
        score_col.metric("Credit score", result["credit_score"])
        risk_col.metric("Modeled default risk", f"{result['risk_probability']:.1%}")
        st.caption(f"Risk band: {result['risk_band']} | Model version: {result['model_version']}")
        st.subheader(result["decision"].replace("_", " ").title())
        st.info(result["rationale"])
        for warning in result.get("warnings", []):
            st.warning(warning)
        st.subheader("Reason codes")
        for reason in result["reason_codes"]:
            st.write(f"• {reason}")
        with st.expander("Structured model explanations"):
            st.json(result.get("explanations", []))
        if result["decision"] == "manual_review" and st.button("Create review case"):
            review_response = requests.post(f"{api_url}/review-cases", json=values, timeout=30)
            if review_response.ok:
                st.success(f"Review case created: {review_response.json()['case_id']}")
            else:
                st.error(review_response.text)
    except requests.RequestException as exc:
        st.error(f"Could not reach the scoring API: {exc}")

st.divider()
st.subheader("Model evidence")
metrics_path = Path("outputs/metrics.json")
if metrics_path.exists():
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    st.json(report)
else:
    st.info("Run training to populate validation, test, and fairness reports.")

