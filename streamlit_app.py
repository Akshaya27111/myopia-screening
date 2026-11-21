"""
OcuScreen AI - Pediatric & Adolescent Myopia Screening System
Trained and validated on CDC NHANES 1999-2008 (Ages 12-25, n=2,491 held-out test cohort).
Predicts myopia without cycloplegia or refraction equipment.
"""

import os
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------
# Page Configuration & Custom CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="OcuScreen AI | NHANES Myopia Screening",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(135deg, #0d3b66 0%, #005f73 50%, #0a9396 100%);
        color: white;
        padding: 2rem 2.2rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.12);
    }
    .main-header h1 {
        color: white;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #e0f2f1;
        font-size: 1.05rem;
        margin-top: 0.4rem;
        margin-bottom: 0.8rem;
    }
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin-top: 0.6rem;
    }
    .clinical-badge {
        background-color: rgba(255, 255, 255, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.35);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 500;
    }
    .result-card-danger {
        background-color: #fff5f5;
        border-left: 6px solid #e53e3e;
        padding: 1.25rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(229,62,62,0.1);
    }
    .result-card-success {
        background-color: #f0fff4;
        border-left: 6px solid #38a169;
        padding: 1.25rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(56,161,105,0.1);
    }
    .result-card-warning {
        background-color: #fffaf0;
        border-left: 6px solid #dd6b20;
        padding: 1.25rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(221,107,32,0.1);
    }
    .metric-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
    }
    .metric-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 0.5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Snellen & logMAR Conversion Constants
# ---------------------------------------------------------
SNELLEN_OPTIONS = [20, 25, 30, 40, 50, 60, 80, 200, 400]
SNELLEN_LABELS = {
    20: "20/20 (Normal / Standard vision)",
    25: "20/25 (Mild blur)",
    30: "20/30 (Slightly reduced)",
    40: "20/40 (US driving cutoff)",
    50: "20/50 (Noticeable blur)",
    60: "20/60 (Moderate impairment)",
    80: "20/80 (Substantial impairment)",
    200: "20/200 (Legal blindness threshold)",
    400: "20/400 (Severe distance impairment)",
}

# ---------------------------------------------------------
# Model Loader with Caching
# ---------------------------------------------------------
@st.cache_resource
def load_screening_model():
    """Loads the trained screening bundle."""
    possible_paths = [
        "myopia_screening_model.joblib",
        os.path.join(os.path.dirname(__file__), "myopia_screening_model.joblib"),
        os.path.join(os.path.dirname(__file__), "myopia_project_final", "myopia_screening_model.joblib"),
        r"C:\Users\rahul k\.gemini\antigravity\scratch\myopia_project\myopia_screening_model.joblib"
    ]
    bundle_path = None
    for p in possible_paths:
        if os.path.exists(p):
            bundle_path = p
            break

    if not bundle_path:
        st.error("Model file `myopia_screening_model.joblib` not found. Please ensure it is in the application directory.")
        return None
    try:
        bundle = joblib.load(bundle_path)
        return bundle
    except Exception as e:
        st.error(f"Error loading model bundle: {e}")
        return None

@st.cache_data
def load_benchmark_results():
    """Loads benchmark results from myopia_results_v4.csv."""
    possible_paths = [
        "myopia_results_v4.csv",
        os.path.join(os.path.dirname(__file__), "myopia_results_v4.csv"),
        os.path.join(os.path.dirname(__file__), "myopia_project_final", "myopia_results_v4.csv"),
        r"C:\Users\rahul k\.gemini\antigravity\scratch\myopia_project\myopia_results_v4.csv"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            try:
                return pd.read_csv(p)
            except Exception:
                pass
    return None

# Load model bundle
bundle = load_screening_model()
results_df = load_benchmark_results()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/visible.png", width=64)
    st.title("OcuScreen AI")
    st.markdown("**Evidence-Based Adolescent Myopia Screening**")
    st.caption("Developed using CDC NHANES 1999–2008 epidemiological survey data for youths aged 12–25.")

    st.markdown("---")
    st.markdown("### ⚙️ Model Specifications")
    if bundle:
        st.markdown(f"- **Classifier:** `{bundle.get('model', 'RandomForest')}`")
        st.markdown(f"- **Feature Set:** `Set D (+ Glasses Q)`")
        st.markdown(f"- **Operating Cutoff:** `{bundle.get('threshold', 0.425):.3f}`")
        st.markdown("- **Cohort:** 2,491 held-out test subjects")
        st.markdown("- **ROC-AUC:** `0.873 (0.859–0.887)`")
        st.markdown("- **Sensitivity:** `82.4%` | **Specificity:** `79.0%`")
    else:
        st.warning("Model bundle offline.")

    st.markdown("---")
    st.markdown("### ⚠️ Clinical Advisory")
    st.info(
        "This tool is an evidence-based clinical screening aid designed for non-cycloplegic risk stratification. "
        "It does not replace comprehensive cycloplegic retinoscopy or slit lamp examination."
    )

# ---------------------------------------------------------
# Header Banner
# ---------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>👁️ OcuScreen AI: Pediatric & Adolescent Myopia Screener</h1>
        <p>Rapid, non-cycloplegic refractive risk assessment using presenting visual acuity, glasses history, and biometrics.</p>
        <div class="badge-container">
            <span class="clinical-badge">🔬 CDC NHANES 1999–2008 Cohort</span>
            <span class="clinical-badge">🎯 Ages 12–25 Validated</span>
            <span class="clinical-badge">📈 Held-Out Test ROC-AUC: 0.873</span>
            <span class="clinical-badge">⚡ Zero Autorefractor / Cycloplegia Needed</span>
            <span class="clinical-badge">🛡️ Strict Leakage Prevention</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Main App Tabs
# ---------------------------------------------------------
tab_screener, tab_batch, tab_benchmarks, tab_protocol, tab_deploy = st.tabs(
    [
        "🩺 Patient Screener",
        "📁 Batch Cohort Screener",
        "📊 Clinical Evidence & Benchmarks",
        "📋 Clinical Screening Protocol",
        "🚀 Deployment & Guide",
    ]
)

# Helper function to compute features row
def compute_features(age, sex, glasses, va_r, va_l, race, k_r, k_l, cyl_r, cyl_l, bmi, height, pir):
    nan_helper = lambda v: np.nan if v is None or v == "" else float(v)
    lr = float(np.log10(va_r / 20.0))
    ll = float(np.log10(va_l / 20.0))
    dist = 1.0 if glasses == "Yes" else 0.0

    ks = [float(v) for v in (k_r, k_l) if v is not None and not np.isnan(v)]
    cyls = [float(v) for v in (cyl_r, cyl_l) if v is not None and not np.isnan(v)]

    k_mean = np.mean(ks) if ks else np.nan
    k_diff = abs(k_r - k_l) if len(ks) == 2 else np.nan
    cyl_mean = np.mean(cyls) if cyls else np.nan
    cyl_diff = abs(cyl_r - cyl_l) if len(cyls) == 2 else np.nan

    better_logmar = min(lr, ll)
    worse_logmar = max(lr, ll)
    diff_logmar = abs(lr - ll)
    worse_no_glasses = worse_logmar * (1.0 - dist)

    row = {
        "AGE": float(age),
        "K_MEAN": k_mean,
        "K_DIFF": k_diff,
        "CORNEAL_CYL_MEAN": cyl_mean,
        "CORNEAL_CYL_DIFF": cyl_diff,
        "SEX_FEMALE": 1.0 if sex == "Female" else 0.0,
        "PIR": nan_helper(pir),
        "BMI": nan_helper(bmi),
        "HEIGHT": nan_helper(height),
        "RACE": np.nan if race in ["Unknown", None, ""] else race,
        "VA_BETTER_LOGMAR": better_logmar,
        "VA_WORSE_LOGMAR": worse_logmar,
        "VA_DIFF_LOGMAR": diff_logmar,
        "DIST_GLASSES": dist,
        "VA_WORSE_NO_GLASSES": worse_no_glasses,
    }
    return row

# ---------------------------------------------------------
# TAB 1: Patient Screener (Single Evaluation)
# ---------------------------------------------------------
with tab_screener:
    st.subheader("Individual Patient Screening Evaluation")
    st.markdown(
        "Evaluate an individual patient aged 12–25. Preset clinical scenarios can be selected below to inspect typical model outputs."
    )

    # Preset profiles
    presets = {
        "Custom / Enter Manually": None,
        "Case 1: Adolescent with blurred distance vision (Age 15, No glasses, VA 20/80) -> High Risk": {
            "age": 15, "sex": "Female", "glasses": "No", "va_r": 80, "va_l": 80,
            "race": "MexicanAmerican", "k_r": 44.5, "k_l": 44.75, "cyl_r": 0.75, "cyl_l": 0.5,
            "bmi": 22.0, "height": 160.0, "pir": 2.1
        },
        "Case 2: Healthy Emmetropic Student (Age 17, No glasses, 20/20 vision) -> Low Risk": {
            "age": 17, "sex": "Male", "glasses": "No", "va_r": 20, "va_l": 20,
            "race": "NHWhite", "k_r": 43.0, "k_l": 43.2, "cyl_r": 0.25, "cyl_l": 0.25,
            "bmi": 21.5, "height": 175.0, "pir": 3.8
        },
        "Case 3: Distance Glasses / Contact Lens Wearer (Age 20, Wearing distance aids, VA 20/25) -> High Risk": {
            "age": 20, "sex": "Female", "glasses": "Yes", "va_r": 25, "va_l": 20,
            "race": "Other", "k_r": 45.2, "k_l": 45.0, "cyl_r": 1.25, "cyl_l": 1.0,
            "bmi": 20.4, "height": 163.0, "pir": 4.2
        },
        "Case 4: Borderline / Mild Visual Reduction (Age 13, No glasses, VA 20/30) -> Borderline": {
            "age": 13, "sex": "Male", "glasses": "No", "va_r": 30, "va_l": 30,
            "race": "NHBlack", "k_r": 43.8, "k_l": 43.6, "cyl_r": 0.5, "cyl_l": 0.75,
            "bmi": 19.8, "height": 155.0, "pir": 1.5
        },
        "Case 5: Severe Distance Blur / High Myopia Candidate (Age 21, No glasses, VA 20/200) -> Severe Risk": {
            "age": 21, "sex": "Male", "glasses": "No", "va_r": 200, "va_l": 200,
            "race": "Other", "k_r": 46.0, "k_l": 45.8, "cyl_r": 1.5, "cyl_l": 1.25,
            "bmi": 23.5, "height": 178.0, "pir": 3.2
        }
    }

    selected_preset = st.selectbox("📌 Quick Test: Load a Clinical Scenario Preset", list(presets.keys()))
    preset_data = presets[selected_preset]

    col_form, col_result = st.columns([1.1, 0.9], gap="large")

    with col_form:
        with st.form("patient_screener_form"):
            st.markdown("#### 1. Core Screening Information (Essential)")
            
            c1, c2 = st.columns(2)
            with c1:
                age_val = preset_data["age"] if preset_data else 16
                age = st.slider("Patient Age (years)", min_value=12, max_value=25, value=age_val, help="NHANES screening model is calibrated for youths aged 12–25.")
            with c2:
                sex_val = preset_data["sex"] if preset_data else "Female"
                sex_idx = 0 if sex_val == "Female" else 1
                sex = st.radio("Biological Sex", ["Female", "Male"], index=sex_idx, horizontal=True)

            st.markdown("---")
            glasses_val = preset_data["glasses"] if preset_data else "No"
            glasses_idx = 0 if glasses_val == "No" else 1
            glasses = st.radio(
                "👓 Does the patient wear glasses or contact lenses for distance vision?",
                ["No", "Yes"],
                index=glasses_idx,
                horizontal=True,
                help="Key discriminator in NHANES (VIQ070). Identifies known distance refractive errors without requiring prescription data."
            )

            st.markdown("---")
            st.markdown("##### 👁️ Presenting Visual Acuity (Snellen Chart 20/x)")
            st.caption("Measure visual acuity as presented (with current everyday distance correction if worn, or unaided if none).")

            c_va1, c_va2 = st.columns(2)
            with c_va1:
                va_r_val = preset_data["va_r"] if preset_data else 20
                va_r_idx = SNELLEN_OPTIONS.index(va_r_val) if va_r_val in SNELLEN_OPTIONS else 0
                va_r = st.selectbox(
                    "Right Eye (OD) Visual Acuity",
                    SNELLEN_OPTIONS,
                    index=va_r_idx,
                    format_func=lambda x: SNELLEN_LABELS.get(x, f"20/{x}")
                )
                st.caption(f"OD logMAR: **{np.log10(va_r / 20.0):.2f}**")

            with c_va2:
                va_l_val = preset_data["va_l"] if preset_data else 20
                va_l_idx = SNELLEN_OPTIONS.index(va_l_val) if va_l_val in SNELLEN_OPTIONS else 0
                va_l = st.selectbox(
                    "Left Eye (OS) Visual Acuity",
                    SNELLEN_OPTIONS,
                    index=va_l_idx,
                    format_func=lambda x: SNELLEN_LABELS.get(x, f"20/{x}")
                )
                st.caption(f"OS logMAR: **{np.log10(va_l / 20.0):.2f}**")

            st.markdown("---")
            with st.expander("🔬 Optional Biometrics & Demographics (Imputed with Medians if Blank)"):
                st.caption("These biometric inputs refine the prediction but are not mandatory for preliminary screening.")
                
                race_options = ["Unknown", "MexicanAmerican", "OtherHispanic", "NHWhite", "NHBlack", "Other"]
                race_val = preset_data["race"] if preset_data else "Unknown"
                race_idx = race_options.index(race_val) if race_val in race_options else 0
                race = st.selectbox("Race / Ethnicity", race_options, index=race_idx)

                ck1, ck2 = st.columns(2)
                with ck1:
                    kr_val = preset_data["k_r"] if preset_data else None
                    k_r = st.number_input("Right Corneal Power K (Diopters)", value=kr_val, min_value=35.0, max_value=60.0, step=0.25, help="Normal human range: 41.0 to 46.0 D.")
                    cylr_val = preset_data["cyl_r"] if preset_data else None
                    cyl_r = st.number_input("Right Corneal Astigmatism (D)", value=cylr_val, min_value=0.0, max_value=10.0, step=0.25)

                with ck2:
                    kl_val = preset_data["k_l"] if preset_data else None
                    k_l = st.number_input("Left Corneal Power K (Diopters)", value=kl_val, min_value=35.0, max_value=60.0, step=0.25)
                    cyll_val = preset_data["cyl_l"] if preset_data else None
                    cyl_l = st.number_input("Left Corneal Astigmatism (D)", value=cyll_val, min_value=0.0, max_value=10.0, step=0.25)

                cb1, cb2, cb3 = st.columns(3)
                with cb1:
                    h_val = preset_data["height"] if preset_data else None
                    height = st.number_input("Height (cm)", value=h_val, min_value=100.0, max_value=220.0, step=0.5)
                with cb2:
                    bmi_val = preset_data["bmi"] if preset_data else None
                    bmi = st.number_input("BMI (kg/m²)", value=bmi_val, min_value=10.0, max_value=70.0, step=0.1)
                with cb3:
                    pir_val = preset_data["pir"] if preset_data else None
                    pir = st.number_input("Income-to-Poverty Ratio", value=pir_val, min_value=0.0, max_value=5.0, step=0.1, help="PIR 0.0 to 5.0 (higher = higher socioeconomic status).")

            submit_btn = st.form_submit_button("⚡ Compute Myopia Risk Assessment", use_container_width=True, type="primary")

    with col_result:
        st.markdown("#### 2. Clinical Risk Analysis")

        # Automatically calculate upon load or submit
        row = compute_features(age, sex, glasses, va_r, va_l, race, k_r, k_l, cyl_r, cyl_l, bmi, height, pir)
        
        if bundle is not None:
            X_eval = pd.DataFrame([row])[bundle["features"]]
            prob = float(bundle["pipeline"].predict_proba(X_eval)[:, 1][0])
            threshold = float(bundle.get("threshold", 0.42489))

            # Determine risk tier and recommendation
            is_referral = prob >= threshold
            if prob < 0.30:
                tier = "Low Risk"
                tier_color = "#38a169"
                status_icon = "✅"
                recommendation = "Low probability of significant myopia. Recommend standard annual vision screening and routine outdoor daylight hygiene."
                card_class = "result-card-success"
            elif prob < threshold:
                tier = "Borderline / Low-Moderate"
                tier_color = "#d69e2e"
                status_icon = "⚠️"
                recommendation = "Borderline screening score below the intervention cutoff. Re-evaluate visual acuity in 6 months or refer if symptomatic for distance blur or asthenopia."
                card_class = "result-card-warning"
            elif prob < 0.70:
                tier = "Elevated Risk"
                tier_color = "#dd6b20"
                status_icon = "🚨"
                recommendation = "Positive screening result exceeding the optimal decision cutoff (42.5%). Recommend formal cycloplegic refraction and comprehensive pediatric/adolescent eye exam."
                card_class = "result-card-danger"
            else:
                tier = "High Risk"
                tier_color = "#e53e3e"
                status_icon = "🚨"
                recommendation = "Strong likelihood of clinically significant myopia (spherical equivalent ≤ -0.5 D). Immediate referral for cycloplegic eye examination, axial length biometry, and myopia progression management."
                card_class = "result-card-danger"

            # Top KPI metrics
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-label">Estimated Risk</div>
                        <div class="metric-value" style="color: {tier_color};">{prob:.1%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m2:
                status_text = "REFER" if is_referral else "MONITOR"
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-label">Screening Decision</div>
                        <div class="metric-value" style="color: {tier_color};">{status_icon} {status_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m3:
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-label">Decision Threshold</div>
                        <div class="metric-value">{threshold:.1%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Interactive Plotly Gauge Meter
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prob * 100.0,
                    number={"suffix": "%", "font": {"size": 34, "color": tier_color}},
                    title={"text": f"<b>Myopia Screening Probability: {tier}</b>", "font": {"size": 16, "color": "#1e293b"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748b"},
                        "bar": {"color": tier_color, "thickness": 0.3},
                        "bgcolor": "white",
                        "borderwidth": 1,
                        "bordercolor": "#cbd5e1",
                        "steps": [
                            {"range": [0, 30], "color": "rgba(56, 161, 105, 0.2)"},
                            {"range": [30, threshold * 100], "color": "rgba(221, 107, 32, 0.15)"},
                            {"range": [threshold * 100, 70], "color": "rgba(221, 107, 32, 0.35)"},
                            {"range": [70, 100], "color": "rgba(229, 62, 62, 0.35)"},
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 4},
                            "thickness": 0.85,
                            "value": threshold * 100,
                        },
                    },
                )
            )
            fig_gauge.update_layout(
                height=260,
                margin=dict(l=25, r=25, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"family": "sans-serif"}
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Clinical action callout box
            st.markdown(
                f"""
                <div class="{card_class}">
                    <h4 style="margin: 0 0 0.5rem 0; color: #1e293b;">
                        {status_icon} Clinical Recommendation: <strong>{tier}</strong>
                    </h4>
                    <p style="margin: 0; color: #334155; font-size: 0.95rem;">{recommendation}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Key clinical drivers breakdown
            with st.expander("🔍 Risk Factor Breakdown for This Patient"):
                st.markdown("**Evaluated Input Parameters:**")
                breakdown_items = [
                    ("Distance Vision Aids", f"{glasses} (Wears distance glasses/contacts)" if glasses == "Yes" else "No glasses reported"),
                    ("Worse Eye Visual Acuity", f"20/{max(va_r, va_l)} (logMAR: {row['VA_WORSE_LOGMAR']:.2f})"),
                    ("Better Eye Visual Acuity", f"20/{min(va_r, va_l)} (logMAR: {row['VA_BETTER_LOGMAR']:.2f})"),
                    ("Inter-ocular Acuity Difference", f"{row['VA_DIFF_LOGMAR']:.2f} logMAR"),
                    ("Mean Corneal Curvature", f"{row['K_MEAN']:.2f} D" if not np.isnan(row['K_MEAN']) else "Imputed from cohort median (43.8 D)"),
                    ("Corneal Astigmatism", f"{row['CORNEAL_CYL_MEAN']:.2f} D" if not np.isnan(row['CORNEAL_CYL_MEAN']) else "Imputed from cohort median (0.6 D)"),
                ]
                for label, val in breakdown_items:
                    st.markdown(f"- **{label}:** {val}")

            # Printable / Copyable Clinical Receipt
            report_text = f"""--- OCUSCREEN AI CLINICAL SCREENING REPORT ---
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
Patient Demographics: Age {age}, Biological Sex: {sex}, Race: {race}
Presenting Visual Acuity: OD 20/{va_r} | OS 20/{va_l}
Distance Glasses Worn: {glasses}
Predicted Myopia Probability: {prob:.1%}
Screening Decision Cutoff: {threshold:.1%}
Outcome: {'REFERRAL RECOMMENDED' if is_referral else 'ROUTINE MONITORING'}
Clinical Tier: {tier}
Guidance: {recommendation}
Reference: CDC NHANES 1999-2008 Random Forest Model (ROC-AUC 0.873)
-----------------------------------------------"""

            st.download_button(
                label="📄 Download Clinical Screening Summary (TXT)",
                data=report_text,
                file_name=f"myopia_screening_report_{age}yo_{sex.lower()}.txt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.error("Model unavailable. Please verify model file.")

# ---------------------------------------------------------
# TAB 2: Batch Cohort Screener
# ---------------------------------------------------------
with tab_batch:
    st.subheader("📁 Batch Cohort Screener (School & Clinic Populations)")
    st.markdown(
        "Upload a roster or survey of patient records to rapidly screen an entire cohort without manual entry. "
        "Useful for school nurses, pediatric vision drives, and epidemiological research."
    )

    # Provide template download
    sample_file_path = "sample_patients.csv"
    if not os.path.exists(sample_file_path):
        sample_file_path = os.path.join(os.path.dirname(__file__), "sample_patients.csv")

    col_b1, col_b2 = st.columns([1, 1])
    with col_b1:
        st.markdown("**Step 1: Download Template**")
        if os.path.exists(sample_file_path):
            with open(sample_file_path, "rb") as f:
                st.download_button(
                    "📥 Download Sample Patient CSV Template (10 Patients)",
                    data=f.read(),
                    file_name="sample_patients_template.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.info("Sample template file will be generated on first run.")

    with col_b2:
        st.markdown("**Step 2: Upload CSV File**")
        uploaded_file = st.file_uploader("Upload CSV Roster", type=["csv"])

    if uploaded_file is not None and bundle is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(batch_df)} patient records.")

            # Required or mapped columns
            col_map = {
                "Age": "Age",
                "Sex": "Sex",
                "Distance_Glasses": "Distance_Glasses",
                "Right_Eye_VA": "Right_Eye_VA",
                "Left_Eye_VA": "Left_Eye_VA"
            }

            # Normalize column names
            batch_cols_lower = {c.lower(): c for c in batch_df.columns}
            
            # Process rows
            processed_rows = []
            predictions = []

            for idx, r in batch_df.iterrows():
                p_age = r.get("Age", r.get("age", 16))
                p_sex = r.get("Sex", r.get("sex", "Female"))
                p_glasses = r.get("Distance_Glasses", r.get("distance_glasses", "No"))
                p_va_r = r.get("Right_Eye_VA", r.get("right_eye_va", 20))
                p_va_l = r.get("Left_Eye_VA", r.get("left_eye_va", 20))
                p_race = r.get("Race", r.get("race", "Unknown"))
                p_kr = r.get("K_Right", r.get("k_right", np.nan))
                p_kl = r.get("K_Left", r.get("k_left", np.nan))
                p_cylr = r.get("Cyl_Right", r.get("cyl_right", np.nan))
                p_cyll = r.get("Cyl_Left", r.get("cyl_left", np.nan))
                p_bmi = r.get("BMI", r.get("bmi", np.nan))
                p_height = r.get("Height_cm", r.get("height_cm", np.nan))
                p_pir = r.get("PIR", r.get("pir", np.nan))

                f_row = compute_features(
                    p_age, p_sex, p_glasses, p_va_r, p_va_l,
                    p_race, p_kr, p_kl, p_cylr, p_cyll, p_bmi, p_height, p_pir
                )
                processed_rows.append(f_row)

            X_batch = pd.DataFrame(processed_rows)[bundle["features"]]
            probs = bundle["pipeline"].predict_proba(X_batch)[:, 1]
            threshold = float(bundle.get("threshold", 0.42489))

            # Augment original DataFrame
            batch_df["Predicted_Probability"] = np.round(probs, 4)
            batch_df["Screening_Decision"] = ["REFERRAL" if p >= threshold else "LOW RISK" for p in probs]
            batch_df["Risk_Tier"] = [
                "Low" if p < 0.30 else ("Borderline" if p < threshold else ("Elevated" if p < 0.70 else "High"))
                for p in probs
            ]

            st.markdown("---")
            st.markdown("#### 📊 Cohort Screening Summary")

            total_screened = len(batch_df)
            referrals_count = int((batch_df["Screening_Decision"] == "REFERRAL").sum())
            referral_pct = referrals_count / total_screened if total_screened > 0 else 0
            avg_risk = float(probs.mean())

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Screened", f"{total_screened}")
            k2.metric("Referrals Flagged", f"{referrals_count}", f"{referral_pct:.1%}")
            k3.metric("Passed (Low Risk)", f"{total_screened - referrals_count}")
            k4.metric("Mean Cohort Risk", f"{avg_risk:.1%}")

            # Visualizations: Donut chart of Risk Tiers + Histogram of Probabilities
            c_chart1, c_chart2 = st.columns(2)
            with c_chart1:
                tier_counts = batch_df["Risk_Tier"].value_counts().reset_index()
                tier_counts.columns = ["Risk_Tier", "Count"]
                color_map = {
                    "Low": "#38a169",
                    "Borderline": "#d69e2e",
                    "Elevated": "#dd6b20",
                    "High": "#e53e3e"
                }
                fig_donut = px.pie(
                    tier_counts,
                    names="Risk_Tier",
                    values="Count",
                    hole=0.45,
                    title="<b>Cohort Risk Tier Breakdown</b>",
                    color="Risk_Tier",
                    color_discrete_map=color_map
                )
                fig_donut.update_layout(margin=dict(t=40, b=10, l=10, r=10), height=300)
                st.plotly_chart(fig_donut, use_container_width=True)

            with c_chart2:
                fig_hist = px.histogram(
                    batch_df,
                    x="Predicted_Probability",
                    nbins=15,
                    title="<b>Predicted Risk Distribution</b>",
                    color="Screening_Decision",
                    color_discrete_map={"REFERRAL": "#e53e3e", "LOW RISK": "#38a169"},
                    labels={"Predicted_Probability": "Predicted Probability", "Screening_Decision": "Decision"}
                )
                fig_hist.add_vline(x=threshold, line_dash="dash", line_color="black", annotation_text=f"Threshold ({threshold:.2f})")
                fig_hist.update_layout(margin=dict(t=40, b=10, l=10, r=10), height=300)
                st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown("#### 📋 Detailed Cohort Results")
            st.dataframe(batch_df, use_container_width=True)

            # CSV Download
            csv_buffer = io.StringIO()
            batch_df.to_csv(csv_buffer, index=False)
            st.download_button(
                "📥 Download Augmented Screening Results CSV",
                data=csv_buffer.getvalue(),
                file_name="ocuscreen_cohort_results.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Error processing uploaded CSV: {e}")

# ---------------------------------------------------------
# TAB 3: Clinical Evidence & Model Benchmarks
# ---------------------------------------------------------
with tab_benchmarks:
    st.subheader("📊 Research Benchmarks & Validation Results (NHANES v4)")
    st.markdown(
        "Evaluation results across **five machine learning algorithms** and **four feature sets** on the held-out test cohort (n = 2,491)."
    )

    if results_df is not None:
        target_filter = st.radio("Select Target Condition:", ["MYOPIA (General, SE ≤ -0.5 D)", "HIGH_MYOPIA (Severe, SE ≤ -5.0 D)"], horizontal=True)
        target_code = "MYOPIA" if "General" in target_filter else "HIGH_MYOPIA"

        filtered_results = results_df[results_df["Target"] == target_code].copy()

        # Feature Set Mapping for clean display
        set_map = {
            "A_physiological": "Set A: Physiological (Corneal K + Age)",
            "B_plus_demographics": "Set B: + Demographics (Sex, Race, PIR)",
            "C_plus_visual_acuity": "Set C: + Visual Acuity (logMAR)",
            "D_plus_glasses_question": "Set D: + Distance Glasses Question"
        }
        filtered_results["Feature_Set_Display"] = filtered_results["Feature set"].map(lambda x: set_map.get(x, x))

        col_m1, col_m2 = st.columns([1, 1])
        with col_m1:
            fig_auc = px.bar(
                filtered_results,
                x="Feature_Set_Display",
                y="Test ROC-AUC",
                color="Model",
                barmode="group",
                title="<b>Test ROC-AUC by Feature Set & Model</b>",
                labels={"Feature_Set_Display": "Feature Set", "Test ROC-AUC": "Held-Out ROC-AUC"},
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_auc.update_layout(xaxis_tickangle=-15, margin=dict(t=40, b=80, l=40, r=20), height=380)
            st.plotly_chart(fig_auc, use_container_width=True)

        with col_m2:
            fig_f1 = px.bar(
                filtered_results,
                x="Feature_Set_Display",
                y="Test F1",
                color="Model",
                barmode="group",
                title="<b>Test F1 Score by Feature Set & Model</b>",
                labels={"Feature_Set_Display": "Feature Set", "Test F1": "F1 Score"},
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            # Add baseline line for all predicted positive
            baseline_f1 = 0.653 if target_code == "MYOPIA" else 0.089
            fig_f1.add_hline(y=baseline_f1, line_dash="dot", line_color="red", annotation_text=f"Baseline F1 ({baseline_f1})")
            fig_f1.update_layout(xaxis_tickangle=-15, margin=dict(t=40, b=80, l=40, r=20), height=380)
            st.plotly_chart(fig_f1, use_container_width=True)

        st.markdown("#### 🔍 Key Takeaways from CDC NHANES Epidemiological Analysis")
        st.markdown(
            """
            1. **The Visual Acuity Jump (Set B → Set C):** Adding presenting visual acuity elevates ROC-AUC from **0.606 to 0.780** (+0.174). Corneal curvature and demographics alone provide limited predictive power for refractive error.
            2. **The Glasses Question Leap (Set C → Set D):** Adding a single self-report question (*"Do you wear glasses or contacts for distance vision?"*) drives ROC-AUC to **0.873** and MCC to **0.61**. This resolves ambiguity for people presenting with 20/20 vision who are wearing their distance prescription.
            3. **High Myopia (≤ -5.0 D):** While high myopia achieves high ROC-AUC (0.909), the low clinical prevalence (~4.7% in US adolescents) limits unweighted precision.
            4. **Ensemble Consistency:** Random Forest, XGBoost, and HistGradientBoosting achieve virtually identical discrimination (ROC-AUC 0.872–0.873), confirming robustness across algorithmic families.
            """
        )

        st.markdown("#### 📋 Full Results Data Table")
        display_cols = [
            "Feature set", "Model", "Test ROC-AUC", "ROC 95% CI",
            "Test PR-AUC", "Test F1", "Test Sensitivity" if "Test Sensitivity" in filtered_results.columns else "Test Recall",
            "Test Specificity", "Test MCC"
        ]
        available_cols = [c for c in display_cols if c in filtered_results.columns]
        st.dataframe(filtered_results[available_cols], use_container_width=True)

# ---------------------------------------------------------
# TAB 4: Clinical Screening Protocol
# ---------------------------------------------------------
with tab_protocol:
    st.subheader("📋 Clinical Screening Protocol & Practice Guidelines")
    st.markdown(
        """
        ### 🎯 Standard Operating Procedure (SOP) for Community & School Screenings
        
        #### Phase 1: Patient Preparation
        - **Target Population:** Adolescents and young adults aged **12 to 25 years**.
        - **Informed Consent / Assent:** Ensure appropriate consent from guardians or adult patients.
        - **Distance Glasses Verification:** Inquire if the patient currently wears prescribed glasses or contact lenses for distance vision (driving, classroom, television). If yes, test acuity **with distance aids on**.

        #### Phase 2: Visual Acuity Testing
        - Position patient at calibrated distance (typically 20 feet / 6 meters or standard electronic chart distance).
        - Test **Right Eye (OD)** first while occluding the left eye; then test **Left Eye (OS)** while occluding the right eye.
        - Record the lowest line on which the patient correctly identifies at least 50% of optotypes.
        - Convert Snellen to logMAR: $\\text{logMAR} = \\log_{10}(\\frac{\\text{Denominator}}{20})$.
        """
    )

    st.markdown("---")
    st.markdown("#### 📐 Visual Acuity Conversion Matrix")
    va_data = [
        {"Snellen (US)": "20/20", "Metric": "6/6", "Decimal": "1.00", "logMAR": "0.00", "Clinical Interpretation": "Standard / Normal visual acuity"},
        {"Snellen (US)": "20/25", "Metric": "6/7.5", "Decimal": "0.80", "logMAR": "0.10", "Clinical Interpretation": "Minimal blur / Borderline"},
        {"Snellen (US)": "20/30", "Metric": "6/9", "Decimal": "0.67", "logMAR": "0.18", "Clinical Interpretation": "Slight reduction"},
        {"Snellen (US)": "20/40", "Metric": "6/12", "Decimal": "0.50", "logMAR": "0.30", "Clinical Interpretation": "Driver's license standard threshold"},
        {"Snellen (US)": "20/50", "Metric": "6/15", "Decimal": "0.40", "logMAR": "0.40", "Clinical Interpretation": "Moderate distance impairment"},
        {"Snellen (US)": "20/80", "Metric": "6/24", "Decimal": "0.25", "logMAR": "0.60", "Clinical Interpretation": "Substantial impairment"},
        {"Snellen (US)": "20/200", "Metric": "6/60", "Decimal": "0.10", "logMAR": "1.00", "Clinical Interpretation": "Legal blindness cutoff"},
    ]
    st.table(pd.DataFrame(va_data))

    st.markdown("---")
    st.markdown(
        """
        ### 🛡️ Pediatric Myopia Management & Lifestyle Interventions
        When a screening results in a positive referral, standard evidence-based myopia control protocols should be considered:
        - **Outdoor Daylight Exposure:** Minimum **2 hours per day** of natural outdoor light has been shown in clinical trials to slow myopia onset.
        - **Near-Work Visual Hygiene (20-20-20 Rule):** Every 20 minutes of continuous screen or book work, pause for 20 seconds to look at an object 20 feet away.
        - **Optical & Pharmacological Interventions (Upon Optometrist/Ophthalmologist Examination):**
          - Low-dose Atropine eye drops (0.01% to 0.05%)
          - Orthokeratology (Ortho-K overnight reverse-geometry corneal molds)
          - Dual-focus / peripheral defocus soft contact lenses (e.g. MiSight)
          - Defocus Incorporated Multiple Segments (DIMS) spectacle lenses
        """
    )

# ---------------------------------------------------------
# TAB 5: Deployment & Guide
# ---------------------------------------------------------
with tab_deploy:
    st.subheader("🚀 Deployment to Streamlit Community Cloud & Local Setup")

    st.markdown(
        """
        This web application is completely self-contained and ready to deploy online for free on **Streamlit Community Cloud**, accessible to anyone via a web browser.
        """
    )

    st.markdown("### 🌐 How to Deploy to Streamlit Community Cloud (Step-by-Step)")
    st.markdown(
        """
        1. **Create or Initialize a GitHub Repository:**
           - Initialize git inside this folder or upload the project files to a GitHub repository:
             - `app.py`
             - `myopia_screening_model.joblib`
             - `requirements.txt`
             - `myopia_results_v4.csv`
             - `sample_patients.csv`
             - `README.md`
           ```bash
           git init
           git add app.py myopia_screening_model.joblib requirements.txt myopia_results_v4.csv sample_patients.csv README.md
           git commit -m "Deploy OcuScreen AI Streamlit App"
           git branch -M main
           git remote add origin https://github.com/<your-username>/<your-repo-name>.git
           git push -u origin main
           ```
        
        2. **Sign in to Streamlit Community Cloud:**
           - Visit [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
        
        3. **Create New App:**
           - Click **"Create app"** → **"I already have an app"**.
           - Select your GitHub repository (`<your-username>/<your-repo-name>`).
           - Branch: `main`
           - Main file path: `app.py`
           - Click **"Deploy!"**
        
        4. **Live URL:**
           - Streamlit will automatically install dependencies from `requirements.txt` and launch your app with a public URL (e.g. `https://myopia-screening.streamlit.app`).
        """
    )

    st.markdown("---")
    st.markdown("### 💻 How to Run Locally on Your Computer")
    st.markdown(
        """
        In your terminal or command prompt:
        ```bash
        # 1. Install dependencies
        pip install -r requirements.txt

        # 2. Launch Streamlit web app
        streamlit run app.py
        ```
        The browser will open automatically at `http://localhost:8501`.
        """
    )

    st.markdown("---")
    st.markdown("### 📦 Dependencies (`requirements.txt`)")
    st.code(
        """streamlit>=1.35.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.4.0
joblib>=1.3.0
imbalanced-learn>=0.12.0
plotly>=5.18.0""",
        language="text"
    )

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 1rem 0;">
        <strong>OcuScreen AI</strong> • Based on CDC NHANES 1999–2008 Research Cohort • Non-Diagnostic Screening Aid<br>
        Developed for pediatric and adolescent vision screening programs.
    </div>
    """,
    unsafe_allow_html=True,
)
