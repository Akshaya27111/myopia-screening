# 👁️ OcuScreen AI: Pediatric & Adolescent Myopia Screening System

An evidence-based machine learning platform for rapid, non-cycloplegic myopia screening in adolescents and young adults (ages 12–25). Trained and validated across 10 years of CDC National Health and Nutrition Examination Survey (NHANES 1999–2008) multi-cycle epidemiological data.

Unlike conventional eye examinations requiring autorefractors, phoropters, or cycloplegic eye drops, OcuScreen AI screens for myopia using simple, universally available measurements: **presenting visual acuity (Snellen chart)**, **one distance-glasses question**, **demographics**, and **optional keratometry/anthropometrics**.

---

## 🌟 Streamlit Web Application Features

The interactive web application (`app.py`) provides a full suite of screening tools:

1. **🩺 Patient Screener (Single Evaluation):**
   - Quick 1-click loading of preset clinical scenarios (e.g. typical uncorrected myopic teen, emmetropic 20/20 youth, distance glasses wearer, borderline case, high myopia candidate).
   - Form for presenting visual acuity (Snellen 20/20 to 20/400) with automatic logMAR conversions.
   - Distance vision aid question (NHANES VIQ070).
   - Optional biometrics: corneal curvature (K), corneal astigmatism, BMI, height, and socioeconomic PIR.
   - **Interactive Plotly Gauge Indicator** showing predicted risk (0–100%) against the optimal decision threshold (42.5%).
   - Clear clinical referral guidance (Refer vs. Monitor) and factor breakdown.
   - 1-click printable/downloadable patient screening summary report.

2. **📁 Batch Cohort Screener:**
   - Upload CSV rosters for entire schools, community health drives, or epidemiological research.
   - 1-click download of a sample CSV template (`sample_patients.csv`).
   - Generates summary KPI metrics, risk tier breakdown donut chart, and probability distribution histogram.
   - Export enriched CSV with individual probabilities, risk tiers, and referral flags.

3. **📊 Clinical Evidence & Research Benchmarks:**
   - Interactive model performance comparison across five algorithms (Random Forest, XGBoost, HistGradientBoosting, Logistic Regression, Linear SVM).
   - Comparison across all four feature sets (A: Physiological, B: +Demographics, C: +Visual Acuity, D: +Distance Glasses).
   - Interactive charts of ROC-AUC, PR-AUC, F1, and MCC metrics on the held-out test cohort (n = 2,491).

4. **📋 Clinical Screening Protocol:**
   - Standard Operating Procedure (SOP) for school nurses and vision screeners.
   - Comprehensive Snellen to logMAR conversion reference matrix.
   - Evidence-based pediatric myopia management and lifestyle interventions (outdoor daylight exposure, 20-20-20 rule, optical/pharmacological therapies).

5. **🚀 Deployment & Guide:**
   - Complete instructions for deploying to Streamlit Community Cloud and running locally.

---

## 🚀 How to Run Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web App
```bash
streamlit run app.py
```
The application will open in your default browser at `http://localhost:8501`.

---

## 🌐 How to Deploy to Streamlit Community Cloud (Free Hosting)

Deploy your application to a public, 24/7 web link in under 3 minutes:

1. **Push this project to a GitHub repository:**
   ```bash
   git init
   git add app.py myopia_screening_model.joblib requirements.txt myopia_results_v4.csv sample_patients.csv README.md
   git commit -m "Initial commit for OcuScreen AI"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```

2. **Connect to Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
   - Click **"Create app"** → **"I already have an app"**.

3. **Configure Deployment:**
   - **Repository:** `<your-username>/<your-repo-name>`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - Click **"Deploy!"**

Streamlit Community Cloud will automatically build your dependencies from `requirements.txt` and launch your live web URL!

---

## 📊 Held-Out Test Set Performance (n = 2,491)

| Feature Set | Best Model | Test ROC-AUC (95% CI) | Test F1 (95% CI) | Test MCC |
|---|---|---|---|---|
| A: Physiological (corneal + age) | XGBoost | 0.605 (0.581–0.627) | 0.653 | 0.01 |
| B: + Demographics | XGBoost | 0.606 (0.584–0.627) | 0.652 | 0.03 |
| C: + Presenting Visual Acuity | HistGradientBoosting | 0.780 (0.762–0.797) | 0.712 (0.695–0.731) | 0.38 |
| **D: + Distance-Glasses Question** | **Random Forest** | **0.873 (0.859–0.887)** | **0.805 (0.788–0.822)** | **0.61** |

*Note: Predicting all subjects as myopic produces a baseline F1 of 0.653.*

- **Temporal Validation (Train 1999–2004, Test 2005–2008):** ROC-AUC 0.886, F1 0.813.
- **Survey-Weighted (US Youth Population 12–25):** ROC-AUC 0.879, F1 0.809.
- **Decision Threshold:** `0.425` (optimized via out-of-fold F1 on training set).

---

## 📁 Repository Structure

```
myopia_project/
├── app.py                           # Main Streamlit web application
├── myopia_screening_model.joblib   # Trained pipeline (ColumnTransformer + RandomForest)
├── requirements.txt                 # Python dependencies
├── sample_patients.csv              # Template CSV for batch cohort evaluation
├── myopia_results_v4.csv            # Held-out benchmark metrics across models/sets
├── myopia_results_v4.xlsx           # Full benchmark workbook with DeLong tests
├── myopia_model_v4.ipynb            # Training and model evaluation notebook
├── myopia_model_v4_with_outputs.ipynb # Complete notebook run with figures and tables
├── nhanes_data.zip                  # Raw CDC NHANES 1999-2008 data files
└── README.md                        # Documentation and deployment guide
```

---

## 🔒 Design Integrity & Anti-Leakage Rules

- **Strict Non-Leakage:** Autorefractor sphere/cylinder readings, lens prescriptions (VIXPRS/VIXPRC), and post-refraction acuity (VIDROVA/VIDLOVA) are strictly excluded from features.
- **Prior Surgery Exclusion:** Individuals with prior refractive eye surgery (VIQ180 = 1) were excluded.
- **Test Integrity:** Held-out test set evaluated strictly once; optimal thresholds derived from out-of-fold cross-validation.
