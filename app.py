"""
app.py — Customer Churn Predictor
A non-technical stakeholder tool to identify at-risk accounts and understand why.
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(
    page_title="Churn Risk Dashboard",
    page_icon="🔮",
    layout="wide"
)

# ── Load model artifacts ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    import pickle
    with open("model_artifacts.pkl", "rb") as f:
        return pickle.load(f)

# ── Data cleaning (mirrors notebook) ─────────────────────────────────────────
def clean_data(df):
    df = df.copy()

    # Fix boolean column
    bool_map = {"True": 1, "true": 1, "1": 1, "False": 0, "false": 0, "0": 0}
    df["onboarding_complete"] = df["onboarding_complete"].map(bool_map).fillna(0).astype(int)

    # Fix NPS score
    df["nps_score"] = pd.to_numeric(df["nps_score"].replace({"N/A": np.nan, "MISSING": np.nan}), errors="coerce")
    df["nps_missing"] = df["nps_score"].isna().astype(int)
    df["nps_score"] = df["nps_score"].fillna(df["nps_score"].median())

    # Fix company_size
    df["company_size"] = df["company_size"].replace("MISSING", "Unknown")

    # Fix cs_health_score
    df["cs_health_missing"] = df["cs_health_score"].isna().astype(int)
    df["cs_health_score"] = df["cs_health_score"].fillna(df["cs_health_score"].median())

    # Encode categoricals
    size_map = {"Small": 1, "Medium": 2, "Large": 3, "Enterprise": 4, "Unknown": 0}
    plan_map = {"Starter": 1, "Pro": 2, "Business": 3, "Enterprise": 4}
    df["company_size_enc"] = df["company_size"].map(size_map).fillna(0)
    df["plan_enc"]         = df["plan"].map(plan_map).fillna(1)

    return df

FEATURES = [
    "tenure_months", "login_count_30d", "feature_adoption_pct",
    "support_tickets_90d", "nps_score", "nps_missing",
    "contract_end_days", "last_login_days_ago", "billing_failures_6m",
    "integrations_active", "onboarding_complete", "cs_health_score",
    "cs_health_missing", "monthly_revenue", "company_size_enc", "plan_enc"
]

RISK_COLORS = {"🔴 HIGH": "#ffcccc", "🟡 MEDIUM": "#fff3cd", "🟢 LOW": "#d4edda"}

def risk_label(prob):
    if prob >= 0.6: return "🔴 HIGH"
    if prob >= 0.3: return "🟡 MEDIUM"
    return "🟢 LOW"

def explain_risk(row):
    """Generate plain-English explanation of churn drivers."""
    reasons = []
    if row.get("login_count_30d", 99) < 5:
        reasons.append(f"Very low login activity ({int(row['login_count_30d'])} logins in 30 days)")
    if row.get("last_login_days_ago", 0) > 30:
        reasons.append(f"Last login was {int(row['last_login_days_ago'])} days ago")
    if row.get("feature_adoption_pct", 100) < 20:
        reasons.append(f"Only using {row['feature_adoption_pct']:.0f}% of available features")
    if row.get("support_tickets_90d", 0) > 3:
        reasons.append(f"Raised {int(row['support_tickets_90d'])} support tickets in 90 days")
    if row.get("billing_failures_6m", 0) > 0:
        reasons.append(f"{int(row['billing_failures_6m'])} billing failure(s) in last 6 months")
    if row.get("contract_end_days", 999) < 60:
        days = int(row["contract_end_days"])
        if days < 0:
            reasons.append(f"Contract expired {abs(days)} days ago")
        else:
            reasons.append(f"Contract renews in {days} days")
    if row.get("nps_missing", 0) == 1:
        reasons.append("No NPS score on record")
    elif row.get("nps_score", 10) < 5:
        reasons.append(f"Low NPS score ({row['nps_score']:.0f}/10)")
    if row.get("integrations_active", 99) < 2:
        reasons.append(f"Only {int(row['integrations_active'])} integration(s) active (low stickiness)")
    if not reasons:
        reasons.append("No major red flags detected")
    return reasons

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🔮 Customer Churn Risk Dashboard")
st.markdown("*Predict which accounts are at risk of churning — and understand why.*")

# Sidebar
with st.sidebar:
    st.header("📊 How to Use")
    st.markdown("""
    1. **Upload** your customer CSV file
    2. **View** risk scores for all accounts
    3. **Click** any account to see why they're at risk
    4. **Export** the at-risk list for your CS team
    
    **Risk Levels:**
    - 🔴 HIGH (≥60%) — Call this week
    - 🟡 MEDIUM (30-60%) — Monitor closely  
    - 🟢 LOW (<30%) — Routine check-in
    """)

    st.header("📋 Required Columns")
    st.markdown("""
    `account_id`, `tenure_months`, `login_count_30d`,
    `feature_adoption_pct`, `support_tickets_90d`,
    `nps_score`, `contract_end_days`, `last_login_days_ago`,
    `billing_failures_6m`, `integrations_active`,
    `onboarding_complete`, `cs_health_score`,
    `monthly_revenue`, `company_size`, `plan`
    """)

# Check for model
if not os.path.exists("model_artifacts.pkl"):
    st.warning("⚠️ Model not trained yet. Please run the notebook first to generate `model_artifacts.pkl`.")
    st.info("Or upload the sample dataset below to see a demo.")

    st.markdown("### 📥 Download Sample Data")
    sample = pd.DataFrame({
        "account_id": ["ACCT-0001", "ACCT-0002", "ACCT-0003"],
        "company_size": ["Medium", "Small", "Enterprise"],
        "plan": ["Pro", "Starter", "Business"],
        "monthly_revenue": [1200, 300, 5000],
        "tenure_months": [6, 2, 24],
        "login_count_30d": [2, 1, 20],
        "feature_adoption_pct": [15, 8, 75],
        "support_tickets_90d": [5, 2, 0],
        "nps_score": ["3", "N/A", "9"],
        "contract_end_days": [15, 200, 90],
        "last_login_days_ago": [45, 60, 3],
        "billing_failures_6m": [1, 0, 0],
        "integrations_active": [1, 0, 5],
        "onboarding_complete": ["True", "False", "True"],
        "cs_health_score": [3.0, None, 8.0],
    })
    st.dataframe(sample)
    st.stop()

# Load model
artifacts = load_artifacts()
model     = artifacts["model"]
threshold = artifacts.get("threshold", 0.4)

# File upload
uploaded = st.file_uploader("📁 Upload customer CSV", type=["csv"])

if uploaded:
    raw_df = pd.read_csv(uploaded)
    st.success(f"✅ Loaded {len(raw_df):,} accounts")

    # Clean and predict
    df = clean_data(raw_df)
    missing_cols = [c for c in FEATURES if c not in df.columns]

    if missing_cols:
        st.error(f"Missing columns after cleaning: {missing_cols}")
        st.stop()

    X         = df[FEATURES]
    probs     = model.predict_proba(X)[:, 1]
    df["churn_probability"] = probs
    df["risk_level"]        = [risk_label(p) for p in probs]

    # Summary metrics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    high   = (df["risk_level"] == "🔴 HIGH").sum()
    medium = (df["risk_level"] == "🟡 MEDIUM").sum()
    low    = (df["risk_level"] == "🟢 LOW").sum()
    rev_at_risk = raw_df.loc[df["risk_level"] == "🔴 HIGH", "monthly_revenue"].sum() if "monthly_revenue" in raw_df else 0

    col1.metric("🔴 High Risk",   f"{high}")
    col2.metric("🟡 Medium Risk", f"{medium}")
    col3.metric("🟢 Low Risk",    f"{low}")
    col4.metric("💰 MRR at Risk", f"${rev_at_risk:,.0f}")

    # At-risk accounts table
    st.markdown("---")
    st.subheader("🚨 Accounts Requiring Attention")

    risk_filter = st.selectbox("Filter by risk level", ["All", "🔴 HIGH", "🟡 MEDIUM", "🟢 LOW"])

    display_df = df.copy()
    if risk_filter != "All":
        display_df = display_df[display_df["risk_level"] == risk_filter]

    display_df = display_df.sort_values("churn_probability", ascending=False)

    show_cols = ["account_id", "risk_level", "churn_probability", "plan",
                 "monthly_revenue", "login_count_30d", "last_login_days_ago",
                 "feature_adoption_pct", "support_tickets_90d"]
    show_cols = [c for c in show_cols if c in display_df.columns]

    out = display_df[show_cols].copy()
    out["churn_probability"] = (out["churn_probability"] * 100).round(1).astype(str) + "%"

    st.dataframe(out.reset_index(drop=True), use_container_width=True)

    # Account deep-dive
    st.markdown("---")
    st.subheader("🔍 Account Deep-Dive")

    account_list = display_df["account_id"].tolist()
    selected = st.selectbox("Select an account to investigate", account_list)

    if selected:
        row = df[df["account_id"] == selected].iloc[0]
        raw_row = raw_df[raw_df["account_id"] == selected].iloc[0]
        prob = row["churn_probability"]
        level = row["risk_level"]

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Churn Probability", f"{prob*100:.1f}%")
            st.metric("Risk Level", level)
            if "monthly_revenue" in raw_row:
                st.metric("Monthly Revenue", f"${raw_row['monthly_revenue']:,.0f}")

        with col2:
            st.markdown("**🚩 Why is this account at risk?**")
            reasons = explain_risk(row)
            for r in reasons:
                st.markdown(f"- {r}")

        st.markdown("**📞 Recommended Action:**")
        if level == "🔴 HIGH":
            st.error("Call immediately. Focus on re-engagement and understanding pain points. Offer a success review session.")
        elif level == "🟡 MEDIUM":
            st.warning("Schedule a check-in within 2 weeks. Share relevant feature tutorials. Monitor login activity.")
        else:
            st.success("Routine quarterly check-in. Good candidate for upsell conversation.")

    # Export
    st.markdown("---")
    high_risk = df[df["risk_level"] == "🔴 HIGH"][show_cols].copy()
    high_risk["churn_probability"] = (high_risk["churn_probability"] * 100).round(1)
    csv = high_risk.to_csv(index=False)
    st.download_button(
        "📥 Export High-Risk Accounts (CSV)",
        csv,
        "high_risk_accounts.csv",
        "text/csv"
    )

else:
    st.info("👆 Upload a CSV file to get started. Use the sample data format shown in the sidebar.")

    # Demo with built-in data
    if os.path.exists("churn_data.csv"):
        st.markdown("### 🎯 Or try with the built-in demo dataset")
        if st.button("Load Demo Data"):
            demo = pd.read_csv("churn_data.csv").drop(columns=["churned_90d"], errors="ignore")
            df2  = clean_data(demo)
            X2   = df2[FEATURES]
            probs2 = model.predict_proba(X2)[:, 1]
            df2["churn_probability"] = probs2
            df2["risk_level"]        = [risk_label(p) for p in probs2]

            high2 = (df2["risk_level"] == "🔴 HIGH").sum()
            med2  = (df2["risk_level"] == "🟡 MEDIUM").sum()
            st.success(f"Demo loaded! {high2} high-risk, {med2} medium-risk accounts found.")

            top10 = df2.nlargest(10, "churn_probability")[["account_id","risk_level","churn_probability","plan","login_count_30d","feature_adoption_pct"]]
            top10["churn_probability"] = (top10["churn_probability"]*100).round(1).astype(str) + "%"
            st.subheader("Top 10 At-Risk Accounts")
            st.dataframe(top10.reset_index(drop=True), use_container_width=True)
