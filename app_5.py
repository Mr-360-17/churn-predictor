"""
app.py — Customer Churn Predictor
Trains model on startup from churn_data.csv — no pickle version issues.
"""
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Churn Risk Dashboard",
    page_icon="🔮",
    layout="wide"
)

FEATURES = [
    "tenure_months","login_count_30d","feature_adoption_pct",
    "support_tickets_90d","nps_score","nps_missing",
    "contract_end_days","last_login_days_ago","billing_failures_6m",
    "integrations_active","onboarding_complete","cs_health_score",
    "cs_health_missing","monthly_revenue","company_size_enc","plan_enc"
]

def clean_data(df):
    df = df.copy()
    bool_map = {"True":1,"true":1,"1":1,"False":0,"false":0,"0":0}
    df["onboarding_complete"] = df["onboarding_complete"].map(bool_map).fillna(0).astype(int)
    df["nps_score"] = pd.to_numeric(df["nps_score"].replace({"N/A":np.nan,"MISSING":np.nan}), errors="coerce")
    df["nps_missing"] = df["nps_score"].isna().astype(int)
    df["nps_score"] = df["nps_score"].fillna(df["nps_score"].median())
    df["company_size"] = df["company_size"].replace("MISSING","Unknown")
    df["cs_health_missing"] = df["cs_health_score"].isna().astype(int)
    df["cs_health_score"] = df["cs_health_score"].fillna(df["cs_health_score"].median())
    size_map = {"Small":1,"Medium":2,"Large":3,"Enterprise":4,"Unknown":0}
    plan_map = {"Starter":1,"Pro":2,"Business":3,"Enterprise":4}
    df["company_size_enc"] = df["company_size"].map(size_map).fillna(0)
    df["plan_enc"] = df["plan"].map(plan_map).fillna(1)
    return df

@st.cache_resource(show_spinner="Training model on startup...")
def train_model():
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    df = pd.read_csv("churn_data.csv")
    df_clean = clean_data(df)

    X = df_clean[FEATURES]
    y = df_clean["churned_90d"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
        random_state=42, eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    return model, auc

def risk_label(prob):
    if prob >= 0.6: return "🔴 HIGH"
    if prob >= 0.3: return "🟡 MEDIUM"
    return "🟢 LOW"

def explain_risk(row):
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
        reasons.append(f"Contract {'expired' if days < 0 else 'renews'} in {abs(days)} days")
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

with st.sidebar:
    st.header("📊 How to Use")
    st.markdown("""
    1. Use the **demo dataset** or upload your own CSV
    2. View risk scores for all accounts
    3. Click any account to see why they're at risk
    4. Export the at-risk list for your CS team
    
    **Risk Levels:**
    - 🔴 HIGH (≥60%) — Call this week
    - 🟡 MEDIUM (30-60%) — Monitor closely
    - 🟢 LOW (<30%) — Routine check-in
    """)

# Train model
model, auc = train_model()
st.success(f"✅ Model ready — Test ROC-AUC: {auc:.3f}")

# Load demo data
demo_df = pd.read_csv("churn_data.csv")
demo_clean = clean_data(demo_df)

tab1, tab2 = st.tabs(["📊 Demo Dataset", "📁 Upload Your Data"])

with tab1:
    st.subheader("Demo: 1,500 Synthetic B2B SaaS Accounts")

    X_demo = demo_clean[FEATURES]
    probs  = model.predict_proba(X_demo)[:,1]
    demo_clean["churn_probability"] = probs
    demo_clean["risk_level"] = [risk_label(p) for p in probs]
    demo_clean["account_id"] = demo_df["account_id"]

    high   = (demo_clean["risk_level"] == "🔴 HIGH").sum()
    medium = (demo_clean["risk_level"] == "🟡 MEDIUM").sum()
    low    = (demo_clean["risk_level"] == "🟢 LOW").sum()
    rev_at_risk = demo_df.loc[demo_clean["risk_level"] == "🔴 HIGH", "monthly_revenue"].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🔴 High Risk",   str(high))
    col2.metric("🟡 Medium Risk", str(medium))
    col3.metric("🟢 Low Risk",    str(low))
    col4.metric("💰 MRR at Risk", f"${rev_at_risk:,.0f}")

    st.markdown("---")
    risk_filter = st.selectbox("Filter by risk level", ["All","🔴 HIGH","🟡 MEDIUM","🟢 LOW"])
    filtered = demo_clean if risk_filter == "All" else demo_clean[demo_clean["risk_level"] == risk_filter]
    filtered = filtered.sort_values("churn_probability", ascending=False)

    show_cols = ["account_id","risk_level","churn_probability","login_count_30d",
                 "last_login_days_ago","feature_adoption_pct","support_tickets_90d"]
    out = filtered[show_cols].copy()
    out["churn_probability"] = (out["churn_probability"]*100).round(1).astype(str) + "%"
    st.dataframe(out.head(50).reset_index(drop=True), use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 Account Deep-Dive")
    selected = st.selectbox("Select account", filtered["account_id"].tolist()[:50])
    if selected:
        row   = demo_clean[demo_clean["account_id"] == selected].iloc[0]
        prob  = row["churn_probability"]
        level = row["risk_level"]
        col1, col2 = st.columns([1,2])
        with col1:
            st.metric("Churn Probability", f"{prob*100:.1f}%")
            st.metric("Risk Level", level)
            st.metric("Monthly Revenue", f"${demo_df.loc[demo_df['account_id']==selected,'monthly_revenue'].values[0]:,.0f}")
        with col2:
            st.markdown("**🚩 Why is this account at risk?**")
            for r in explain_risk(row):
                st.markdown(f"- {r}")
        st.markdown("**📞 Recommended Action:**")
        if level == "🔴 HIGH":
            st.error("Call immediately. Focus on re-engagement. Offer a success review session.")
        elif level == "🟡 MEDIUM":
            st.warning("Schedule check-in within 2 weeks. Share relevant feature tutorials.")
        else:
            st.success("Routine quarterly review. Good candidate for upsell conversation.")

    csv = filtered[show_cols].copy()
    csv["churn_probability"] = (csv["churn_probability"]*100).round(1)
    st.download_button("📥 Export At-Risk Accounts", csv.to_csv(index=False), "at_risk_accounts.csv")

with tab2:
    st.subheader("Upload Your Own Customer Data")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        raw = pd.read_csv(uploaded)
        df2 = clean_data(raw)
        missing = [c for c in FEATURES if c not in df2.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            probs2 = model.predict_proba(df2[FEATURES])[:,1]
            df2["churn_probability"] = probs2
            df2["risk_level"] = [risk_label(p) for p in probs2]
            st.success(f"Scored {len(df2):,} accounts!")
            df2_sorted = df2.sort_values("churn_probability", ascending=False)
            st.dataframe(df2_sorted[["account_id","risk_level","churn_probability"]].head(20).reset_index(drop=True))
    else:
        st.info("Upload a CSV with the same columns as the demo dataset.")
