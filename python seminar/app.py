"""
app.py
User-Friendly AI Career & Salary Navigator.
Designed for job seekers, students, and hiring managers.

Run:
    python -m streamlit run app.py
"""

from collections import Counter
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="CareerPilot | AI Career & Salary Navigator",
    page_icon="🧭",
    layout="wide",
)

DATA_PATH = Path("data/processed/cleaned_jobs.csv")
MODEL_PATH = Path("models/salary_model.pkl")

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None

df = load_data()
model_artifact = load_model()

if df is None:
    st.error("⚠️ Data file not found. Please run backend/02_data_cleaning.py first.")
    st.stop()

# Helper columns
df["title_lower"] = df["title"].fillna("").astype(str).str.lower()
df["skills_lower"] = df["skills"].fillna("").astype(str).str.lower()
df["location_clean"] = df["location"].fillna("").astype(str).apply(lambda x: x.split(",")[0].strip().title())

# Options for UI
top_roles = df["title"].value_counts().head(50).index.tolist()
top_locations = [loc for loc, _ in df["location_clean"].value_counts().head(30).items() if loc and loc != "Nan"]

all_skills_counter = Counter()
for s in df["skills_lower"]:
    all_skills_counter.update([x.strip() for x in str(s).split(",") if x.strip()])
common_skills = [skill.title() for skill, _ in all_skills_counter.most_common(120)]


# --- Dynamic City Multipliers ---
TIER1_CITIES = {"Bengaluru", "Bangalore", "Hyderabad", "Gurgaon", "Gurugram", "Noida", "Delhi", "Mumbai", "Pune"}
TIER2_CITIES = {"Ahmedabad", "Jaipur", "Indore", "Kochi", "Chandigarh", "Vadodara", "Coimbatore", "Surat", "Bhopal", "Lucknow"}

def get_city_adjustment(selected_city: str) -> float:
    city_str = str(selected_city).strip().title()
    if city_str in TIER1_CITIES:
        return 1.10  # +10% premium for Tier-1 cost of living / MNC concentration
    elif city_str in TIER2_CITIES:
        return 0.82  # ~18% lower baseline in regional tech hubs
    return 0.90


def get_grounded_salary_benchmark(data: pd.DataFrame, target_role: str, user_loc: str, user_exp: float, user_skills: list):
    """
    Computes statistically grounded salary benchmarks dynamically taking into account:
    - Target Role
    - Exact Experience Level (strict for 0-year freshers)
    - Location / City Impact
    - Skill Relevance / Match Penalty
    """
    valid_salaries = data[data["salary_mid"].between(60000, 15000000)].copy()
    role_clean = target_role.strip().lower()
    loc_clean = user_loc.strip().lower()
    
    # 1. Experience Filtering Window
    if user_exp <= 0.5:
        # Strictly freshers & interns (0 to 1 yr)
        exp_mask = valid_salaries["experience_years"] <= 1.0
        exp_desc = "Freshers (0 - 1 yr exp)"
    elif user_exp <= 2.5:
        exp_mask = valid_salaries["experience_years"].between(0.5, 3.0)
        exp_desc = f"{user_exp:.1f} yrs exp (Junior 1-3 yr band)"
    elif user_exp <= 6.0:
        exp_mask = valid_salaries["experience_years"].between(user_exp - 1.2, user_exp + 1.2)
        exp_desc = f"{user_exp:.1f} yrs exp (Mid-Level band)"
    else:
        exp_mask = valid_salaries["experience_years"].between(user_exp - 1.5, user_exp + 2.0)
        exp_desc = f"{user_exp:.1f} yrs exp (Senior band)"

    exp_pool = valid_salaries[exp_mask].copy()

    # 2. Filter by Role inside Experience pool
    role_exp_pool = exp_pool[exp_pool["title_lower"].str.contains(role_clean, regex=False)]
    
    # Check City in that specific pool
    city_role_pool = role_exp_pool[role_exp_pool["location_clean"].str.lower().str.contains(loc_clean, regex=False)]

    context_msg = ""
    target_salaries = None

    if len(city_role_pool) >= 3:
        target_salaries = city_role_pool["salary_mid"]
        context_msg = f"Based on {len(city_role_pool)} verified postings in **{user_loc}** for **{exp_desc}**."
    elif len(role_exp_pool) >= 4:
        # Scale pan-India numbers by city cost tier
        city_factor = get_city_adjustment(user_loc)
        target_salaries = role_exp_pool["salary_mid"] * city_factor
        context_msg = f"Pan-India postings for {target_role} ({exp_desc}), adjusted for **{user_loc}** market rates."
    else:
        # If ZERO fresher postings exist for this role (e.g. 0-year Data Engineer)
        all_role_jobs = valid_salaries[valid_salaries["title_lower"].str.contains(role_clean, regex=False)]
        city_factor = get_city_adjustment(user_loc)

        if not all_role_jobs.empty:
            avg_exp_market = all_role_jobs["experience_years"].median()
            med_sal_market = all_role_jobs["salary_mid"].median()

            if user_exp <= 1.0:
                # Direct fresher roles don't exist in bulk; scale down to junior trainee level
                scaled_mid = med_sal_market * 0.35 * city_factor
                scaled_mid = max(250000, min(scaled_mid, 650000))  # Grounded fresher tech range in India (₹2.5L - ₹6.5L)
                target_salaries = pd.Series([scaled_mid * 0.8, scaled_mid, scaled_mid * 1.25])
                context_msg = f"Direct fresher openings for {target_role} are rare. Modeled from standard entry trainee rates in {user_loc}."
            else:
                # Experience-based interpolation
                exp_ratio = min(1.5, max(0.4, user_exp / max(1.0, avg_exp_market)))
                est_mid = med_sal_market * exp_ratio * city_factor
                target_salaries = pd.Series([est_mid * 0.75, est_mid, est_mid * 1.3])
                context_msg = f"Derived from {len(all_role_jobs)} postings across India, adjusted for {user_exp} yrs experience & {user_loc} cost index."
        else:
            # Fallback to general market experience baseline
            target_salaries = exp_pool["salary_mid"] * city_factor
            context_msg = f"Role-specific data limited. Based on general tech market baseline in {user_loc} for {exp_desc}."

    # 3. Apply Skill Relevance Check
    user_skills_clean = {s.strip().lower() for s in user_skills if s.strip()}
    
    # Extract core skills for this role
    role_benchmark_jobs = data[data["title_lower"].str.contains(role_clean, regex=False)]
    role_skills_all = []
    for item in role_benchmark_jobs["skills_lower"].dropna():
        role_skills_all.extend([x.strip() for x in str(item).split(",") if x.strip()])
    
    top_role_skills = {k for k, _ in Counter(role_skills_all).most_common(8)}
    
    skill_factor = 1.0
    if top_role_skills and user_skills_clean:
        overlap = len(user_skills_clean & top_role_skills)
        if overlap == 0:
            skill_factor = 0.75  # 25% discount for tech stack mismatch
            context_msg += " (Adjusted: Your current skills do not match core tools for this role)"
        elif overlap >= 3:
            skill_factor = 1.10  # 10% premium for high-demand skill match

    adjusted_salaries = target_salaries * skill_factor

    low_lpa = round(adjusted_salaries.quantile(0.25) / 100000, 1)
    med_lpa = round(adjusted_salaries.median() / 100000, 1)
    high_lpa = round(adjusted_salaries.quantile(0.75) / 100000, 1)

    return low_lpa, med_lpa, high_lpa, context_msg


# --- App Header ---
st.title("🧭 CareerPilot: AI Career & Salary Navigator")
st.markdown("Data-backed insights for job seekers, recruiters, and students in the Indian tech market.")

# Transparency banner
st.caption("ℹ️ **Notice:** All projections are calculated using historical Indian job market dataset (2025–2026). Actual compensation may vary based on company budget, candidate skill proficiency, and interview evaluation.")
st.write("")

# --- TO THIS ---
tab_skills, tab_salary, tab_explorer = st.tabs([
    "🎯 1. Check My Skill Match",
    "💵 2. Salary Reality Check",
    "📈 3. Job Market Insights",
])

# ==========================================
# TAB 1: SKILL GAP ANALYZER
# ==========================================
with tab_skills:

# # ==========================================
# # TAB 1: SKILL GAP ANALYZER
# # ==========================================
# with tabs[0]:
    st.subheader("Are you ready for your target role?")
    st.caption("Select your desired role and check off the skills you already know.")

    col_in, col_out = st.columns([1, 1.4], gap="large")

    with col_in:
        with st.container(border=True):
            user_target_role = st.selectbox("What role are you targeting?", top_roles)
            user_existing_skills = st.multiselect(
                "Pick the skills you currently possess:",
                options=common_skills,
                default=["Python", "Sql"] if "Python" in common_skills and "Sql" in common_skills else []
            )
            extra_skill_text = st.text_input("Type any other skills you have (separated by commas):", "")
            submit_match = st.button("Analyze My Profile", type="primary", use_container_width=True)

    with col_out:
        if submit_match or user_existing_skills:
            my_skills = {s.strip().lower() for s in user_existing_skills}
            if extra_skill_text:
                my_skills.update([s.strip().lower() for s in extra_skill_text.split(",") if s.strip()])

            matched_df = df[df["title_lower"].str.contains(user_target_role.strip().lower(), regex=False)]
            total_jobs = len(matched_df)

            if total_jobs == 0:
                st.warning(f"No job openings found matching '{user_target_role}'.")
            else:
                role_skills_counter = Counter()
                for item in matched_df["skills_lower"]:
                    role_skills_counter.update(set([s.strip() for s in str(item).split(",") if s.strip()]))

                top_role_skills = role_skills_counter.most_common(10)
                
                matched = []
                missing = []
                total_market_demand = 0
                user_demand_score = 0

                for skill, count in top_role_skills:
                    demand_pct = round((count / total_jobs) * 100, 1)
                    total_market_demand += demand_pct
                    if skill in my_skills:
                        matched.append((skill.title(), demand_pct))
                        user_demand_score += demand_pct
                    else:
                        missing.append((skill.title(), demand_pct))

                score = int(round((user_demand_score / total_market_demand) * 100)) if total_market_demand > 0 else 0

                st.markdown("#### Your Match Results")
                if score >= 75:
                    st.success(f"### 🎉 Ready to Apply: **{score}% Match**")
                    st.write("You possess most of the core skills employers are actively asking for.")
                elif score >= 45:
                    st.warning(f"### ⚡ Almost Ready: **{score}% Match**")
                    st.write("You meet the basic requirements, but picking up 1 or 2 high-demand skills will double your shortlist chances.")
                else:
                    st.error(f"### 📚 Skill Gap Identified: **{score}% Match**")
                    st.write("Focus on learning the essential core tools listed below before applying.")

                st.progress(score / 100)

                col_yes, col_no = st.columns(2)
                with col_yes:
                    with st.container(border=True):
                        st.markdown("**✅ Skills You Already Have:**")
                        if matched:
                            for sk, pct in matched:
                                st.write(f"• **{sk}** *(Demanded in {pct}% of jobs)*")
                        else:
                            st.write("None of the top 10 required skills matched yet.")

                with col_no:
                    with st.container(border=True):
                        st.markdown("**🎯 Priority Skills to Learn Next:**")
                        if missing:
                            for idx, (sk, pct) in enumerate(missing[:4], start=1):
                                st.write(f"**{idx}. {sk}** *(Required in {pct}% of jobs)*")
                        else:
                            st.write("You have all the top-priority skills!")


# ==========================================
# TAB 2: SALARY REALITY CHECK (CITY & EXP AWARE)
# ==========================================
with tab_salary:
    st.subheader("Know Your Market Worth")
    st.caption("Salary benchmarks strictly anchored to your experience tier, city, and tech stack.")

    s_col1, s_col2 = st.columns([1, 1.4], gap="large")

    with s_col1:
        with st.container(border=True):
            sal_role = st.selectbox("Role:", top_roles, key="sal_role")
            sal_loc = st.selectbox("Location:", top_locations, key="sal_loc")
            sal_exp = st.slider("Total Years of Experience:", 0.0, 20.0, 0.0, 0.5)
            sal_skills = st.multiselect("Your Skills:", common_skills, default=["Python"] if "Python" in common_skills else [])
            check_salary = st.button("Check Expected Salary", type="primary", use_container_width=True)

    with s_col2:
        if check_salary:
            low_lpa, med_lpa, high_lpa, context_note = get_grounded_salary_benchmark(
                data=df,
                target_role=sal_role,
                user_loc=sal_loc,
                user_exp=sal_exp,
                user_skills=sal_skills
            )

            if low_lpa is not None:
                st.markdown("#### Expected Salary Range")
                st.caption(f"📌 {context_note}")

                c1, c2, c3 = st.columns(3)
                c1.metric("Starting / Small Firm", f"₹{low_lpa:.1f} LPA")
                c2.metric("Market Average (Median)", f"₹{med_lpa:.1f} LPA")
                c3.metric("Top Tier / Product MNC", f"₹{high_lpa:.1f} LPA")

                if sal_exp == 0.0:
                    st.info(f"💡 **Fresher Insight:** Starting salaries in {sal_loc} for entry-level tech talent generally hover between **₹{low_lpa:.1f} LPA and ₹{med_lpa:.1f} LPA**.")
                else:
                    st.info(f"💡 **Negotiation Tip:** With {sal_exp:.1f} years of experience, a competitive target range is **₹{med_lpa:.1f} LPA to ₹{high_lpa:.1f} LPA**.")

                # Simple comparison bar chart
                breakdown_df = pd.DataFrame({
                    "Tier": ["Entry / Small Firms", "Standard Market Rate", "High Paying / MNC"],
                    "Salary (LPA)": [low_lpa, med_lpa, high_lpa]
                })
                fig = px.bar(
                    breakdown_df,
                    x="Tier",
                    y="Salary (LPA)",
                    color="Tier",
                    text_auto=".1f",
                    title=f"Realistic Compensation Spread: {sal_role} ({sal_exp:.1f} Yrs in {sal_loc})"
                )
                fig.update_layout(showlegend=False, yaxis_title="Annual Package (Lakhs INR)")
                st.plotly_chart(fig, use_container_width=True)

                # Dashboard Disclaimer Callout
                st.caption("⚠️ **Note:** This projection is derived strictly from verified Indian job market records. Actual compensation may vary based on candidate skill proficiency, company size, and performance during technical rounds.")
            else:
                st.warning("Insufficient data available to build a reliable salary benchmark for this profile.")


# ==========================================
# TAB 3: SIMPLE MARKET INSIGHTS
# ==========================================
with tab_explorer:
    st.subheader("Where is the Market Heading?")
    st.caption("Macro hiring trends across India.")

    m_col1, m_col2 = st.columns(2)

    with m_col1:
        top_skills_series = pd.DataFrame(all_skills_counter.most_common(8), columns=["Skill", "Job Openings"])
        top_skills_series["Skill"] = top_skills_series["Skill"].str.title()
        fig_sk = px.bar(
            top_skills_series,
            x="Job Openings",
            y="Skill",
            orientation="h",
            title="Top 8 Most In-Demand Skills in India",
            color="Job Openings",
            color_continuous_scale="Teal"
        )
        fig_sk.update_layout(yaxis={"autorange": "reversed"})
        st.plotly_chart(fig_sk, use_container_width=True)

    with m_col2:
        top_loc_clean = (
            df["location_clean"].value_counts().head(8).reset_index()
        )
        top_loc_clean.columns = ["City", "Openings"]
        fig_loc = px.bar(
            top_loc_clean,
            x="Openings",
            y="City",
            orientation="h",
            title="Top 8 Cities with Most Tech Openings",
            color="Openings",
            color_continuous_scale="Purples"
        )
        fig_loc.update_layout(yaxis={"autorange": "reversed"})
        st.plotly_chart(fig_loc, use_container_width=True)