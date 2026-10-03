"""
02_data_cleaning.py
Clean the Indian Job Market dataset and create an analysis-ready CSV.

Expected source:
    data/raw/indian_jobs.xlsx (or .csv)

Output:
    data/processed/cleaned_jobs.csv
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
OUTPUT_PATH = OUTPUT_DIR / "cleaned_jobs.csv"

COLUMN_ALIASES = {
    "title": ["title", "job_title", "jobTitle", "job title"],
    "company": ["companyName", "company", "company_name"],
    "location": ["location", "locations"],
    "skills": ["tagsAndSkills", "skills", "skill", "tags_and_skills"],
    "experience": ["experience", "experienceLevel", "experience_level"],
    "min_experience": ["minimumExperience", "minExperience", "minimum_experience"],
    "max_experience": ["maximumExperience", "maxExperience", "maximum_experience"],
    "salary": ["salary", "Salary"],
    "min_salary": ["minimumSalary", "minSalary", "minimum_salary"],
    "max_salary": ["maximumSalary", "maxSalary", "maximum_salary"],
    "currency": ["currency", "Currency"],
    "job_uploaded": ["jobUploaded", "job_uploaded", "postedDate", "datePosted"],
    "employment_type": ["employmentType", "employment_type", "type"],
}

def resolve_raw_file() -> Path:
    """Finds raw dataset regardless of extension or naming variation."""
    candidates = [
        RAW_DIR / "indian_jobs.xlsx",
        RAW_DIR / "indian-job-market-dataset-2025.xlsx",
        RAW_DIR / "indian_jobs.csv",
        Path("data/row/indian-job-market-dataset-2025.xlsx"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find raw dataset. Checked: {[str(p) for p in candidates]}"
    )

def find_column(df, aliases):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        key = alias.strip().lower()
        if key in lookup:
            return lookup[key]
    return None

def clean_text(value):
    if pd.isna(value):
        return np.nan
    val_str = str(value).strip()
    val_str = re.sub(r"\s+", " ", val_str)
    return val_str if val_str else np.nan

def extract_number(value):
    if pd.isna(value):
        return np.nan
    cleaned = str(value).replace(",", "")
    match = re.search(r"\d+(?:\.\d+)?", cleaned)
    return float(match.group()) if match else np.nan

def standardize_skills(value):
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = re.sub(r"[\[\]'\"`()]", "", text)
    text = text.replace("|", ",").replace(";", ",").replace("/", ",")
    
    parts = [re.sub(r"\s+", " ", x).strip() for x in text.split(",")]
    parts = [x for x in parts if x and x not in {"nan", "none", "null", "not disclosed"}]
    
    return ", ".join(dict.fromkeys(parts))

def categorize_experience(exp):
    if pd.isna(exp) or exp <= 1.5:
        return "Entry-Level (0-1.5 yrs)"
    elif exp <= 4.0:
        return "Junior (2-4 yrs)"
    elif exp <= 8.0:
        return "Mid-Senior (5-8 yrs)"
    else:
        return "Senior / Lead (8+ yrs)"

def main():
    raw_path = resolve_raw_file()
    print(f"Loading raw dataset from: {raw_path}")

    if raw_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(raw_path, engine="openpyxl")
    else:
        df = pd.read_csv(raw_path)

    print(f"Initial raw shape: {df.shape}")

    # 1. Drop Unnamed columns & strip whitespace
    unnamed = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    df = df.drop(columns=unnamed, errors="ignore")
    df.columns = [str(c).strip() for c in df.columns]

    # 2. Standardize column names
    rename_map = {}
    for standard_name, aliases in COLUMN_ALIASES.items():
        found = find_column(df, aliases)
        if found and found != standard_name:
            rename_map[found] = standard_name
    df = df.rename(columns=rename_map)

    # 3. Drop records with missing titles immediately
    if "title" in df:
        before_title_drop = len(df)
        df = df[df["title"].notna()].copy()
        print(f"Dropped {before_title_drop - len(df)} rows with missing title.")

    # 4. Remove exact duplicates
    before_dedup = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed exact duplicates: {before_dedup - len(df):,}")

    # 5. Clean text fields
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].map(clean_text)

    # 6. Standardize Skills
    if "skills" in df:
        df["skills"] = df["skills"].map(standardize_skills)

    # 7. Standardize Numerical Ranges
    for col in ["min_experience", "max_experience", "min_salary", "max_salary"]:
        if col in df:
            df[col] = pd.to_numeric(df[col].map(extract_number), errors="coerce")

    # Clean text salary column if 'not disclosed' exists
    if "salary" in df:
        not_disclosed = df["salary"].astype(str).str.lower().str.contains("not disclosed|undisclosed", regex=True)
        df.loc[not_disclosed, "salary"] = np.nan

    # CRITICAL: Convert 0 or negative salary values to NaN (DO NOT DROP ROWS)
    for col in ["min_salary", "max_salary"]:
        if col in df:
            df.loc[df[col] <= 0, col] = np.nan

    # 8. Experience Midpoint & Bracket Creation
    if "min_experience" in df and "max_experience" in df:
        df["experience_years"] = df[["min_experience", "max_experience"]].mean(axis=1)
    elif "experience" in df:
        df["experience_years"] = df["experience"].map(extract_number)

    # Fallback to single bound if one is NaN
    if "min_experience" in df and "experience_years" in df:
        df["experience_years"] = df["experience_years"].fillna(df["min_experience"])
    if "max_experience" in df and "experience_years" in df:
        df["experience_years"] = df["experience_years"].fillna(df["max_experience"])

    df["experience_bracket"] = df["experience_years"].apply(categorize_experience)

    # 9. Salary Midpoint Calculation
    if "min_salary" in df and "max_salary" in df:
        df["salary_mid"] = df[["min_salary", "max_salary"]].mean(axis=1)
        # If one bound is missing, fallback to the available one
        df["salary_mid"] = df["salary_mid"].fillna(df["min_salary"]).fillna(df["max_salary"])
    elif "salary" in df:
        df["salary_mid"] = df["salary"].map(extract_number)

    # Clamp salary to realistic annual market bounds (50k to 1.5 Cr INR)
    # Replaces zeros, negative numbers, or monthly test values with NaN
    if "salary_mid" in df:
        df.loc[(df["salary_mid"] < 50000) | (df["salary_mid"] > 15000000), "salary_mid"] = np.nan

    # 10. Balanced Salary Categorization (3-tier tertiles on valid disclosures only)
    if "salary_mid" in df:
        valid_salaries = df["salary_mid"].dropna()
        print(f"\nTotal rows with verified disclosed salaries: {len(valid_salaries):,} / {len(df):,}")

        if len(valid_salaries) >= 100:
            q1, q2 = valid_salaries.quantile([0.333, 0.666])
            df["salary_category"] = pd.cut(
                df["salary_mid"],
                bins=[-np.inf, q1, q2, np.inf],
                labels=["Low", "Medium", "High"],
                include_lowest=True
            ).astype("string")
            print(f"Salary brackets computed on valid data:")
            print(f"  • Low    : <= ₹{q1:,.0f}")
            print(f"  • Medium : ₹{q1:,.0f} to ₹{q2:,.0f}")
            print(f"  • High   : > ₹{q2:,.0f}")

    # 11. Save Cleaned Dataset
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nFinal cleaned dataset shape: {df.shape}")
    print(f"Saved cleaned file to: {OUTPUT_PATH}")
    print("\nColumns ready for EDA, Skill Gap Analysis & ML:")
    print(df.columns.tolist())

if __name__ == "__main__":
    main()