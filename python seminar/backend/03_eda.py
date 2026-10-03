"""
03_eda.py
Exploratory Data Analysis, statistics and visualizations.

Run:
    python backend/03_eda.py

Charts are saved in:
    outputs/eda/
"""

from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = Path("data/processed/cleaned_jobs.csv")
OUTPUT_DIR = Path("outputs/eda")

def save_bar(series, title, filename, xlabel="Count"):
    series = series.sort_values(ascending=True)
    plt.figure(figsize=(10, 6))
    series.plot(kind="barh")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=160)
    plt.close()

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run 02_data_cleaning.py first.")

    df = pd.read_csv(DATA_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n========== DATASET ==========")
    print(df.shape)

    # Job titles
    if "title" in df:
        top_titles = df["title"].value_counts().head(10)
        print("\nTop 10 job titles:")
        print(top_titles)
        save_bar(top_titles, "Top 10 Job Titles", "top_job_titles.png")

    # Locations
    if "location" in df:
        top_locations = df["location"].value_counts().head(10)
        print("\nTop 10 locations:")
        print(top_locations)
        save_bar(top_locations, "Top 10 Job Locations", "top_locations.png")

    # Experience
    if "experience_years" in df:
        exp = pd.to_numeric(df["experience_years"], errors="coerce").dropna()
        if not exp.empty:
            print("\nExperience statistics:")
            print(exp.describe())
            plt.figure(figsize=(9, 5))
            exp.clip(lower=0).plot(kind="hist", bins=20)
            plt.title("Experience Distribution")
            plt.xlabel("Experience (years)")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "experience_distribution.png", dpi=160)
            plt.close()

    # Salary
    if "salary_mid" in df:
        salary = pd.to_numeric(df["salary_mid"], errors="coerce").dropna()
        if not salary.empty:
            print("\nSalary statistics:")
            print(salary.describe())
            plt.figure(figsize=(9, 5))
            salary.plot(kind="hist", bins=30)
            plt.title("Salary Distribution")
            plt.xlabel("Salary")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "salary_distribution.png", dpi=160)
            plt.close()

    # Skills
    if "skills" in df:
        skill_counter = {}
        for value in df["skills"].fillna(""):
            for skill in str(value).split(","):
                skill = re.sub(r"\s+", " ", skill.strip().lower())
                if skill:
                    skill_counter[skill] = skill_counter.get(skill, 0) + 1

        top_skills = pd.Series(skill_counter).sort_values(ascending=False).head(15)
        print("\nTop skills:")
        print(top_skills)
        save_bar(top_skills, "Top 15 In-Demand Skills", "top_skills.png")

    # Correlation for numeric columns
    numeric = df.select_dtypes(include=np.number)
    if numeric.shape[1] >= 2:
        corr = numeric.corr(numeric_only=True)
        print("\nCorrelation matrix:")
        print(corr.round(3).to_string())
        corr.to_csv(OUTPUT_DIR / "correlation_matrix.csv")

    # Simple scientific-method hypothesis setup
    if {"experience_years", "salary_mid"}.issubset(df.columns):
        test_df = df[["experience_years", "salary_mid"]].dropna()
        if len(test_df) >= 3:
            correlation = test_df["experience_years"].corr(test_df["salary_mid"])
            print("\n========== SCIENTIFIC METHOD ==========")
            print("Research question: Is experience related to salary?")
            print("H0: Experience and salary have no linear relationship.")
            print("H1: Experience and salary have a linear relationship.")
            print(f"Pearson correlation: {correlation:.4f}")

    print(f"\nCharts saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
