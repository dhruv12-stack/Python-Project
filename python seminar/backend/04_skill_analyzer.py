"""
04_skill_analyzer.py
Skill extraction, normalization and skill-gap matching.

This module can be imported by Streamlit:

    from backend.04_skill_analyzer import ...

Because Python module names cannot normally start with a number when imported
using the standard syntax, the recommended Streamlit approach is to rename
this file to skill_analyzer.py OR load it with importlib.

For simplicity, app.py can also copy these functions directly.
"""

import re
from collections import Counter
import pandas as pd

COMMON_ALIASES = {
    "ml": "machine learning",
    "machine-learning": "machine learning",
    "ai": "artificial intelligence",
    "js": "javascript",
    "ts": "typescript",
    "postgres": "postgresql",
    "mongo": "mongodb",
    "powerbi": "power bi",
}

def normalize_skill(skill):
    skill = str(skill).strip().lower()
    skill = re.sub(r"\s+", " ", skill)
    return COMMON_ALIASES.get(skill, skill)

def parse_skills(value):
    if value is None:
        return []

    text = str(value).lower()
    text = text.replace("|", ",").replace(";", ",").replace("/", ",")
    skills = []

    for item in text.split(","):
        item = normalize_skill(item)
        if item and item not in {"nan", "none", "null"}:
            skills.append(item)

    return list(dict.fromkeys(skills))

def build_skill_frequency(df, skills_column="skills"):
    counter = Counter()

    if skills_column not in df.columns:
        return counter

    for value in df[skills_column].fillna(""):
        counter.update(parse_skills(value))

    return counter

def get_required_skills(df, job_title, title_column="title", skills_column="skills", top_n=20):
    if title_column not in df.columns or skills_column not in df.columns:
        return []

    job_title = str(job_title).strip().lower()

    matched = df[
        df[title_column].fillna("").str.lower().str.contains(
            re.escape(job_title), regex=True
        )
    ]

    counter = build_skill_frequency(matched, skills_column)
    return [skill for skill, _ in counter.most_common(top_n)]

def calculate_skill_gap(user_skills, required_skills):
    user = {normalize_skill(x) for x in user_skills if str(x).strip()}
    required = {normalize_skill(x) for x in required_skills if str(x).strip()}

    matched = sorted(user & required)
    missing = sorted(required - user)

    score = (len(matched) / len(required) * 100) if required else 0.0

    return {
        "match_score": round(score, 2),
        "matched_skills": matched,
        "missing_skills": missing,
        "required_skills": sorted(required),
    }

if __name__ == "__main__":
    print("Skill analyzer module loaded successfully.")
