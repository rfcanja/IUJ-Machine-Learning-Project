# app.py

# app.py

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Senior High School and Career Guidance Recommender",
    page_icon="🎓",
    layout="centered",
)

# ---------------------------------------------------------
# 1. Helper: train model from training_dataset_5.csv
# ---------------------------------------------------------

def train_model_from_csv():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "training_dataset_5.csv")

    st.warning(
        "Using fallback: training model from training_dataset_5.csv on this server. "
        "This may take a few seconds."
    )

    df = pd.read_csv(data_path)

    TARGET_COL = "strand"
    ML_STRANDS = ["ABM", "HUMSS", "STEM", "GAS", "TVL"]
    df = df[df[TARGET_COL].isin(ML_STRANDS)].copy()

    numeric_features = [
        "region_code",
        "absence_days",
        "extracurricular_activities",
        "weekly_self_study_hours",
        "math_score",
        "history_score",
        "physics_score",
        "chemistry_score",
        "biology_score",
        "english_score",
        "mapeh_score",
        "stem_index",
        "humss_index",
        "arts_index",
        "abm_index",
        "apt_numerical_reasoning",
        "apt_verbal_reasoning",
        "apt_scientific_reasoning",
        "grit_score",
        "aspiration_strength",
        "shs_stem_available",
        "shs_abm_available",
        "shs_humss_available",
        "shs_tvl_available",
        "ict_skill",
    ]

    categorical_features = [
        "gender",
        "varsity",
        "career_aspiration",
        "learning_style",
        "preferred_work_type",
    ]

    # keep only existing columns
    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]

    X = df[numeric_features + categorical_features]
    y = df[TARGET_COL]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight="balanced",
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", clf),
        ]
    )

    # quick train; we don't even need a test split just to run the app
    pipe.fit(X, y)

    return pipe, numeric_features, categorical_features, TARGET_COL, ML_STRANDS

# ---------------------------------------------------------
# 2. Try to load pickle; if it fails, train from CSV
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "strand_model_best.pkl")

    try:
        data = joblib.load(model_path)
        st.info("Loaded pre-trained model from strand_model_best.pkl")
        return (
            data["pipeline"],
            data["numeric_features"],
            data["categorical_features"],
            data["target_col"],
            data["ml_strands"],
        )
    except Exception as e:
        st.warning(
            f"Could not load pre-trained model from `{model_path}`.\n\n"
            f"Reason: {e}\n\n"
            "The app will instead train a fresh model from the dataset."
        )
        return train_model_from_csv()

pipeline, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COL, ML_STRANDS = load_model()


# --- Helper mappings ---

REGION_OPTIONS = {
    13: "13 - National Capital Region (NCR)",
    14: "14 - Cordillera Administrative Region (CAR)",
    1:  "1 - Region I (Ilocos Region)",
    2:  "2 - Region II (Cagayan Valley)",
    3:  "3 - Region III (Central Luzon)",
    4:  "4 - Region IVA (CALABARZON)",
    5:  "5 - Region V (Bicol Region)",
    6:  "6 - Region VI (Western Visayas)",
    7:  "7 - Region VII (Central Visayas)",
    8:  "8 - Region VIII (Eastern Visayas)",
    9:  "9 - Region IX (Zamboanga Peninsula)",
    10: "10 - Region X (Northern Mindanao)",
    11: "11 - Region XI (Davao Region)",
    12: "12 - Region XII (SOCCSKSARGEN)",
    16: "16 - Region XIII (Caraga)",
    15: "15 - BARMM",
    17: "17 - MIMAROPA Region",
}

def strand_to_track(strand: str) -> str:
    if strand in ["ABM", "HUMSS", "STEM", "GAS"]:
        return "Academic"
    elif strand == "TVL":
        return "Technical-Vocational-Livelihood (TVL)"
    elif strand == "Sports":
        return "Sports"
    elif strand == "Arts":
        return "Arts and Design"
    else:
        return "Other"

def strand_to_course_cluster(strand: str) -> str:
    if strand == "ABM":
        return "Business & Management"
    elif strand == "HUMSS":
        return "Social Sciences & Humanities"
    elif strand == "STEM":
        return "Science, Technology & Engineering"
    elif strand == "GAS":
        return "General / Undecided"
    elif strand == "TVL":
        return "Technical-Vocational Programs"
    elif strand == "Sports":
        return "Sports Science, PE & Coaching"
    elif strand == "Arts":
        return "Arts, Design, Media & Communication"
    else:
        return "Other"

def strand_to_careers(strand: str):
    if strand == "ABM":
        return [
            "Accountant / CPA",
            "Entrepreneur / Business Owner",
            "Marketing or Finance Professional",
        ]
    if strand == "HUMSS":
        return [
            "Lawyer / Legal Professional",
            "Teacher / Professor",
            "Psychologist / Social Worker",
        ]
    if strand == "STEM":
        return [
            "Engineer",
            "Doctor / Health Professional",
            "IT / Data Scientist",
        ]
    if strand == "GAS":
        return [
            "Undecided – explore multiple fields",
            "Future public servant",
            "Multidisciplinary professional",
        ]
    if strand == "TVL":
        return [
            "Technician / Skilled Trades",
            "Chef / Hospitality Professional",
            "Automotive / Electrical / ICT Specialist",
        ]
    if strand == "Sports":
        return [
            "Athlete",
            "Coach / Trainer",
            "PE Teacher / Sports Manager",
        ]
    if strand == "Arts":
        return [
            "Designer / Visual Artist",
            "Performer / Musician / Actor",
            "Media & Creative Professional",
        ]
    return ["Various career paths depending on your interests."]

# --- App UI ---

st.title("🎓Contextual Senior High School Career Guidance Recommender")

st.write(
    "Answer the questions below. The system will suggest a suitable **SHS strand**, "
    "its **track and course cluster**, and some **possible careers**. "
    "Sports and Arts recommendations are based on special rules; the other strands "
    "come from a trained machine learning model."
)

# Sidebar basic info
st.sidebar.header("Student Info")
student_id = st.sidebar.text_input("Student ID (optional)", value="S001")

gender = st.sidebar.selectbox(
    "Gender",
    options=["Male", "Female", "Prefer not to say"],
)

varsity = st.sidebar.selectbox(
    "Varsity / School Team Member?",
    options=["No", "Yes"],
)

# Region selection
st.sidebar.subheader("Location")
region_label = st.sidebar.selectbox(
    "Region",
    options=list(REGION_OPTIONS.values()),
)

region_code = [
    code for code, label in REGION_OPTIONS.items() if label == region_label
][0]

# Academic section
st.subheader("📚 Academic Performance")

math_score = st.slider("Math Score", 0, 100, 85)
physics_score = st.slider("Physics Score", 0, 100, 80)
chemistry_score = st.slider("Chemistry Score", 0, 100, 80)
biology_score = st.slider("Biology Score", 0, 100, 80)
english_score = st.slider("English Score", 0, 100, 85)
history_score = st.slider("History / Social Studies Score", 0, 100, 82)
mapeh_score = st.slider("MAPEH Score", 0, 100, 88)

st.subheader("🧠 Study Habits & Behavior")

weekly_self_study_hours = st.slider("Weekly Self-Study Hours", 0, 40, 8)
absence_days = st.slider("Number of Absence Days this year", 0, 40, 3)
extracurricular_activities = st.slider("Number of Active Extracurricular Activities", 0, 10, 2)

st.subheader("🎯 Aspirations & Non-Cognitive Factors")

career_aspiration = st.text_input(
    "Dream Career (e.g., Doctor, Engineer, Lawyer, Entrepreneur, Artist)",
    value="Engineer",
)

aspiration_strength = st.slider(
    "How strong is this aspiration? (1 = weak, 5 = very strong)",
    1,
    5,
    4,
)

grit_score = st.slider(
    "Grit / Perseverance (self-assessment, 1–5)",
    1,
    5,
    4,
)

learning_style = st.selectbox(
    "Preferred Learning Style",
    options=["Visual", "Auditory", "Reading", "Kinesthetic"],
)

preferred_work_type = st.selectbox(
    "Preferred Type of Work",
    options=[
        "Analytical",
        "Creative_Communication",
        "Business_Planning",
        "Practical_Tasks",
        "Leadership_Organizing",
    ],
)

# --- Aptitude Quiz ---

st.subheader("🧪 Aptitude Quiz (9 Questions)")

st.markdown("**Numerical Reasoning**")

q1 = st.radio(
    "Q1. A class has 60 students. 15% of them are absent. How many students are absent?",
    options=["Choose an answer", "6", "9", "12"],
    index=0,
)
q2 = st.radio(
    "Q2. Solve for x: 3x + 5 = 17",
    options=["Choose an answer", "3", "4", "5"],
    index=0,
)
q3 = st.radio(
    "Q3. The average of 12 and 18 is:",
    options=["Choose an answer", "6", "12", "15"],
    index=0,
)

st.markdown("**Verbal Reasoning**")

q4 = st.radio(
    "Q4. Which word is closest in meaning to 'rapid'?",
    options=["Choose an answer", "Slow", "Careful", "Fast"],
    index=0,
)
q5 = st.radio(
    "Q5. Which sentence is correct?",
    options=[
        "Choose an answer",
        "The book are on the table.",
        "The books is on the table.",
        "The books are on the table.",
    ],
    index=0,
)
q6 = st.radio(
    "Q6. Read: 'The science quiz will be on Friday. Please bring a calculator and a pencil.' What can we correctly say?",
    options=[
        "Choose an answer",
        "The quiz is on Thursday.",
        "Students need to bring a calculator.",
        "The quiz is about English.",
    ],
    index=0,
)

st.markdown("**Scientific Reasoning**")

q7 = st.radio(
    "Q7. Water usually boils at:",
    options=["Choose an answer", "0°C", "50°C", "100°C"],
    index=0,
)
q8 = st.radio(
    "Q8. Which of these is a renewable source of energy?",
    options=["Choose an answer", "Coal", "Wind", "Oil"],
    index=0,
)
q9 = st.radio(
    "Q9. At noon, when the sun is high in the sky, your shadow is usually:",
    options=[
        "Choose an answer",
        "Longer than in the morning",
        "Shorter than in the morning",
        "The same all day",
    ],
    index=0,
)

# --- Scoring ---
num_correct_num = 0
num_correct_verbal = 0
num_correct_science = 0

# Numerical correct answers
if q1 == "9":
    num_correct_num += 1
if q2 == "4":
    num_correct_num += 1
if q3 == "15":
    num_correct_num += 1

# Verbal correct answers
if q4 == "Fast":
    num_correct_verbal += 1
if q5 == "The books are on the table.":
    num_correct_verbal += 1
if q6 == "Students need to bring a calculator.":
    num_correct_verbal += 1

# Scientific correct answers
if q7 == "100°C":
    num_correct_science += 1
if q8 == "Wind":
    num_correct_science += 1
if q9 == "Shorter than in the morning":
    num_correct_science += 1

apt_numerical_reasoning = (num_correct_num / 3) * 100
apt_verbal_reasoning = (num_correct_verbal / 3) * 100
apt_scientific_reasoning = (num_correct_science / 3) * 100

st.caption(
    f"Numerical score: {apt_numerical_reasoning:.0f} / 100 • "
    f"Verbal score: {apt_verbal_reasoning:.0f} / 100 • "
    f"Scientific score: {apt_scientific_reasoning:.0f} / 100"
)


# --- School context & skills ---

st.subheader("🏫 School Context & Skills")

col1, col2 = st.columns(2)
with col1:
    shs_stem_available = st.checkbox("STEM Track available in your area?", value=True)
    shs_abm_available = st.checkbox("ABM Track available?", value=True)
with col2:
    shs_humss_available = st.checkbox("HUMSS Track available?", value=True)
    shs_tvl_available = st.checkbox("TVL Track available?", value=True)

ict_skill_level = st.selectbox(
    "Basic ICT / Computer Skill Level",
    ["Low", "Medium", "High"],
    index=1,
)

ict_skill_map = {"Low": 0, "Medium": 1, "High": 2}
ict_skill = ict_skill_map[ict_skill_level]

# --- Derived indices ---

stem_index = (math_score + physics_score + chemistry_score + biology_score) / 4
humss_index = (english_score + history_score) / 2
arts_index = (mapeh_score + english_score) / 2
abm_index = (math_score + english_score) / 2

# Build model input row
input_data = {
    "region_code": region_code,
    "absence_days": absence_days,
    "extracurricular_activities": extracurricular_activities,
    "weekly_self_study_hours": weekly_self_study_hours,
    "math_score": math_score,
    "history_score": history_score,
    "physics_score": physics_score,
    "chemistry_score": chemistry_score,
    "biology_score": biology_score,
    "english_score": english_score,
    "mapeh_score": mapeh_score,
    "stem_index": stem_index,
    "humss_index": humss_index,
    "arts_index": arts_index,
    "abm_index": abm_index,
    "apt_numerical_reasoning": apt_numerical_reasoning,
    "apt_verbal_reasoning": apt_verbal_reasoning,
    "apt_scientific_reasoning": apt_scientific_reasoning,
    "grit_score": grit_score,
    "aspiration_strength": aspiration_strength,
    "shs_stem_available": int(shs_stem_available),
    "shs_abm_available": int(shs_abm_available),
    "shs_humss_available": int(shs_humss_available),
    "shs_tvl_available": int(shs_tvl_available),
    "ict_skill": ict_skill,
    "gender": gender,
    "varsity": varsity,
    "career_aspiration": career_aspiration,
    "learning_style": learning_style,
    "preferred_work_type": preferred_work_type,
}

X_input = pd.DataFrame(
    [{k: input_data[k] for k in (NUMERIC_FEATURES + CATEGORICAL_FEATURES)}]
)

st.markdown("---")

if st.button("🔮 Predict Strand, Track & Careers"):
    # --- 1. Rule-based check for Sports / Arts (EXTRA ONLY) ---
    is_varsity = (varsity == "Yes")
    sports_candidate = is_varsity and (mapeh_score >= 85)

    arts_candidate = (
        arts_index >= 85
        or (mapeh_score >= 85 and preferred_work_type == "Creative_Communication")
    )

    extra_rule_strand = None
    extra_rule_reason = None

    if sports_candidate:
        extra_rule_strand = "Sports"
        extra_rule_reason = (
            "You are in the varsity and have high MAPEH, which are strong signals "
            "for the **Sports** track."
        )
    elif arts_candidate:
        extra_rule_strand = "Arts"
        extra_rule_reason = (
            "Your MAPEH/Arts-related scores and creative work preferences suggest "
            "a strong fit for the **Arts & Design** track."
        )

    # --- 2. ML prediction on the 5 main strands ---
    proba = pipeline.predict_proba(X_input)[0]
    classes = pipeline.named_steps["model"].classes_

    sorted_idx = np.argsort(proba)[::-1]

    st.subheader("Recommended Strands (Machine Learning – Top 3)")
    for i in range(min(3, len(classes))):
        strand_i = classes[sorted_idx[i]]
        p = proba[sorted_idx[i]] * 100
        st.write(f"**{i+1}. {strand_i}** – {p:.1f}%")

    best_ml_strand = classes[sorted_idx[0]]

    # 🔹 MAIN recommendation is ALWAYS the ML strand
    main_strand = best_ml_strand
    st.success(f"🎓 Main Suggested SHS Strand (ML): **{best_ml_strand}**")

    # 🔹 EXTRA: show Sports/Arts suggestion if triggered
    if extra_rule_strand is not None:
        st.info(
            f"🏅 **Extra Option to Consider:** {extra_rule_strand}\n\n"
            f"{extra_rule_reason}\n\n"
            f"💡 *This is based on specific indicators from your responses that align with "
            f"the strengths typically associated with the {extra_rule_strand} track.*",
            icon="⭐",
        )


    # Track, cluster, careers are based on the ML main strand
    main_track = strand_to_track(main_strand)
    main_cluster = strand_to_course_cluster(main_strand)
    careers = strand_to_careers(main_strand)

    st.markdown("#### 🏫 Track & Course Cluster")
    st.write(f"- **SHS Track:** {main_track}")
    st.write(f"- **Course Cluster:** {main_cluster}")

    st.markdown("#### 💡 Possible Career Paths")
    for c in careers:
        st.write(f"- {c}")

    st.caption(
        "These are recommendations based on your answers. The main suggestion comes from "
        "a trained model on ABM/HUMSS/STEM/GAS/TVL, with optional extra hints for "
        "Sports or Arts. Final choices should still consider your interests, family "
        "situation, and guidance from teachers or counselors."
    )

    # --- Aptitude Feedback & Explanations (unchanged) ---
    st.markdown("### 📘 Aptitude Quiz Feedback")

    st.write(
        f"Numerical score: **{apt_numerical_reasoning:.0f}/100**  •  "
        f"Verbal score: **{apt_verbal_reasoning:.0f}/100**  •  "
        f"Scientific score: **{apt_scientific_reasoning:.0f}/100**"
    )

    explanations = [
        {
            "label": "Q1 (Numerical)",
            "question": "A class has 60 students. 15% of them are absent. How many students are absent?",
            "your_answer": q1,
            "correct_answer": "9",
            "explanation": "15% of 60 = 0.15 × 60 = 9.",
        },
        {
            "label": "Q2 (Numerical)",
            "question": "Solve for x: 3x + 5 = 17.",
            "your_answer": q2,
            "correct_answer": "4",
            "explanation": "3x + 5 = 17 → 3x = 12 → x = 4.",
        },
        {
            "label": "Q3 (Numerical)",
            "question": "The average of 12 and 18 is:",
            "your_answer": q3,
            "correct_answer": "15",
            "explanation": "(12 + 18) ÷ 2 = 30 ÷ 2 = 15.",
        },
        {
            "label": "Q4 (Verbal)",
            "question": "Which word is closest in meaning to 'rapid'?",
            "your_answer": q4,
            "correct_answer": "Fast",
            "explanation": "'Rapid' means quick or fast.",
        },
        {
            "label": "Q5 (Verbal)",
            "question": "Which sentence is correct?",
            "your_answer": q5,
            "correct_answer": "The books are on the table.",
            "explanation": "'Books' is plural, so the verb must be 'are'.",
        },
        {
            "label": "Q6 (Verbal)",
            "question": "Based on the note about the science quiz on Friday:",
            "your_answer": q6,
            "correct_answer": "Students need to bring a calculator.",
            "explanation": "The note clearly says to bring a calculator and a pencil.",
        },
        {
            "label": "Q7 (Scientific)",
            "question": "Water usually boils at:",
            "your_answer": q7,
            "correct_answer": "100°C",
            "explanation": "At normal air pressure, water boils at 100°C.",
        },
        {
            "label": "Q8 (Scientific)",
            "question": "Which of these is a renewable source of energy?",
            "your_answer": q8,
            "correct_answer": "Wind",
            "explanation": "Wind is naturally renewed and does not run out.",
        },
        {
            "label": "Q9 (Scientific)",
            "question": "At noon, when the sun is high in the sky, your shadow is usually:",
            "your_answer": q9,
            "correct_answer": "Shorter than in the morning",
            "explanation": "When the sun is overhead, shadows become shorter.",
        },
    ]

    for item in explanations:
        is_correct = (item["your_answer"] == item["correct_answer"])
        icon = "✅" if is_correct else "❌"

        with st.expander(f"{icon} {item['label']}"):
            st.write(f"**Question:** {item['question']}")
            st.write(f"**Your answer:** {item['your_answer']}")
            st.write(f"**Correct answer:** {item['correct_answer']}")
            st.write(f"**Why?** {item['explanation']}")



