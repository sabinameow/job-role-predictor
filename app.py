import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

model = joblib.load('models/best_model.pkl')
feature_cols = joblib.load('models/feature_columns.pkl')
top_skills = joblib.load('models/top_skills.pkl')

st.set_page_config(page_title='Job Role Predictor', layout='centered')

st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        .block-container { padding-top: 2rem; }
        .stButton>button {
            background-color: #4F8BF9;
            color: white;
            border-radius: 8px;
            padding: 0.5rem 2rem;
            font-size: 16px;
            border: none;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #3a6fd8;
        }
        .result-box {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            margin-top: 1rem;
        }
        .top-role {
            font-size: 2rem;
            font-weight: 700;
            color: #4F8BF9;
            text-align: center;
        }
        .top-prob {
            font-size: 1rem;
            color: #888;
            text-align: center;
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("# 🎯 Job Role Predictor")
st.markdown("#### Find out which tech role matches your skill set best.")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🛠 Your Skills")
    selected_skills = st.multiselect(
        'Select all skills that apply to you',
        options=[s.replace('skill_', '') for s in top_skills],
        placeholder='Start typing or scroll to select...'
    )

with col2:
    st.markdown("### 📅 Experience")
    experience = st.selectbox(
        'Your experience level',
        options=[
            'No experience',
            '1–3 years',
            '3–6 years',
            'More than 6 years'
        ]
    )

experience_map = {
    'No experience': 0,
    '1–3 years': 1,
    '3–6 years': 2,
    'More than 6 years': 3
}

st.markdown("")
predict = st.button('Predict My Role')

if predict:
    if len(selected_skills) == 0:
        st.warning('Please select at least one skill.')
    else:
        input_vector = pd.DataFrame([np.zeros(len(feature_cols))], columns=feature_cols)

        for skill in selected_skills:
            col_name = 'skill_' + skill
            if col_name in input_vector.columns:
                input_vector[col_name] = 1

        input_vector['experience'] = experience_map[experience]
        input_vector['skills_count'] = len(selected_skills)

        probabilities = model.predict_proba(input_vector)[0]
        classes = model.classes_

        result_df = pd.DataFrame({
            'Job Role': classes,
            'Probability': probabilities
        }).sort_values('Probability', ascending=False)

        top_role = result_df['Job Role'].iloc[0]
        top_prob = result_df['Probability'].iloc[0]

        st.markdown("---")
        st.markdown("### 📊 Results")

        st.markdown(f"""
            <div class="result-box">
                <div class="top-role">🏆 {top_role}</div>
                <div class="top-prob">Top match with {round(top_prob * 100, 1)}% confidence</div>
            </div>
        """, unsafe_allow_html=True)

        if top_prob < 0.4:
            st.warning('Low confidence - your skill set matches multiple roles. Try adding more specific skills.')

        st.markdown("")

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#4F8BF9' if i == 0 else '#c9d9f5' for i in range(len(result_df))]
        ax.barh(result_df['Job Role'], result_df['Probability'], color=colors)
        ax.set_xlabel('Probability')
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('#f8f9fa')
        plt.tight_layout()
        st.pyplot(fig)