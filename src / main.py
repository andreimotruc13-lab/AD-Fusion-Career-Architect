import sys
import types

# Creating a fake 'main' module in memory
if 'main' not in sys.modules:
    main_module = types.ModuleType('main')
    sys.modules['main'] = main_module
else:
    main_module = sys.modules['main']

try:
    from sklearn.compose._column_transformer import _RemainderColsList
    main_module._RemainderColsList = _RemainderColsList
except ImportError:
    class _RemainderColsList(list):
        pass
    main_module._RemainderColsList = _RemainderColsList

import streamlit as st
import pandas as pd
import os
import io
import joblib
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import requests

# detectam calea catre folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Loading models securizat prin descărcare directă
@st.cache_resource
def load_ml_models():
    import sys
    import types
    
    try:
        from sklearn.compose._column_transformer import _RemainderColsList
    except ImportError:
        class _RemainderColsList(list):
            pass

    for mod_name in ['main', '__main__', __name__]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)
        setattr(sys.modules[mod_name], '_RemainderColsList', _RemainderColsList)

    if os.path.exists(os.path.join(BASE_DIR, "models")):
        MODELS_DIR = os.path.join(BASE_DIR, "models")
    else:
        MODELS_DIR = os.path.join(os.path.dirname(BASE_DIR), "models")
    os.makedirs(MODELS_DIR, exist_ok=True)
        
    rec_fixed_path = os.path.join(MODELS_DIR, "recommender_fixed.pkl")
    prep_fixed_path = os.path.join(MODELS_DIR, "preprocessor_fixed.pkl")
    tfidf_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    rec_text_path = os.path.join(MODELS_DIR, "recommender_text.pkl") 
    
    # FIX: Removed '/src/' from paths and matched the exact filenames on Hugging Face
    urls = {
        "recommender_fixed.pkl": "https://huggingface.co/spaces/AndrIIII7/career-architect/resolve/main/recommender_fixed.pkl", 
        "preprocessor_fixed.pkl": "https://huggingface.co/spaces/AndrIIII7/career-architect/resolve/main/preprocessor_fixed.pkl",
        "tfidf_vectorizer.pkl": "https://huggingface.co/spaces/AndrIIII7/career-architect/resolve/main/tfidf_vectorizer.pkl",
        "recommender_text.pkl": "https://huggingface.co/spaces/AndrIIII7/career-architect/resolve/main/recommender_text.pkl"
    }
    
    try:
        # Loop through models to download them safely and avoid repetitive blocks
        paths_map = {
            "recommender_fixed.pkl": rec_fixed_path,
            "preprocessor_fixed.pkl": prep_fixed_path,
            "tfidf_vectorizer.pkl": tfidf_path,
            "recommender_text.pkl": rec_text_path
        }

        for file_name, file_path in paths_map.items():
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                r = requests.get(urls[file_name], stream=True)
                if r.status_code == 200:
                    with open(file_path, 'wb') as f:
                        f.write(r.content)
                else:
                    # Clearing error visibility instead of failing silently
                    return None, None, None, None, f"Failed to download {file_name}. Status code: {r.status_code}"
        
        # Loading into memory
        recommender = joblib.load(rec_fixed_path)
        preprocessor = joblib.load(prep_fixed_path)
        tfidf_vec = joblib.load(tfidf_path)
        rec_text = joblib.load(rec_text_path)
        
        return recommender, preprocessor, tfidf_vec, rec_text, True
    except Exception as e:
        return None, None, None, None, f"Eroare la incarcare modele: {str(e)}"

recommender, preprocessor, tfidf_vectorizer, recommender_text, ml_status = load_ml_models()
ml_loaded = (ml_status is True)

if not ml_loaded:
    st.sidebar.error(f"Modelele ML nu s-au încărcat: {ml_status}")

# Configuration and styling
st.set_page_config(page_title="Candidate Data Manager", page_icon="📋", layout="wide")

st.markdown("""
    <style>
    .st-emotion-cache-1v0mbdj > img { border-radius: 10px; }
    .nested-header { font-weight: 600; color: #4F8BF9; margin-top: 10px; margin-bottom: 5px; }
    </style>
""", unsafe_allow_html=True)

# Constants and data mapping
MAIN_CSV = "https://docs.google.com/uc?export=download&id=1quji8xUbgNZYzZcv6vQJgcNaM7qZc0S9"
SUB_CSV = "user_submissions.csv"

CITIES = ['Balti', 'Cahul', 'Chisinau', 'Comrat', 'Edinet', 'Orhei', 'Soroca', 'Stauceni', 'Tiraspol', 'Ungheni']

SKILLS_MAP = {
    "Tech & Development": {
        "Backend": "Tech_Backend", "Frontend": "Tech_Frontend",
        "DevOps/Cloud": "Tech_DevOps_Cloud", "Data & AI": "Tech_Data_AI"
    },
    "Business & Management": {
        "Project/Agile": "Mgmt_Project_Agile", "Strategic Leadership": "Mgmt_Strategic_Leadership"
    },
    "Finance & Legal": {
        "Accounting/Audit": "Fin_Accounting_Audit", "Analysis/Investment": "Fin_Analysis_Investment",
        "Legal Services": "Legal_Services"
    },
    "Marketing & Creative": {
        "Digital/SEO": "Mkt_Digital_SEO", "Design/Art": "Creative_Design_Art"
    },
    "Health & Science": {
        "Health/Medical": "Health_Medical", "Science/Engineering": "Science_Engineering"
    },
    "Services & Niche": {
        "Events": "Service_Events", "Specialized": "Service_Specialized"
    }
}

BENEFITS = [
    "Bonuses and Incentive Programs", "Casual Dress Code", "Childcare Assistance",
    "Employee Assistance Programs (EAP)", "Employee Discounts", "Employee Recognition Programs",
    "Employee Referral Programs", "Financial Counseling", "Flexible Spending Accounts (FSAs)",
    "Flexible Work Arrangements", "Health Insurance", "Health and Wellness Facilities",
    "Legal Assistance", "Life and Disability Insurance", "Paid Time Off (PTO)",
    "Parental Leave", "Professional Development", "Profit-Sharing", "Relocation Assistance",
    "Retirement Plans", "Social and Recreational Activities", "Stock Options or Equity Grants",
    "Transportation Benefits", "Tuition Reimbursement", "Wellness Programs"
]

EXP_RANGES = {
    "0-2 years": 1.0, "2-4 years": 3.0, "4-6 years": 5.0, "6-8 years": 7.0,
    "8-10 years": 9.0, "10-12 years": 11.0, "12-15 years": 13.5, "15+ years": 17.0
}

WORK_TYPES = {
    'Intern': 1,
    'Part-Time': 2,
    'Temporary': 3,
    'Contract': 4,
    'Full-Time': 5
}

QUALIFICATIONS = {
    'UG General': 1,
    'UG Technical': 2,
    'PG Commerce/Management': 3,
    'PG Technical': 4,
    'Doctorate': 5
}

LENGTHS = ['Short And Concise', 'Medium', 'Long And Detailed']

tech_cols = ['Tech_Backend', 'Tech_Frontend', 'Tech_DevOps_Cloud',
            'Tech_Data_AI', 'Mgmt_Project_Agile', 'Mgmt_Strategic_Leadership',
            'Fin_Accounting_Audit', 'Fin_Analysis_Investment', 'Legal_Services',
            'Mkt_Digital_SEO', 'Creative_Design_Art', 'Health_Medical',
            'Science_Engineering', 'Service_Events', 'Service_Specialized']

benefit_cols = ['Female', 'Male', 'Cahul', 'Chisinau',
                 'Comrat', 'Edinet', 'Orhei', 'Soroca', 'Stauceni', 'Tiraspol',
                 'Ungheni', 'Bonuses and Incentive Programs', 'Casual Dress Code',
                 'Childcare Assistance', 'Employee Assistance Programs (EAP)',
                 'Employee Discounts', 'Employee Recognition Programs', 'Employee Referral Programs',
                 'Financial Counseling', 'Flexible Spending Accounts (FSAs)', 'Flexible Work Arrangements',
                 'Health Insurance', 'Health and Wellness Facilities', 'Legal Assistance', 'Life and Disability Insurance',
                 'Paid Time Off (PTO)', 'Parental Leave', 'Professional Development', 'Profit-Sharing',
                 'Relocation Assistance', 'Retirement Plans', 'Social and Recreational Activities',
                 'Stock Options or Equity Grants', 'Transportation Benefits', 'Tuition Reimbursement', 'Wellness Programs']

required_cols = ['Qualifications', 'Work Type', 'Min_experience', 'Max_experience',
                 'Salary', 'Female', 'Male', 'Cahul', 'Chisinau', 'Comrat', 'Edinet',
                 'Orhei', 'Soroca', 'Stauceni', 'Tiraspol', 'Ungheni',
                 'Bonuses and Incentive Programs', 'Casual Dress Code', 'Childcare Assistance',
                 'Employee Assistance Programs (EAP)', 'Employee Discounts', 'Employee Recognition Programs',
                 'Employee Referral Programs', 'Financial Counseling', 'Flexible Spending Accounts (FSAs)',
                 'Flexible Work Arrangements', 'Health Insurance', 'Health and Wellness Facilities',
                 'Legal Assistance', 'Life and Disability Insurance', 'Paid Time Off (PTO)',
                 'Parental Leave', 'Professional Development', 'Profit-Sharing', 'Relocation Assistance',
                 'Retirement Plans', 'Social and Recreational Activities', 'Stock Options or Equity Grants',
                 'Transportation Benefits', 'Tuition Reimbursement', 'Wellness Programs', 'Tech_Backend', 'Tech_Frontend',
                 'Tech_DevOps_Cloud', 'Tech_Data_AI', 'Mgmt_Project_Agile', 'Mgmt_Strategic_Leadership',
                 'Fin_Accounting_Audit', 'Fin_Analysis_Investment', 'Legal_Services', 'Mkt_Digital_SEO',
                 'Creative_Design_Art', 'Health_Medical', 'Science_Engineering',
                 'Service_Events', 'Service_Specialized', 'Soft_Skills_Score', 'Cluster', 'Archetype']

def get_all_columns():
    cols = ["Female", "Male"] + [city for city in CITIES if city != "Balti"] + \
           ["Qualifications", "Work Type", "Min_experience", "Max_experience", "Soft_Skills_Score"]
    for group in SKILLS_MAP.values():
        cols.extend(group.values())
    cols.extend(BENEFITS)
    return cols

def init_files():
    cols = get_all_columns()
    if not os.path.exists(SUB_CSV):
        pd.DataFrame(columns=cols).to_csv(SUB_CSV, index=False)

init_files()

# Sidebar Layout
st.sidebar.markdown(
    """Note: Job locations and salaries are deterministic simulations,
    adapting a global dataset to reflect the Moldovan market context for demonstration purposes."""
)

st.sidebar.title("About A&D Fusion")
st.sidebar.markdown(
    """We are A&D Fusion, a team dedicated to building a better future for everyone.
    Through this platform, we offer you the opportunity to build a career,
    whether you are just starting out or already have extensive experience in the field.
    Our goal is to reduce unemployment and emigration in the Republic of Moldova by giving every
    person the chance to contribute creatively to the development of society and to create a better life
    for all its residents. Now is your moment. Get employed and move closer to achieving your dream."""
)

# Main app tabs
st.title("📋 Candidate Data Manager")
tab1, tab2, tab3 = st.tabs(["📝 Form Entry", "📊 Dataset Viewer", "🤖 AI Recommender"])

# Tab 1: Candidate form
with tab1:
    st.header("Candidate Information Form")
   
    with st.form("candidate_form", clear_on_submit=True):
        st.subheader("1. Personal Information")
        col1, col2 = st.columns(2)
        with col1:
            full_name = st.text_input("Full Name")
            age = st.number_input("Age", min_value=18, max_value=100, value=25)
        with col2:
            gender = st.radio("Gender", ["Female", "Male", "Other / Prefer not to say"], horizontal=True)
            city = st.selectbox("City", CITIES)
            city_selected = city
           
        st.divider()
       
        st.subheader("2. Qualifications & Work Type")
        col3, col4 = st.columns(2)
        with col3:
            qualifications = st.selectbox("Qualifications Score", options = list(QUALIFICATIONS.keys()))
        with col4:
            work_type = st.selectbox("Work Type", options = list(WORK_TYPES.keys()))
           
        st.divider()

        st.subheader("3. Experience")
        experience = st.select_slider("Years of Experience", options=list(EXP_RANGES.keys()))
       
        st.divider()

        st.subheader("4. Skills & Domains")
        st.caption("Check all applicable skills")
        skills_state = {}
       
        exp_col1, exp_col2 = st.columns(2)
        exp_cols = [exp_col1, exp_col2]
       
        for idx, (domain, subskills) in enumerate(SKILLS_MAP.items()):
            with exp_cols[idx % 2].expander(f"📁 {domain}"):
                for label, col_name in subskills.items():
                    with st.container(border=True):
                        st.markdown(f"<div class='nested-header'>🔹 {label}</div>", unsafe_allow_html=True)
                        skills_state[col_name] = st.checkbox(f"Add {label} skill", key=col_name)
       
        st.markdown("<br>", unsafe_allow_html=True)
        soft_skills = st.slider("Soft Skills Score", min_value=0, max_value=3, value=1)

        st.divider()

        st.subheader("5. Benefits & Perks")
        benefits_state = {}
        b_cols = st.columns(3)
        for i, benefit in enumerate(BENEFITS):
            with b_cols[i % 3]:
                benefits_state[benefit] = st.checkbox(benefit)

        st.subheader("6. Recommendation Settings")
        num_matches = st.slider("Number of jobs to recommend", min_value=1, max_value=10, value=5)

        st.divider()
        submitted = st.form_submit_button("💾 Submit Profile", use_container_width=True)
       
        if submitted:
            if not full_name.strip():
                st.error("Please provide a Full Name.")
            else:
                row = {c: 0 for c in get_all_columns()}
                if gender == "Female": row["Female"] = 1
                if gender == "Male": row["Male"] = 1
                if city in CITIES: row[city] = 1
               
                row["Qualifications"] = QUALIFICATIONS[qualifications]
                row["Work Type"] = WORK_TYPES[work_type]
               
                midpoint = EXP_RANGES[experience]
                row["Min_experience"] = midpoint
                row["Max_experience"] = midpoint
                row["Soft_Skills_Score"] = soft_skills
               
                for col_name, checked in skills_state.items():
                    row[col_name] = 1 if checked else 0
                for benefit, checked in benefits_state.items():
                    row[benefit] = 1 if checked else 0
                   
                new_df = pd.DataFrame([row])
                if 'Full name' in new_df:
                    cols_to_drop = ['Balti', 'Full name', 'Age']
                    new_df.drop(columns=[c for c in cols_to_drop if c in new_df.columns], inplace=True)

                existing_df = pd.read_csv(SUB_CSV)
                updated_df = pd.concat([existing_df, new_df], ignore_index=True)
                updated_df.to_csv(SUB_CSV, index=False)
                   
                st.success(f"✅ Profile for {full_name} submitted successfully!")

                if ml_loaded:
                    st.divider()
                    st.subheader("🎯 Your Recommended Jobs")
           
                    with st.spinner("Analyzing profile and finding the best matches..."):
                        try:
                            df_temp = new_df.copy()

                            for col in required_cols:
                                if col not in df_temp.columns:
                                    df_temp[col] = 0

                            df_temp = df_temp[required_cols]

                            for col in tech_cols:
                                df_temp[col] = df_temp[col] * 20.0

                            df_temp['Qualifications'] = df_temp['Qualifications'] * 5

                            for col in benefit_cols:
                                df_temp[col] = df_temp[col] * 0.5

                            X_new_processed = preprocessor.transform(df_temp)

                            df_jobs = None
                            try:
                                df_jobs = pd.read_csv(MAIN_CSV)
                            except Exception:
                                df_jobs = pd.DataFrame()

                            total_possible = len(df_jobs)

                            if total_possible == 0:
                                st.error(f"⚠️ The dataset is currently empty. Direct download failed.")
                            else:
                                search_k = min(300, total_possible)

                                distances, indices = recommender.kneighbors(X_new_processed, n_neighbors=search_k)

                                recommendations_pool = df_jobs.iloc[indices[0]].copy()
                                recommendations_pool['distance'] = distances[0]

                                results = recommendations_pool[recommendations_pool['City'] == city_selected].head(num_matches)

                                for i, (idx, row) in enumerate(results.iterrows()):
                                    raw_dist = row['distance']
                                    
                                    strictness = 1.5
                                    realistic_score = np.exp(-strictness * raw_dist) * 100
                                    if realistic_score > 99.9: realistic_score = 99.9
                                    
                                    st.info(
                                        f"**[{i+1}] {row.get('Job Title', 'Unknown Job')}** \n"
                                        f"🏢 **Company:** {row.get('Company', 'N/A')} | 💼 **Role:** {row.get('Role', 'N/A')}  \n"
                                        f"📞 **Contact:** {row.get('Contact', 'N/A')} | 🏢 **City:** {row.get('City', 'Unknown city')}  \n"
                                        f"💵 **Salary:** {row.get('Salary Range', 'N/A')}  \n"
                                        f"💭 **Job Description:** {row.get('Job Description', 'N/A')}  \n"
                                        f"➕ **Benefits:** {row.get('Benefits', 'N/A')}  \n"
                                        f"🔥 **Match Strength:** `{realistic_score:.1f}%`"
                                    )
                        except Exception as e:
                            st.error(f"⚠️ Could not generate recommendations. Error: {e}")
                else:
                    st.warning(f"⚠️ Model Status: {ml_status}")


# Tab 2: Dataset viewer
with tab2:
    st.header("Dataset Viewer")
   
    try:
        df_main = pd.read_csv(MAIN_CSV)
    except Exception as e:
        df_main = pd.DataFrame(columns=get_all_columns())
       
    search_query = st.text_input("🔍 Search across all columns (case-insensitive)", placeholder="Type a name, city, skill, etc...")
   
    if search_query:
        mask = df_main.astype(str).apply(lambda x: x.str.contains(search_query, case=False, na=False)).any(axis=1)
        filtered_df = df_main[mask]
    else:
        filtered_df = df_main

    colA, colB = st.columns([1, 4])
    with colA:
        st.metric("Total Rows", len(filtered_df))
    with colB:
        csv_buffer = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Filtered CSV",
            data=csv_buffer,
            file_name="filtered_candidates.csv",
            mime="text/csv",
        )

    st.dataframe(filtered_df, use_container_width=True, hide_index=True)


# Tab 3: AI recommender chatbot
with tab3:
    st.header("🤖 A&D Fusion Career Architect")
    st.markdown("Paste your CV below, and our AI will analyze it against our curated Moldova job dataset to suggest high-impact profile improvements.")
   
    target_city = st.selectbox("📍 Target City for Localized Advice:", CITIES)
    length_of_response = st.selectbox("Preferred length of response:", LENGTHS)
   
    cv_text = st.text_area("📄 Paste your CV here:", height=250, placeholder="Experience: \n- Worked at X for 3 years...\n\nSkills: \n- Python, Excel, Management...")
    analyze_btn = st.button("🧠 Analyze CV", type="primary")

    if analyze_btn:
        if not cv_text.strip():
            st.warning("Please paste your CV text to analyze.")
        elif not ml_loaded:
            st.error("Text processing models are not loaded. Please check your Hugging Face connections.")
        else:
            with st.spinner("Analyzing CV against market data..."):
                try:
                    df_jobs = None
                    try:
                        df_jobs = pd.read_csv(MAIN_CSV)
                    except Exception as e:
                        st.error(f"Could not load main dataset. Error: {e}")
                        df_jobs = pd.DataFrame()

                    if df_jobs is not None and not df_jobs.empty:
                        # Filling missing text identically to training script to prevent transform errors
                        df_jobs['skills'] = df_jobs.get('skills', pd.Series(dtype=str)).fillna('')
                        df_jobs['Job Title'] = df_jobs.get('Job Title', pd.Series(dtype=str)).fillna('')
                        df_jobs['Job Description'] = df_jobs.get('Job Description', pd.Series(dtype=str)).fillna('')
                        df_jobs['Responsibilities'] = df_jobs.get('Responsibilities', pd.Series(dtype=str)).fillna('')
                        
                        # Vectorizing the incoming CV string using the tfidf vectorizer
                        cv_vector = tfidf_vectorizer.transform([cv_text]).toarray()
                        
                        # Finding the closest matches
                        search_k = min(15, len(df_jobs))
                        distances, indices = recommender_text.kneighbors(cv_vector, n_neighbors=search_k)
                        
                        # Isolating for the API feeding
                        top_matches_df = df_jobs.iloc[indices[0]].copy()
                        sample_data = top_matches_df.to_csv(index=False, sep='|')
                    else:
                        sample_data = "DATA NOT FOUND"

                    if "OPENROUTER_API_KEY" in st.secrets:
                        active_key = st.secrets["OPENROUTER_API_KEY"]
                    else:
                        st.error("🔑 API Key is completely missing from your Streamlit Secrets Panel!")
                        st.stop()

                    from openai import OpenAI
                    client = OpenAI(
                        base_url="https://openrouter.ai/api/v1",
                        timeout=500.0,
                        api_key=active_key,
                        default_headers={
                            "HTTP-Referer": "http://localhost:8501",
                            "X-Title": "A&D Fusion App",
                        }
                    )

                    system_instructions = f"""
                    You are the "A&D Fusion Career Architect." You are working with a curated dataset of the most relevant job market entries in Moldova matching the user's career path.

                    --- START OF RELEVANT RAG DATA ---
                    {sample_data}
                    --- END OF RELEVANT RAG DATA ---

                    1. PRECISION: Treat these matching rows as the definitive benchmarks for 'Salary Range' and 'Skills'.
                    2. CV UPGRADE: If a user provides a CV, perform a "Deep Audit":
                       - Find the closest 2-5 roles in the provided data depending on the closeness and length of response.
                       - After finding those closest roles please show/recommend them to the sender so they know what exactly they are improving their CV for.
                       - Use the 'skills' and 'Experience' from those roles to suggest 2 or more (depending on the CV) high-impact additions to the user's CV.
                       - Rewrite a section of their CV using the STAR method (Action -> Result).
                    3. THE CITY CONTEXT: The user is looking for roles in {target_city}. Use the 'City' data to ensure the advice is localized to this specific market.
                    4. LIMITATIONS: If the user's career path isn't perfectly represented in these rows, honestly state: "Based on our current market subset, we don't have a direct match, but here is the closest strategic advice."
                    5. LENGTH OF OUTPUT: Make sure that the length of what you give the user is: {length_of_response}
                    6. IMPORTANT: Keep in mind that the user will only be able to input something once, so do not ask follow-up questions and make sure to clarify all possible questions in 1 message.
                    """

                    response = client.chat.completions.create(
                        model="openrouter/owl-alpha",
                        messages=[
                            {"role": "system", "content": system_instructions},
                            {"role": "user", "content": f"Here is my CV:\n{cv_text}"}
                        ],
                        temperature=0.4
                    )

                    if response and response.choices:
                        result_text = response.choices[0].message.content
                        st.success("CV Audit Complete!")
                        st.markdown(
                            f"""
                            <div style="
                            background-color: #262730;
                            color: #ffffff;
                            padding: 25px;
                            border-radius: 10px;
                            border: 1px solid #464b5d;
                            line-height: 1.6;
                            ">
                            {result_text}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.error("The API returned nothing. Check your token count or connection.")

                except Exception as e:
                    st.error(f"API Error: {e}")
