# app.py

import streamlit as st
import os
from process_module import extract_text_from_file, analyze_resume

# --- App Configuration ---
st.set_page_config(
    page_title="Resume Screening Tool",
    page_icon="📄",
    layout="wide"
)

# --- Main App Interface ---
st.title("📄 Resume Screening Tool")

st.markdown("""
This tool analyzes a resume against a job description to determine the candidate's fit.
It uses a hybrid approach, checking for both required **hard skills** and the **contextual similarity** of work experience using an AI model.
""")

st.sidebar.header("Upload and Analyze")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your resume (PDF or TXT)", type=["pdf", "txt"])

# Text area for job description
job_description = st.sidebar.text_area("Paste the Job Description here", height=300)

# Analyze button
analyze_button = st.sidebar.button("Analyze Candidate")

if analyze_button:
    if uploaded_file is not None and job_description:
        with st.spinner('Analyzing... This may take a moment.'):
            # Save the uploaded file temporarily
            temp_dir = "uploads"
            os.makedirs(temp_dir, exist_ok=True)
            file_path = os.path.join(temp_dir, uploaded_file.name)

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # --- Call our ML Engine ---
            resume_text = extract_text_from_file(file_path)
            if resume_text and resume_text.strip():
                analysis_result = analyze_resume(resume_text, job_description)

                # --- Display the Report ---
                st.header("Candidate Analysis Report")

                # Display final score with a progress bar
                st.subheader(f"Overall Match Score: {analysis_result['final_score']:.2%}")
                st.progress(analysis_result['final_score'])

                # Create columns for detailed scores
                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        label=f"Required Skills Found ({len(analysis_result['matched_skills'])}/{len(analysis_result['required_skills'])})",
                        value=f"{analysis_result['skill_match_score']:.2%}"
                    )
                    # Display matched and missing skills
                    if analysis_result['matched_skills']:
                        st.success(f"**Matched:** {', '.join(analysis_result['matched_skills'])}")
                    if analysis_result['missing_skills']:
                        st.warning(f"**Missing:** {', '.join(analysis_result['missing_skills'])}")

                with col2:
                    st.metric(
                        label="Contextual Experience Match",
                        value=f"{analysis_result['experience_score']:.2%}"
                    )

                # Display verdict
                st.subheader("Verdict")
                if analysis_result['final_score'] > 0.70:
                    st.success("Strong Candidate: High match in both skills and experience. ✅")
                elif analysis_result['final_score'] > 0.55:
                    st.info("Potential Candidate: Good skill overlap, review experience. 🤔")
                else:
                    st.error("Unlikely Match: Significant skill gap. ❌")

            else:
                st.error("Could not extract text from the resume. The file might be empty or corrupted.")

            # Clean up the uploaded file
            os.remove(file_path)

    else:
        st.warning("Please upload a resume and paste a job description.")