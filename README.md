Resume Screening Tool 📄
This project is an end-to-end application designed to automate the initial resume screening process. It uses a hybrid AI approach to analyze a candidate's resume against a given job description, providing a detailed and interpretable match score. The entire application is containerized with Docker for easy deployment and portability.

<img width="1920" height="870" alt="image" src="https://github.com/user-attachments/assets/f654fcc5-bf77-4fe4-b30f-f6b1c37dbf35" />


## Features
Hybrid Scoring System: Combines a hard skills check with a contextual experience match for a more accurate and reliable analysis.

Keyword Extraction: Automatically identifies key skills required by the job description and checks for their presence in the resume.

Semantic Search: Uses a powerful Sentence-BERT model (all-mpnet-base-v2) to understand the semantic meaning and context of the candidate's work experience, not just keywords.

Support for Multiple File Types: Can process resumes from both .pdf and .txt files.

Interactive Web Interface: A user-friendly UI built with Streamlit allows for easy file uploads and clear visualization of the results.

Containerized for Portability: The entire application is packaged with Docker, allowing it to run consistently in any environment.

## Tech Stack
Backend & ML Engine: Python, Sentence-Transformers (SBERT), scikit-learn, pdfplumber

Frontend: Streamlit

Deployment: Docker

## Project Structure
resume-screening-tool/
├── .gitignore
├── app.py
├── Dockerfile
├── process_module.py
├── README.md
├── requirements.txt
├── demo_files/
│   └── positive_example2.pdf
└── uploads/
    └── .gitkeep
## Setup and Installation
Follow these steps to set up the project locally.

Clone the repository:

Bash

git clone https://github.com/your-username/resume-screening-tool.git
Navigate to the project directory:

Bash

cd resume-screening-tool
Create and activate a virtual environment (recommended):

Bash

# Using conda
conda create --name resume-screener python=3.9
conda activate resume-screener
Install the required dependencies:

Bash

pip install -r requirements.txt
## How to Run
You can run the application either locally using Streamlit or as a Docker container.

### Running Locally
Make sure your virtual environment is activated.

Run the Streamlit application from the root directory:

Bash

streamlit run app.py
Open your web browser and navigate to http://localhost:8501.

### Running with Docker
Make sure you have Docker Desktop installed and running.

Build the Docker image:

Bash

docker build -t resume-screener .
Run the Docker container:

Bash

docker run -p 8501:8501 resume-screener
Open your web browser and navigate to http://localhost:8501.

## Future Improvements
Batch Processing: Add functionality to upload and analyze multiple resumes at once.

Automatic Skill Extraction: Enhance the get_skills_from_jd function to use NLP (e.g., spaCy's EntityRuler) to automatically identify required skills instead of relying on a predefined list.

Cloud Deployment: Deploy the containerized application to a cloud service like Hugging Face Spaces, AWS, or GCP for live public access.
