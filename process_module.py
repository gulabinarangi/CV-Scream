import spacy, pdfplumber
from PIL import Image
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.lang.en.stop_words import STOP_WORDS
import os
import numpy as np
import cv2 as cv2
from paddleocr import PaddleOCR
from sentence_transformers import SentenceTransformer
from cv2 import *
import uuid
import re
#
#
# MODEL_NAME = 'all-mpnet-base-v2'
# try:
#     nlp = spacy.load("en_core_web_sm")
# except OSError:
#     print("Downloading 'en_core_web_sm' model...")
#     os.system("python -m spacy download en_core_web_sm")
#     nlp = spacy.load("en_core_web_sm")
#
#
#
# print("Loading SBERT model...")
# model = SentenceTransformer(MODEL_NAME)
# print("SBERT model loaded successfully.")
#
# def clean_text(text):
#     """Applies light cleaning to the raw text before encoding."""
#     # Remove email addresses
#     text = re.sub(r'\S+@\S+', '', text)
#     # Remove phone numbers (basic pattern)
#     text = re.sub(r'(\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}', '', text)
#     # Remove extra whitespace and newlines
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text
#
# def chunk_text(text, chunk_size=256, overlap=64):
#     """Breaks text into smaller, overlapping chunks."""
#     tokens = text.split()
#     chunks = []
#     for i in range(0, len(tokens), chunk_size - overlap):
#         chunk = " ".join(tokens[i:i + chunk_size])
#         chunks.append(chunk)
#     return chunks
#
# # def preprocess_text(text):
# #     """Lemmatize and tokenize text."""
# #     text = " ".join(text.split())
# #     doc = nlp(text.lower())
# #
# #     result = []
# #     for token in doc:
# #         # Keep the token if it's not a stop word and consists of alphabetic characters
# #         cleaned = re.sub(r"[^\w\s]", "", token.text).lower()
# #
# #         if cleaned and cleaned not in STOP_WORDS and cleaned.isalpha():
# #             result.append(nlp(cleaned)[0].lemma_)
# #
# #     return " ".join(result)
#
# def preprocess_image(img_path):
#     # Read image from path
#     img = cv2.imread(img_path)
#     if img is None:
#         raise ValueError(f"Could not read image from path: {img_path}")
#
#     # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # Apply median blur to reduce noise
#     blurred = cv2.medianBlur(gray, 3)
#
#     # Boost contrast with CLAHE
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     contrasted = clahe.apply(blurred)
#
#     # Apply adaptive thresholding
#     thresh = cv2.adaptiveThreshold(contrasted, 255,
#                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                    cv2.THRESH_BINARY, 11, 2)
#
#     # Resize if text is small
#     height, width = thresh.shape
#     if height < 800:
#         thresh = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
#
#     # Deskewing
#     coords = np.column_stack(np.where(thresh > 0))
#     angle = cv2.minAreaRect(coords)[-1]
#     angle = -(90 + angle) if angle < -45 else -angle
#     (h, w) = thresh.shape
#     M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
#     deskewed = cv2.warpAffine(thresh, M, (w, h),
#                               flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#
#     # Morphological cleaning
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
#     opened = cv2.morphologyEx(deskewed, cv2.MORPH_OPEN, kernel)
#
#     return opened
#
#
# def extract_text_from_pdf(pdf_path):
#     """extracts text from a PDF file."""
#     text = ""
#     try:
#         with pdfplumber.open(pdf_path) as pdf:
#             for page in pdf.pages:
#                 page_text = page.extract_text()
#                 if page_text:
#                     text += page_text + "\n"
#     except Exception as e:
#         print(f"Error reading PDF {pdf_path}: {e}")
#     return text
#
#
# def extract_text_from_image(image_path):
#     """Extracts txt from photo. Tesseract OCR. Not working. Trying PaddleOCR"""
#     text = ""
#     ocr = PaddleOCR(use_textline_orientation=True, lang='en')
#     try:
#         result = ocr.ocr(image_path)
#         for line in result:
#             for word_info in line:
#                 txt, conf = word_info[1]
#                 text += txt + " "
#     except Exception as e:
#         print(f"Error reading image {image_path}: {e}")
#     return text
#
#
# def get_text_from_file(file_path):
#     """
#     AUtomtcally detects file type and extracts text.
#     Please enter .pdf, .png, .jpg, .jpeg file
#     """
#
#     file_extension = os.path.splitext(file_path)[1].lower()
#
#     if file_extension == '.pdf':
#         return extract_text_from_pdf(file_path)
#     elif file_extension in ['.png', '.jpg', '.jpeg']:
#         return extract_text_from_image(file_path)
#     elif file_extension == '.txt':
#         try:
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 content = f.read()
#                 print(f"DEBUG: Read {len(content)} characters from .txt file.")
#                 return content
#         except Exception as e:
#             print(f"Error reading TXT file {file_path}: {e}")
#             return ""
#     else:
#         print(f"Unsupported file type: {file_extension}")
#         return ""
#
#
# def calculate_similarity(resume_text, job_description_text):
#     """Calculates semantic similarity using a chunking strategy."""
#     if not resume_text or not job_description_text:
#         return 0
#
#     cleaned_resume = clean_text(resume_text)
#     cleaned_jd = clean_text(job_description_text)
#
#     # 1. Create chunks from the resume
#     resume_chunks = chunk_text(cleaned_resume)
#     if not resume_chunks:
#         return 0
#
#     # 2. Encode the job description once, and all resume chunks
#     jd_embedding = model.encode([cleaned_jd])
#     resume_chunk_embeddings = model.encode(resume_chunks)
#
#     # 3. Calculate similarity between the JD and each resume chunk
#     similarities = cosine_similarity(jd_embedding, resume_chunk_embeddings)[0]
#
#     # 4. The final score is the highest similarity found
#     max_similarity = np.max(similarities)
#
#     return max_similarity
#
#
# if __name__ == '__main__':
#
#     resume_path = r"D:\Origin\Books\Projects and Learning\CV Scream\positive_example2.pdf"
#
#     job_description = """
#     Job Title: Software Engineer (Python)
#
#     We are looking for a skilled Python Developer to join our backend development team.
#     The ideal candidate will have strong experience with Python, Django, and REST APIs.
#     Familiarity with containerization technologies like Docker and cloud platforms like AWS is a plus.
#     Responsibilities include writing and testing code, debugging programs, and integrating applications
#     with third-party web services.
#     """
#
#     print(f"Processing resume: {resume_path}")
#     resume_text = get_text_from_file(resume_path)
#
#
#
#     if resume_text:
#         print("Resume text extracted successfully.")
#
#         score = calculate_similarity(resume_text, job_description)
#
#         print("\n--- Analysis Complete ---")
#         print(f"Job Description Match Score: {score:.2%}")
#
#         if score > 0.65:
#             print("You appear an ideal candidate for the job. ")
#         elif score > 0.50:
#             print("Your diverse skills may prove valuable")
#         else:
#             print("You might not be a good fit for the companay. ")
#     else:
#         print(f"Could not extract text from {resume_path}.")

# process_module.py

# process_module.py

import os
import pdfplumber
from sentence_transformers import SentenceTransformer
import re
import numpy as np

# --- Using the powerful model ---
MODEL_NAME = 'all-mpnet-base-v2'

print(f"Loading SBERT model: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
print("SBERT model loaded successfully.")


def extract_text_from_file(file_path):
    # (No changes to this function)
    if not os.path.exists(file_path):
        print(f"ERROR: File not found at the specified path: {file_path}")
        return ""
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.pdf':
        with pdfplumber.open(file_path) as pdf:
            return "".join(page.extract_text() or "" for page in pdf.pages)
    elif file_extension == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        print(f"Unsupported file type: {file_extension}")
        return ""


# --- Part 1: Hard Skills Analysis ---
def get_skills_from_jd(job_description_text):
    """Extracts a predefined list of skills from the JD text."""
    # In a more advanced system, this could use NLP to find skills automatically.
    # For now, we define them based on our sample JD.
    skills = [
        "python", "django", "rest apis", "docker", "aws", "sql",
        "javascript", "html", "css", "git"
    ]
    # Find which of these skills are mentioned in the JD
    found_skills = {skill for skill in skills if
                    re.search(r'\b' + re.escape(skill) + r'\b', job_description_text, re.IGNORECASE)}
    return list(found_skills)


def check_for_skills(resume_text, skills_to_find):
    """Checks for the presence of skills in the resume and returns matches."""
    found_skills = {skill for skill in skills_to_find if
                    re.search(r'\b' + re.escape(skill) + r'\b', resume_text, re.IGNORECASE)}
    return list(found_skills)


# --- Part 2: Semantic Experience Analysis ---
def calculate_experience_similarity(resume_text, job_description_text):
    """Calculates semantic similarity using a chunking strategy."""

    # Helper for chunking
    def chunk_text(text, chunk_size=256, overlap=50):
        tokens = text.split()
        return [" ".join(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size - overlap)]

    resume_chunks = chunk_text(resume_text)
    if not resume_chunks: return 0

    jd_embedding = model.encode([job_description_text])
    resume_chunk_embeddings = model.encode(resume_chunks)

    similarities = cosine_similarity(jd_embedding, resume_chunk_embeddings)[0]
    return np.max(similarities)


# --- Part 3: The Main Analysis Pipeline ---
def analyze_resume(resume_text, job_description):
    """Performs a hybrid analysis of a resume against a job description."""

    # 1. Hard Skills Check
    required_skills = get_skills_from_jd(job_description)
    matched_skills = check_for_skills(resume_text, required_skills)
    skill_match_score = len(matched_skills) / len(required_skills) if required_skills else 1.0

    # 2. Experience Context Check
    experience_score = calculate_experience_similarity(resume_text, job_description)

    # 3. Combined Weighted Score
    # Give more weight to the hard skills match, as it's a direct requirement.
    final_score = (skill_match_score * 0.6) + (experience_score * 0.4)

    return {
        "final_score": final_score,
        "skill_match_score": skill_match_score,
        "experience_score": experience_score,
        "required_skills": required_skills,
        "matched_skills": matched_skills,
        "missing_skills": list(set(required_skills) - set(matched_skills))
    }


if __name__ == '__main__':
    resume_path = r"D:\Origin\Books\Projects and Learning\CV Scream\positive_example2.pdf"

    job_description = """
    Job Title: Software Engineer (Python)
    We are looking for a skilled Python Developer to join our backend development team. 
    The ideal candidate will have strong experience with Python, Django, and REST APIs. 
    Familiarity with containerization technologies like Docker and cloud platforms like AWS is a plus.
    Responsibilities include writing and testing code, debugging programs, and integrating applications 
    with third-party web services.
    """

    resume_text = extract_text_from_file(resume_path)

    if resume_text and resume_text.strip():
        analysis_result = analyze_resume(resume_text, job_description)

        print("\n--- Candidate Analysis Report ---")
        print(f"Overall Match Score: {analysis_result['final_score']:.2%}")
        print("-" * 30)
        print(
            f"Required Skills Found: {len(analysis_result['matched_skills'])}/{len(analysis_result['required_skills'])} ({analysis_result['skill_match_score']:.2%})")
        print(f"  ✓ Matched: {', '.join(analysis_result['matched_skills'])}")
        if analysis_result['missing_skills']:
            print(f"  ✗ Missing: {', '.join(analysis_result['missing_skills'])}")
        print(f"Contextual Experience Match: {analysis_result['experience_score']:.2%}")
        print("-" * 30)

        if analysis_result['final_score'] > 0.70:
            print("Verdict: Strong Candidate. High match in both skills and experience. ✅")
        elif analysis_result['final_score'] > 0.55:
            print("Verdict: Potential Candidate. Good skill overlap, review experience. 🤔")
        else:
            print("Verdict: Unlikely Match. Significant skill gap. ❌")
    else:
        print(f"Could not extract text from {resume_path}.")