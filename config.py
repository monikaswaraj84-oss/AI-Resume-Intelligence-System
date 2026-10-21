"""Central configuration: every module imports its file paths from here so the
project works no matter which directory you start Python from."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
RAW_RESUME_CSV = os.path.join(DATASET_DIR, "resume_dataset.csv")
CLEAN_RESUME_CSV = os.path.join(DATASET_DIR, "resume_dataset_clean.csv")
JOB_CSV = os.path.join(BASE_DIR, "job_descriptions", "job_descriptions.csv")
SKILLS_TXT = os.path.join(BASE_DIR, "technical_skills", "skills.txt")
SKILLS_CSV = os.path.join(BASE_DIR, "technical_skills", "resumes_development.csv")
MODEL_PATH = os.path.join(BASE_DIR, "ml_model", "resume_job_role_model.joblib")

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB
