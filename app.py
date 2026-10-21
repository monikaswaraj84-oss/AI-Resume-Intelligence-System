"""FastAPI application: serves the web UI and the /analyze-resume API."""
import asyncio
import json
import logging
import os
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import role_model
from ats_analyzer import analyze_resume
from config import BASE_DIR, FRONTEND_DIR, MAX_UPLOAD_BYTES, MODEL_PATH
from resume_generator import generate_latex_resume
from resume_parser import extract_profile, extract_text, split_sections, target_title
from skills_vocab import display_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ats_app")


@asynccontextmanager
async def lifespan(_app):
    role_model.load_model()  # load ML model once at startup (no retraining at serve-time)
    yield


app = FastAPI(
    title="AI Resume ATS Analyzer",
    lifespan=lifespan,
    description="ATS scoring, job-role prediction and authentic LaTeX resume generation",
    version="5.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def home():
    response = FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "AI Resume ATS Analyzer", "model": role_model.status()}


@app.get("/api/sample-data")
def sample_data():
    sample_jd_path = os.path.join(BASE_DIR, "tests", "sample_job_description.txt")
    sample_jd = ""
    if os.path.exists(sample_jd_path):
        with open(sample_jd_path, "r", encoding="utf-8") as f:
            sample_jd = f.read().strip()
    return {
        "job_descriptions": {
            "Data Scientist": sample_jd or (
                "Data Scientist\n\n"
                "Responsibilities:\n"
                "- Develop machine learning models and NLP pipelines.\n"
                "- Perform statistical analysis and data visualization using Python, Pandas, and NumPy.\n"
                "- Build predictive models with Scikit-learn and TensorFlow.\n\n"
                "Required Skills:\n"
                "Python, Machine Learning, Pandas, NumPy, Scikit-learn, SQL, Statistics.\n\n"
                "Preferred Skills:\n"
                "AWS, Docker, Deep Learning, Natural Language Processing, PyTorch."
            ),
            "Software Engineer": (
                "Software Engineer\n\n"
                "Responsibilities:\n"
                "- Build scalable software applications and robust backend services.\n"
                "- Design and integrate REST APIs and microservices.\n"
                "- Write clean, maintainable code following TDD and agile best practices.\n\n"
                "Required Skills:\n"
                "Python, Django, Microservices, REST API, SQL, Git, TDD.\n\n"
                "Preferred Skills:\n"
                "React, Docker, CI/CD, Redis, Cloud Platforms."
            ),
            "Frontend Engineer": (
                "Frontend Engineer\n\n"
                "Responsibilities:\n"
                "- Build responsive, high-performance web applications using modern component-based architecture.\n"
                "- Translate UI/UX designs into pixel-perfect web interfaces using React and TypeScript.\n"
                "- Integrate backend REST and GraphQL APIs and optimize frontend rendering.\n\n"
                "Required Skills:\n"
                "React, TypeScript, JavaScript, HTML, CSS, Redux, Git.\n\n"
                "Preferred Skills:\n"
                "Next.js, Tailwind CSS, Vue, GraphQL, Jest."
            ),
            "Backend Engineer": (
                "Backend Engineer\n\n"
                "Responsibilities:\n"
                "- Design and develop high-throughput server-side applications and microservices.\n"
                "- Architect database schemas and optimize complex SQL queries in PostgreSQL.\n"
                "- Implement security best practices, caching layers with Redis, and CI/CD automation.\n\n"
                "Required Skills:\n"
                "Python, Node.js, PostgreSQL, REST API, Microservices, Docker, Git.\n\n"
                "Preferred Skills:\n"
                "Redis, Kubernetes, AWS, GraphQL, MongoDB, CI/CD."
            ),
            "Full Stack Engineer": (
                "Full Stack Engineer\n\n"
                "Responsibilities:\n"
                "- Design and implement scalable frontend interfaces using React, TypeScript, and modern CSS.\n"
                "- Develop backend REST APIs and microservices using Node.js, Express, and Python.\n"
                "- Manage database schemas using PostgreSQL and MongoDB.\n"
                "- Implement CI/CD deployment workflows using Docker and Git.\n\n"
                "Required Skills:\n"
                "React, Node.js, TypeScript, JavaScript, REST API, SQL, Git, Docker.\n\n"
                "Preferred Skills:\n"
                "AWS, Kubernetes, GraphQL, Next.js, Microservices, CI/CD."
            ),
            "DevOps Engineer": (
                "DevOps Engineer\n\n"
                "Responsibilities:\n"
                "- Design and automate cloud infrastructure using AWS, Linux, and Terraform.\n"
                "- Deploy and manage containerized applications with Docker and Kubernetes.\n"
                "- Build automated CI/CD deployment pipelines using Jenkins and GitHub Actions.\n\n"
                "Required Skills:\n"
                "AWS, Docker, Kubernetes, Linux, CI/CD, Python, Terraform.\n\n"
                "Preferred Skills:\n"
                "Ansible, Bash, Microservices, Security Best Practices, Prometheus."
            ),
            "Data Engineer": (
                "Data Engineer\n\n"
                "Responsibilities:\n"
                "- Design, construct, and maintain scalable ETL pipelines and big data workflows.\n"
                "- Optimize relational and distributed data warehouses for analytical queries.\n"
                "- Ensure data quality, pipeline reliability, and schema evolution management.\n\n"
                "Required Skills:\n"
                "Python, SQL, Apache Spark, ETL, PostgreSQL, Airflow, Data Engineering.\n\n"
                "Preferred Skills:\n"
                "AWS, Snowflake, Apache Kafka, Docker, Big Data, Hadoop."
            ),
            "Mobile Developer": (
                "Mobile Developer\n\n"
                "Responsibilities:\n"
                "- Develop performant cross-platform mobile applications for iOS and Android.\n"
                "- Integrate secure mobile APIs, authentication, and offline caching.\n"
                "- Publish and maintain applications through app store deployment cycles.\n\n"
                "Required Skills:\n"
                "React Native, Flutter, JavaScript, TypeScript, REST API, Git.\n\n"
                "Preferred Skills:\n"
                "Swift, Kotlin, Android, iOS, Firebase, GraphQL."
            ),
            "QA Engineer": (
                "QA Engineer\n\n"
                "Responsibilities:\n"
                "- Design comprehensive test plans, automated test suites, and regression frameworks.\n"
                "- Execute functional, performance, security, and API integration testing.\n"
                "- Identify, log, and track software defects and collaborate closely with developers.\n\n"
                "Required Skills:\n"
                "Python, Selenium, Unit Testing, Pytest, Integration Testing, REST API, Git.\n\n"
                "Preferred Skills:\n"
                "CI/CD, Docker, JIRA, Postman, Performance Tuning."
            ),
            "Product Engineer": (
                "Product Engineer\n\n"
                "Responsibilities:\n"
                "- Bridge customer needs and technical architecture to deliver high-impact software features.\n"
                "- Prototype rapidly, measure user engagement analytics, and iterate quickly.\n"
                "- Work across the full stack to build reliable, user-centric software solutions.\n\n"
                "Required Skills:\n"
                "JavaScript, Python, React, REST API, SQL, Software Development, Git.\n\n"
                "Preferred Skills:\n"
                "AWS, TypeScript, Docker, Analytics, Agile, Microservices."
            ),
            "Financial Analyst": (
                "Financial Analyst\n\n"
                "Responsibilities:\n"
                "- Build detailed financial models, forecasts, valuation analyses, and variance reports.\n"
                "- Assist leadership in quarterly budgeting, expenditure tracking, and investment appraisal.\n"
                "- Analyze economic trends, financial statements, and risk mitigation strategies.\n\n"
                "Required Skills:\n"
                "Financial Modeling, Budgeting, Forecasting, Accounting, Valuation, Risk Analysis, Excel.\n\n"
                "Preferred Skills:\n"
                "Power BI, Tableau, SQL, Corporate Finance, Data Analysis."
            ),
            "Marketing Manager": (
                "Marketing Manager\n\n"
                "Responsibilities:\n"
                "- Plan and execute integrated multi-channel marketing campaigns across digital platforms.\n"
                "- Drive brand awareness, lead generation, and customer acquisition strategies.\n"
                "- Analyze campaign KPIs, conversion funnels, and ROI using Google Analytics and CRM tools.\n\n"
                "Required Skills:\n"
                "Market Research, Campaign Management, Google Analytics, SEO, SEM, Branding, Lead Generation, Social Media.\n\n"
                "Preferred Skills:\n"
                "CRM, Content Strategy, Email Marketing, Budgeting, Project Management."
            ),
            "HR Executive": (
                "HR Executive\n\n"
                "Responsibilities:\n"
                "- Oversee end-to-end recruitment, talent acquisition, interviewing, and candidate onboarding.\n"
                "- Implement and enforce company HR policies, compliance, and employee documentation.\n"
                "- Manage employee engagement programs, performance appraisals, and payroll administration.\n\n"
                "Required Skills:\n"
                "Recruitment, Talent Acquisition, Onboarding, Payroll, HR Policies, Employee Engagement, Compliance.\n\n"
                "Preferred Skills:\n"
                "Performance Management, Conflict Resolution, Communication, Labor Laws."
            ),
        }
    }


@app.get("/api/metrics")
def get_metrics():
    metrics_file = os.path.join(os.path.dirname(MODEL_PATH), "evaluation_metrics.json")
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Error reading metrics file: %s", exc)
    return {"status": "Model loaded", "model_status": role_model.status()}


def _process_analysis_sync(filename, data, job_description):
    """Synchronous CPU worker run via thread pool to keep event loop unblocked."""
    text = extract_text(filename, data)
    if len(text.split()) < 20:
        raise ValueError(
            "Could not extract enough text. Scanned/image-only resumes are not supported; "
            "upload a text-based PDF or DOCX."
        )

    # ML role prediction
    role, confidence = role_model.predict_role(text)

    # ATS Scoring (passing role & confidence avoids double computation)
    result = analyze_resume(text, job_description, pred_role=role, role_conf=confidence)

    sections = split_sections(text)
    profile = extract_profile(text, sections)
    resume_skills = [display_name(s) for s in result["_resume_skills_canon"]]
    matched_first = result["matched_skills"] + [s for s in resume_skills if s not in result["matched_skills"]]

    title = target_title(job_description, role if not role.startswith(("Uncertain", "Not", "Pred")) else "Professional")

    # Name detection fallback
    candidate_name = profile.get("name") or "Candidate Name"
    if candidate_name == "Candidate Name" and filename:
        base = os.path.splitext(filename)[0]
        cleaned_base = re.sub(r"[-_](?:resume|cv|latest|202\d|v\d+).*", "", base, flags=re.I)
        cleaned_base = re.sub(r"[^a-zA-Z\s]", " ", cleaned_base).strip()
        if 1 < len(cleaned_base.split()) <= 4:
            candidate_name = cleaned_base.title()
            profile["name"] = candidate_name

    # Zero Hallucination: use candidate's actual extracted summary if present, do not fabricate
    candidate_summary = " ".join(sections.get("summary", [])).strip()

    # Preserve candidate chronological order (no destructive re-sorting)
    exp_bullets = sections.get("experience", [])
    proj_bullets = sections.get("projects", [])
    edu_bullets = profile.get("education") or sections.get("education", [])
    cert_bullets = sections.get("certifications", [])

    latex = generate_latex_resume(
        name=candidate_name,
        email=profile.get("email", ""),
        phone=profile.get("phone", ""),
        links=profile.get("links", []),
        summary=candidate_summary,
        skills=matched_first,
        education=edu_bullets,
        experience=exp_bullets,
        projects=proj_bullets,
        certifications=cert_bullets,
        missing_skills=result["missing_skills"],
        target_role=title,
    )

    score = result["ats_score"]
    advice = (
        "Strong alignment. Keep the relevant evidence and the ATS-friendly formatting."
        if score >= 75
        else "Moderate alignment. Add truthful missing skills and stronger job-specific evidence."
        if score >= 50
        else "Low alignment. Rework the resume around the target role, using only skills you genuinely have."
    )

    public = {k: v for k, v in result.items() if not k.startswith("_")}
    public.update({
        "resume_filename": filename,
        "predicted_job_role": role,
        "role_confidence": confidence,
        "resume_advice": advice,
        "latex_code": latex,
        "detected_name": candidate_name,
    })
    return public


@app.post("/analyze-resume")
async def analyze_resume_api(resume: UploadFile = File(...), job_description: str = Form(...)):
    if len(job_description.split()) < 5:
        raise HTTPException(400, "Please paste a complete job description (at least a few words).")

    filename = (resume.filename or "").lower()
    if not filename.endswith((".pdf", ".docx")):
        raise HTTPException(400, "Only PDF and DOCX files are supported.")

    # Guard upload size before unbounded memory accumulation (A1)
    chunk = await resume.read(MAX_UPLOAD_BYTES + 1024)
    if len(chunk) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (maximum 5 MB).")
    if not chunk:
        raise HTTPException(400, "The uploaded resume is empty.")

    # Magic byte validation (A4)
    if filename.endswith(".pdf") and not chunk.startswith(b"%PDF"):
        raise HTTPException(400, "Invalid PDF file header. Please upload a genuine PDF.")
    if filename.endswith(".docx") and not chunk.startswith(b"PK\x03\x04"):
        raise HTTPException(400, "Invalid DOCX file header. Please upload a genuine Word document.")

    try:
        # Offload CPU bound parsing/scoring to thread pool (A2)
        return await asyncio.to_thread(_process_analysis_sync, resume.filename, chunk, job_description)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        logger.error("Analysis failed: %s", exc, exc_info=True)
        raise HTTPException(400, "Unable to read the resume. The file may be corrupted or unreadable.")
