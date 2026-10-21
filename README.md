# AI Resume ATS Analyzer & Screening System

An academic, production-ready, machine learning-driven Applicant Tracking System (ATS) resume analyzer, job-role domain classifier, and automated Overleaf-ready LaTeX resume generator.

Designed for college project submission and real-world evaluation, built with **Python 3**, **FastAPI**, **Scikit-learn**, and modern **Vanilla HTML5/CSS3/JavaScript**.

---

## 🚀 Quick Start Guide

### 1. Launch the Application
Run the launcher script from the project root:
```bash
.\venv\Scripts\python.exe run.py
```
Or with standard Python:
```bash
python run.py
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your web browser.

### 2. Run Automated Test Suite
Run the comprehensive suite of 12 unit and integration tests:
```bash
.\venv\Scripts\python.exe -m pytest tests -v
```

---

## 📊 ATS Scoring Formulation

The system scores candidate resumes against job descriptions using a calibrated 5-factor mathematical model:

$$\text{ATS Score} = 0.40 \cdot S_{\text{skill}} + 0.20 \cdot S_{\text{role}} + 0.15 \cdot S_{\text{keyword}} + 0.15 \cdot S_{\text{similarity}} + 0.10 \cdot S_{\text{format}}$$

| Component | Weight | Implementation Details |
|---|:---:|---|
| **Skill Match** ($S_{\text{skill}}$) | **40%** | Canonical alias-mapped technical skill extraction. Direct matches receive $1.0\times$ credit; related clustered transferable competencies receive $0.5\times$ partial credit. Required skills are weighted at $1.0$, while multi-line preferred skills are weighted at $0.5$. |
| **Role Alignment** ($S_{\text{role}}$) | **20%** | Supervised ML domain classification against the target role domain cluster, combined with target title keyword verification. |
| **Keyword Coverage** ($S_{\text{keyword}}$) | **15%** | Frequency and stem-based extraction of high-priority non-skill technical and workflow tokens (e.g. *pipelines*, *architecture*, *databases*). |
| **Text Similarity** ($S_{\text{similarity}}$) | **15%** | Sublinear TF-IDF bi-gram cosine similarity rescaled using a power curve calibrated for document length asymmetry between concise JDs and multi-page resumes. |
| **Format Quality** ($S_{\text{format}}$) | **10%** | Structural ATS readability checks: contact info completeness (email, phone), standard section headings (Skills, Experience, Education), and appropriate word count range (90–1200 words). |

---

## 🤖 Machine Learning Model Architecture

The job-role classifier identifies the domain profile of candidate resumes across 5 core roles:
- **Data Scientist**
- **Software Engineer**
- **Financial Analyst**
- **Marketing Manager**
- **HR Executive**

### ML Pipeline
1. **Preprocessing**: Normalization, PII redaction (anonymizing candidate names, emails, and phone numbers to prevent bias), and Unicode NFKC normalization.
2. **Feature Extraction**: TF-IDF Vectorizer with unigrams and bigrams (`ngram_range=(1, 2)`), sublinear term frequency scaling, and English stop-word removal.
3. **Classifier**: Multi-class Logistic Regression with balanced class weighting (`class_weight="balanced"`, $C=2.0$, L-BFGS solver).
4. **Validation**: 5-Fold Stratified Cross-Validation (`StratifiedKFold(shuffle=True, random_state=42)`) and an independent 20% hold-out test set.

### Honest Academic Metrics (`evaluation_metrics.json`)
* **Hold-out Test Accuracy (20%)**: **73.64%**
* **5-Fold Stratified CV Accuracy**: **73.63% (±1.87%)**
* **Data Leakage Prevention**: Training texts contain zero explicit job-role labels, ensuring the classifier learns genuine domain competencies rather than memorized labels.

---

## 🛠️ Key Architectural Highlights

* **Zero Hallucination Resume Generation**: Generates Overleaf-ready LaTeX (`.tex`) resumes without fabricating fake career claims. Candidate summaries are used only if genuinely present; chronology and bullet structures are preserved.
* **Typographic & DOCX Robustness**: Standardizes PDF ligatures (`ﬁ`, `ﬂ` $\rightarrow$ `fi`, `fl`) and bullet points without mangling words or fusing lines.
* **Secure In-Memory Processing**: Uploaded documents are parsed entirely in memory and never written to disk, ensuring user privacy and data security.
* **Non-Blocking Async Server**: CPU-intensive parsing and scikit-learn matrix computations are offloaded to thread workers (`asyncio.to_thread`), preventing event loop latency.

---

## 📁 Project Directory Structure

```
AI_Resume_Screening_FIXED/
│
├── frontend/                     # Modern Web User Interface
│   ├── index.html                # Single-page UI layout
│   ├── script.js                 # Drag & drop, validation, and rendering logic
│   └── style.css                 # Clean responsive stylesheet
│
├── datasets/                     # Training & benchmark datasets
│   ├── resume_dataset.csv        # Raw initial dataset
│   └── resume_dataset_clean.csv  # Cleaned dataset without label leakage
│
├── ml_model/                     # Machine learning pipeline
│   ├── resume_job_role_model.joblib  # Trained model weights
│   ├── clean_dataset.py          # Data repair and leakage-free dataset builder
│   ├── train_model.py            # Model training & stratified evaluation script
│   └── evaluation_metrics.json   # Honest academic evaluation metrics
│
├── technical_skills/             # Technical vocabulary & aliases
│   ├── skills.txt
│   └── resumes_development.csv
│
├── tests/                        # Automated unit & integration tests
│   ├── sample_job_description.txt
│   ├── sample_resume.pdf
│   ├── sample_resume.docx
│   └── test_app.py               # 12 automated test cases (all passing)
│
├── venv/                         # Python virtual environment
│
├── app.py                        # FastAPI application & REST endpoints
├── ats_analyzer.py               # 5-factor ATS scoring engine
├── config.py                     # Centralized paths and size limits
├── resume_parser.py              # Robust PDF/DOCX extractor
├── resume_generator.py           # Zero-hallucination LaTeX resume generator
├── role_model.py                 # Safe inference wrapper (no serve-time training)
├── skills_vocab.py               # Canonical skills matcher & ambiguity guard
│
├── run.py                        # One-click application launcher
├── requirements.txt              # Production dependency specifications
└── README.md                     # Comprehensive project documentation
```

---

## 🎓 Academic Submission Checklist

- [x] Machine learning data leakage eliminated and verified.
- [x] Honest 5-fold cross-validation metrics saved and documented.
- [x] 12/12 automated pytest tests passing cleanly.
- [x] Zero-hallucination resume generator strictly preserves candidate data.
- [x] Clean directory structure with all redundant/duplicate files removed.
- [x] Localhost security binding and input size guards implemented.
