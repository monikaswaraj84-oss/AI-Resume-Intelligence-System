"""ATS scoring engine.

ATS score (0-100) = 40% skill match (direct + transferable) + 20% role alignment + 15% keyword coverage + 15% text similarity + 10% resume format:
  * skill match      : share of the job's skills found directly (1.0x) or via related transferable competencies (0.5x)
  * role alignment   : machine learning role prediction domain match against the target job profile
  * keyword coverage : share of the job's most important non-skill action/workflow terms found in the resume
  * text similarity  : calibrated TF-IDF cosine similarity between resume and job description (rescaled for document length asymmetry)
  * format           : contact info, standard ATS sections, sensible length
"""
import re
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import role_model
from resume_parser import EMAIL_RE, PHONE_RE, extract_profile, split_sections, target_title
from skills_vocab import display_name, find_skills, normalize

# Calibrated enterprise ATS weighting distribution
WEIGHTS = {
    "skill": 0.40,
    "role": 0.20,
    "keyword": 0.15,
    "similarity": 0.15,
    "format": 0.10,
}

JD_STOP = set(ENGLISH_STOP_WORDS) | {
    "experience", "years", "year", "work", "working", "job", "role", "candidate", "required", "requirements",
    "requirement", "responsibilities", "responsibility", "looking", "knowledge", "skills", "skill", "should",
    "must", "ability", "team", "teams", "strong", "good", "excellent", "preferred", "plus", "etc", "including",
    "will", "join", "company", "position", "opportunity", "qualifications", "qualification", "ideal",
    "least", "like", "using", "use", "new", "highly", "related", "ensure", "across", "well", "within",
    "able", "need", "needs", "build", "building", "help", "make", "also", "key", "degree", "bachelor",
    "bachelors", "master", "masters", "familiarity", "understanding", "proficiency", "proficient", "hands", "nice",
}

PREFERRED_HEADER_RE = re.compile(
    r"\b(preferred|nice[\s-]to[\s-]have|good[\s-]to[\s-]have|bonus|desirable|optional|plus\s+points?)\b",
    re.I
)
REQUIRED_HEADER_RE = re.compile(
    r"\b(requirements?|qualifications?|must[\s-]have|minimum|basic|essential|what\s+you)\b",
    re.I
)

# Canonical skill clusters for transferable competency matching (partial credit 0.5x)
SKILL_CLUSTERS = {
    # Deep Learning Frameworks
    "pytorch": ["tensorflow", "keras", "deep learning"],
    "tensorflow": ["pytorch", "keras", "deep learning"],
    "keras": ["tensorflow", "pytorch", "deep learning"],
    "deep learning": ["machine learning", "pytorch", "tensorflow"],

    # Machine Learning / AI / Data Science
    "machine learning": ["data science", "scikit-learn", "deep learning", "artificial intelligence", "statistics"],
    "scikit-learn": ["machine learning", "data science", "python"],
    "data science": ["machine learning", "data analysis", "statistics", "pandas", "numpy"],
    "statistics": ["data analysis", "machine learning", "forecasting", "data science"],
    "natural language processing": ["deep learning", "large language models", "generative ai"],
    "computer vision": ["deep learning"],
    "generative ai": ["large language models", "deep learning", "rag"],
    "large language models": ["generative ai", "natural language processing", "deep learning"],

    # Cloud & Container Infrastructure
    "docker": ["kubernetes", "ci/cd", "devops"],
    "kubernetes": ["docker", "microservices", "devops"],
    "aws": ["azure", "gcp"],
    "azure": ["aws", "gcp"],
    "gcp": ["aws", "azure"],
    "ci/cd": ["jenkins", "github actions", "docker", "devops"],
    "jenkins": ["ci/cd", "github actions", "docker"],
    "github actions": ["ci/cd", "jenkins"],
    "terraform": ["ansible", "devops"],

    # Big Data & Data Pipelines
    "spark": ["hadoop", "etl", "data engineering"],
    "hadoop": ["spark", "etl", "data engineering"],
    "airflow": ["etl", "data engineering", "spark"],
    "etl": ["data engineering", "spark", "sql", "airflow"],
    "data engineering": ["etl", "spark", "sql", "airflow", "python"],

    # Relational & NoSQL Databases
    "postgresql": ["mysql", "sqlite", "sql"],
    "mysql": ["postgresql", "sqlite", "sql"],
    "sql": ["postgresql", "mysql", "sqlite", "nosql"],
    "mongodb": ["nosql", "dynamodb", "redis", "sql"],
    "redis": ["nosql", "mongodb"],

    # Frontend Technologies
    "react": ["vue", "angular", "next.js", "typescript", "javascript"],
    "vue": ["react", "angular", "javascript"],
    "angular": ["react", "vue", "typescript"],
    "typescript": ["javascript", "react", "node.js"],
    "javascript": ["typescript", "node.js"],
    "next.js": ["react", "node.js"],
    "redux": ["react"],
    "tailwind css": ["css", "bootstrap", "sass"],
    "bootstrap": ["css", "tailwind css", "html"],

    # Backend & API Development
    "rest api": ["fastapi", "flask", "django", "express.js", "graphql", "microservices"],
    "fastapi": ["flask", "django", "rest api", "python"],
    "flask": ["fastapi", "django", "rest api", "python"],
    "django": ["flask", "fastapi", "rest api", "python"],
    "node.js": ["express.js", "javascript", "typescript"],
    "express.js": ["node.js", "rest api"],
    "microservices": ["rest api", "docker", "kubernetes"],
    "graphql": ["rest api"],

    # Testing & Quality Assurance
    "unit testing": ["pytest", "junit", "tdd", "integration testing", "selenium"],
    "integration testing": ["unit testing", "selenium", "pytest"],
    "pytest": ["unit testing", "tdd", "python"],
    "selenium": ["unit testing", "integration testing"],
    "tdd": ["unit testing", "integration testing", "pytest"],

    # Business, HR, & Management
    "recruitment": ["talent acquisition", "onboarding", "hr policies"],
    "talent acquisition": ["recruitment", "onboarding"],
    "onboarding": ["recruitment", "talent acquisition", "hr policies"],
    "payroll": ["accounting", "hr policies", "compliance"],
    "hr policies": ["compliance", "employee engagement", "recruitment"],
    "employee engagement": ["hr policies", "performance management"],

    # Finance & Analytics
    "financial modeling": ["valuation", "budgeting", "forecasting", "excel", "accounting"],
    "budgeting": ["financial modeling", "forecasting", "accounting"],
    "valuation": ["financial modeling", "risk analysis"],
    "accounting": ["financial modeling", "budgeting", "payroll"],
    "power bi": ["tableau", "data visualization", "excel"],
    "tableau": ["power bi", "data visualization", "excel"],
    "excel": ["financial modeling", "data analysis", "accounting"],

    # Marketing
    "seo": ["sem", "google analytics", "content strategy"],
    "sem": ["seo", "google analytics"],
    "google analytics": ["seo", "market research"],
    "campaign management": ["lead generation", "market research", "social media", "branding"],
    "lead generation": ["campaign management", "crm"],
}

# Role domain relationship clusters for ML alignment
ROLE_DOMAIN_CLUSTER = {
    "Data Scientist": {
        "Data Scientist": 100, "Data Engineer": 88, "Software Engineer": 72,
        "Product Engineer": 70, "QA Engineer": 60, "DevOps Engineer": 60
    },
    "Software Engineer": {
        "Software Engineer": 100, "Frontend Engineer": 95, "Backend Engineer": 95,
        "Full Stack Engineer": 98, "DevOps Engineer": 90, "Mobile Developer": 90,
        "QA Engineer": 90, "Product Engineer": 92, "Data Engineer": 82, "Data Scientist": 75
    },
    "Financial Analyst": {"Financial Analyst": 100},
    "Marketing Manager": {"Marketing Manager": 100},
    "HR Executive": {"HR Executive": 100},
}


def _normalize_token(w):
    """Normalize word endings cleanly to handle plurals without mangling bases."""
    w = w.lower()
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("sses", "shes", "ches", "xes", "zes")) and len(w) > 4:
        return w[:-2]
    if w.endswith("es") and len(w) > 4 and not w.endswith(("ses", "tes", "des")):
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) > 3:
        return w[:-1]
    if w.endswith("ing") and len(w) > 5:
        base = w[:-3]
        if len(base) > 2 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if w.endswith("ed") and len(w) > 4:
        base = w[:-2]
        if len(base) > 2 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    return w


def _tokens(text):
    toks = (t.strip(".") for t in re.findall(r"[a-z][a-z0-9+#.]{2,}", normalize(text)))
    return [t for t in toks if len(t) >= 3 and t not in JD_STOP]


def jd_skill_weights(job_description):
    """{canonical skill: weight}.
    
    Skills mentioned under 'preferred' wording or section headers count 0.5.
    Skills mentioned under 'required' or general headers count 1.0.
    Tracks multi-line section context properly.
    """
    if not job_description:
        return {}

    skills = find_skills(job_description)
    if not skills:
        return {}

    weights = {s: 1.0 for s in skills}
    current_mode = "required"
    pref_skills = set()
    req_skills = set()

    for line in job_description.splitlines():
        trimmed = line.strip()
        if not trimmed:
            continue

        # Check section headers
        if PREFERRED_HEADER_RE.search(trimmed) and (len(trimmed) < 60 or trimmed.endswith(":")):
            current_mode = "preferred"
        elif REQUIRED_HEADER_RE.search(trimmed) and (len(trimmed) < 60 or trimmed.endswith(":")):
            current_mode = "required"
        elif trimmed.endswith(":") and len(trimmed) < 40:
            current_mode = "required"

        line_skills = find_skills(trimmed)
        if not line_skills:
            continue

        if PREFERRED_HEADER_RE.search(trimmed):
            pref_skills.update(line_skills)
        elif current_mode == "preferred":
            pref_skills.update(line_skills)
        else:
            req_skills.update(line_skills)

    for s in (pref_skills - req_skills):
        if s in weights:
            weights[s] = 0.5

    return weights


def jd_keywords(job_description, skills, top_n=20):
    """Important non-skill terms of the JD (by frequency, then position).
    
    Returns clean original words from the job description rather than chopped stems.
    """
    skill_tokens = set()
    for s in skills:
        skill_tokens.update(normalize(s).split())
    toks = [t for t in _tokens(job_description) if t not in skill_tokens]
    
    stem_counts = Counter()
    stem_to_words = {}
    first_seen = {}
    
    for i, t in enumerate(toks):
        norm = _normalize_token(t)
        stem_counts[norm] += 1
        if norm not in first_seen:
            first_seen[norm] = i
        if norm not in stem_to_words:
            stem_to_words[norm] = Counter()
        stem_to_words[norm][t] += 1

    ordered_stems = sorted(stem_counts, key=lambda k: (-stem_counts[k], first_seen[k]))
    
    chosen_words = []
    for s in ordered_stems[:top_n]:
        best_word = stem_to_words[s].most_common(1)[0][0]
        chosen_words.append(best_word)
        
    return chosen_words


def keyword_coverage(resume_text, keywords):
    """Measures presence of JD keywords in resume, returning clean original words."""
    if not keywords:
        return 0.0, [], []
    res_toks = _tokens(resume_text)
    resume_norms = {_normalize_token(t) for t in res_toks} | set(res_toks)
    
    hit = []
    miss = []
    for kw in keywords:
        norm = _normalize_token(kw)
        if norm in resume_norms or kw in resume_norms:
            hit.append(kw)
        else:
            miss.append(kw)
            
    pct = round(100.0 * len(hit) / len(keywords), 2)
    return pct, hit, miss


def text_similarity(resume_text, job_description):
    """Calibrated TF-IDF cosine similarity.

    Due to extreme document length and structure asymmetry (concise JD vs. dense resume with bullet points
    and academic history), raw cosine rarely exceeds 0.20-0.24 for high-fit documents. An asymmetric power
    curve maps raw cosine realistically: 0.13 raw -> ~68%, 0.18 raw -> 85%, 0.24 raw -> ~98%, 0.0 raw -> 0%.
    """
    a, b = normalize(resume_text), normalize(job_description)
    if not a or not b:
        return 0.0
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        m = vec.fit_transform([a, b])
        raw = float(cosine_similarity(m[0:1], m[1:2])[0][0])
    except ValueError:
        return 0.0
    if raw <= 0.0:
        return 0.0
    calibrated = ((raw / 0.18) ** 0.75) * 85.0
    return round(min(100.0, max(0.0, calibrated)), 2)


def compute_role_alignment(resume_text, job_description, pred_role=None, role_conf=None):
    """Evaluates candidate domain fit against target job title using ML classifier and title keywords."""
    if not resume_text or not job_description:
        return 0.0
    if pred_role is None or role_conf is None:
        pred_role, role_conf = role_model.predict_role(resume_text)

    # Clean predicted role if format is 'Uncertain (closest: ...)'
    clean_pred = pred_role
    if "(" in pred_role and ":" in pred_role:
        clean_pred = pred_role.split(":")[-1].strip(" )")

    jd_role, jd_conf = role_model.predict_role(job_description)
    if "(" in jd_role and ":" in jd_role:
        jd_role = jd_role.split(":")[-1].strip(" )")

    cluster = ROLE_DOMAIN_CLUSTER.get(clean_pred, {clean_pred: 100})
    base_match = cluster.get(jd_role, 5.0 if clean_pred != jd_role else 100.0)

    # Check title keyword overlap (retaining professional role nouns like engineer/scientist)
    title = target_title(job_description, fallback="")
    title_stop = set(ENGLISH_STOP_WORDS) | {"position", "role", "job", "senior", "junior", "lead", "staff", "principal"}
    title_words = [w for w in normalize(title).split() if len(w) > 2 and w not in title_stop]
    res_norm = normalize(resume_text)
    hits = sum(1 for w in title_words if w in res_norm)
    title_boost = (hits / len(title_words) * 12.0) if title_words else 0.0

    conf_factor = max(0.40, min(1.0, (role_conf or 50.0) / 100.0))
    score = (base_match * conf_factor * 0.88) + title_boost
    return round(min(100.0, max(0.0, score)), 2)


def format_score(resume_text):
    """Simple ATS-friendliness checks. Returns (score, list of problems)."""
    sections = split_sections(resume_text)
    problems, pts = [], 0
    if EMAIL_RE.search(resume_text):
        pts += 20
    else:
        problems.append("No email address found.")
    if extract_profile(resume_text, sections)["phone"]:
        pts += 15
    else:
        problems.append("No phone number found.")
    for key, label, value in (("skills", "Skills", 20), ("experience", "Experience (or Internships)", 15),
                              ("education", "Education", 15)):
        if sections[key]:
            pts += value
        else:
            problems.append(f"No clear '{label}' section heading found.")
    if sections["projects"] or sections["experience"]:
        pts += 5
    words = len(resume_text.split())
    if 90 <= words <= 1200:
        pts += 10
    else:
        problems.append("Resume is very short." if words < 90 else "Resume is very long; aim for 1-2 pages.")
    return min(100, pts), problems


def generate_suggestions(missing_skills, related_skills, missing_keywords, scores, format_problems, matched_count, missing_preferred=None):
    tips = []
    if missing_skills:
        tips.append("Add verified evidence for these missing required job skills: " + ", ".join(missing_skills[:8]) + ".")
    if missing_preferred:
        tips.append("Consider highlighting experience with these preferred tools: " + ", ".join(missing_preferred[:6]) + ".")
    if related_skills:
        rel_names = [r.split(" (")[0] for r in related_skills[:4]]
        tips.append(f"You have transferable skills for {', '.join(rel_names)}. Mention these tools explicitly if you have hands-on experience to maximize automated ATS parsing.")
    if missing_keywords:
        tips.append("Use the job's own terminology where truthful. High-frequency missing terms: "
                    + ", ".join(missing_keywords[:6]) + ".")
    if scores.get("role", 100) < 65:
        tips.append("Tailor your summary headline and experience bullet points to explicitly reflect the target job title and domain responsibilities.")
    if scores.get("similarity", 100) < 50:
        tips.append("Rewrite project bullets to reflect the specific technical methodologies and deliverables described in the job posting.")
    if matched_count and not missing_skills:
        tips.append("All detected job skills are present! Quantify your achievements (metrics, %, performance gains, user scale) to strengthen bullet points.")
    tips.extend(format_problems)
    tips.append("Use a clean single-column layout; avoid multi-column tables, graphics, text boxes, and placing text inside headers/footers.")
    return tips


def analyze_resume(resume_text, job_description, pred_role=None, role_conf=None):
    resume_text = str(resume_text or "")
    job_description = str(job_description or "")
    weights = jd_skill_weights(job_description)
    
    # Compute resume skills once (S10 optimization)
    resume_skills_list = find_skills(resume_text)
    resume_skills = set(resume_skills_list)

    # Exact skill matches
    matched = [s for s in weights if s in resume_skills]

    # Transferable / Related skill matching
    related = {}
    missing_required = []
    missing_preferred = []
    for s in weights:
        if s in resume_skills:
            continue
        cluster_candidates = SKILL_CLUSTERS.get(s, [])
        found_via = [c for c in cluster_candidates if c in resume_skills]
        if found_via:
            related[s] = found_via[0]
        else:
            if weights[s] >= 1.0:
                missing_required.append(s)
            else:
                missing_preferred.append(s)

    extra = [s for s in resume_skills_list if s not in weights]

    # Skill match scoring: 1.0x weight for direct matches, 0.5x partial credit for clustered transferable skills
    if weights:
        earned_pts = sum(weights[s] for s in matched) + sum(weights[s] * 0.5 for s in related)
        skill_score = round(100.0 * earned_pts / sum(weights.values()), 2)
    else:
        skill_score = None

    keywords = jd_keywords(job_description, list(weights))
    kw_score, kw_hit, kw_miss = keyword_coverage(resume_text, keywords)
    if skill_score is None:
        skill_score = kw_score

    sim = text_similarity(resume_text, job_description)
    fmt, fmt_problems = format_score(resume_text)
    role_s = compute_role_alignment(resume_text, job_description, pred_role=pred_role, role_conf=role_conf)

    scores = {
        "skill": skill_score,
        "role": role_s,
        "keyword": kw_score,
        "similarity": sim,
        "format": float(fmt),
    }

    ats = round(max(0.0, min(100.0, float(sum(scores[k] * WEIGHTS[k] for k in WEIGHTS)))), 2)

    related_display = [f"{display_name(s)} (transferable via {display_name(v)})" for s, v in related.items()]
    missing_req_display = [display_name(s) for s in missing_required]
    missing_pref_display = [display_name(s) for s in missing_preferred]
    all_missing_display = missing_req_display + missing_pref_display

    return {
        "ats_score": ats,
        "skill_score": skill_score,
        "role_score": role_s,
        "keyword_score": kw_score,
        "similarity_score": sim,
        "format_score": float(fmt),
        "matched_skills": [display_name(s) for s in matched],
        "related_skills": related_display,
        "missing_skills": all_missing_display,
        "missing_required_skills": missing_req_display,
        "missing_preferred_skills": missing_pref_display,
        "extra_skills": [display_name(s) for s in extra],
        "matched_keywords": kw_hit,
        "missing_keywords": kw_miss,
        "suggestions": generate_suggestions(
            missing_req_display, related_display, kw_miss, scores, fmt_problems, len(matched),
            missing_preferred=missing_pref_display
        ),
        "jd_has_skills": bool(weights),
        "_matched_canon": matched,
        "_related_canon": related,
        "_missing_canon": missing_required + missing_preferred,
        "_resume_skills_canon": resume_skills_list,
    }
