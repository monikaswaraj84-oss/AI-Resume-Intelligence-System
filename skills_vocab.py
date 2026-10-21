"""Skill vocabulary + matcher.

Skills come from (1) a curated technical list with aliases, (2) technical_skills/skills.txt,
(3) the cleaned resume dataset and (4) the job-description dataset. Every skill is mapped
to ONE canonical name so "REST API", "RESTful APIs" and "rest apis" count once.

Hardened against:
- K1: Ambiguity guard was bypassed because canonical name wasn't checked
- K2: 'rest', 'containers', 'excel', 'node', 'express' false positives
- K5: Proper title-casing for GitHub Actions, DevOps, SQLite, etc.
- K6: Efficient nrows=3000 CSV loading
- S3: Job titles (software engineer, data scientist) removed from skills
"""
import ast
import os
import re
from functools import lru_cache

import pandas as pd

from config import CLEAN_RESUME_CSV, JOB_CSV, SKILLS_TXT

# Canonical name -> aliases (S3 fix: role titles are NOT technical skills)
CURATED = {
    "python": [], "java": [], "javascript": ["js", "ecmascript"], "typescript": ["ts"],
    "c++": ["cpp"], "c#": ["csharp", "c sharp"], "c": [], "go": ["golang"], "rust": [],
    "php": [], "ruby": [], "scala": [], "kotlin": [], "swift": [], "r": [], "matlab": [],
    "sql": [], "nosql": [], "html": ["html5"], "css": ["css3"], "sass": [], "bash": ["shell scripting"],
    "react": ["react.js", "reactjs"], "angular": ["angularjs"], "vue": ["vue.js", "vuejs"],
    "svelte": [], "next.js": ["nextjs"], "nuxt.js": ["nuxtjs"], "node.js": ["nodejs", "node"],
    "express.js": ["express", "expressjs"], "redux": [], "jquery": [], "bootstrap": [],
    "tailwind css": ["tailwind"], "material ui": [], "react native": [], "flutter": [],
    "django": [], "flask": [], "fastapi": [], "spring boot": ["springboot"], "spring": ["spring framework"],
    "laravel": [], "ruby on rails": ["rails"], ".net": ["dotnet", ".net core", "asp.net"],
    "rest api": ["rest apis", "restful api", "restful apis", "rest", "restful services", "restful"],
    "graphql": ["graphql apis"], "grpc": [], "microservices": ["microservice", "micro services"],
    "api development": ["api design", "apis"],
    "docker": ["containerization", "containers"], "kubernetes": ["k8s"], "terraform": [],
    "ansible": [], "jenkins": [], "github actions": [], "ci/cd": ["cicd", "ci cd", "continuous integration",
                                                                   "continuous delivery", "continuous deployment"],
    "git": [], "github": [], "gitlab": [], "linux": ["unix"], "aws": ["amazon web services"],
    "azure": ["microsoft azure"], "gcp": ["google cloud", "google cloud platform"],
    "heroku": [], "digitalocean": [], "firebase": [],
    "mysql": [], "postgresql": ["postgres"], "mongodb": ["mongo"], "redis": [], "sqlite": [],
    "dynamodb": [], "cockroachdb": [], "elasticsearch": [], "apache kafka": ["kafka"], "rabbitmq": [],
    "pandas": [], "numpy": [], "scipy": [], "matplotlib": [], "seaborn": [],
    "scikit-learn": ["sklearn", "scikit learn"], "tensorflow": [], "pytorch": [], "keras": [],
    "machine learning": ["ml"], "deep learning": [], "data science": [],
    "data analysis": ["data analytics"], "data visualization": ["data visualisation"],
    "data engineering": [], "etl": [], "statistics": ["statistical analysis", "statistical modeling"],
    "artificial intelligence": ["ai"], "natural language processing": ["nlp"],
    "computer vision": [], "generative ai": ["genai"], "large language models": ["llm", "llms"],
    "rag": ["retrieval augmented generation"], "langchain": [], "streamlit": [],
    "spark": ["apache spark", "pyspark"], "hadoop": [], "airflow": ["apache airflow"],
    "power bi": ["powerbi"], "tableau": [], "excel": ["microsoft excel", "ms excel"],
    "object oriented programming": ["oop", "oops", "object-oriented programming", "object oriented design"],
    "data structures": [], "algorithms": [], "software development": ["software engineering"],
    "devops": [],
    "unit testing": [], "integration testing": [], "tdd": ["test driven development"],
    "pytest": [], "junit": [], "selenium": [], "agile": ["agile methodologies", "scrum"],
    "jira": [], "performance tuning": ["performance optimization"], "security best practices": [],
    "system design": [], "scaling": [], "monolithic apps": [],
    # Business / non-tech domains
    "recruitment": ["recruiting"], "talent acquisition": [], "onboarding": [], "payroll": [],
    "hr policies": [], "employee engagement": [], "performance management": [], "compliance": [],
    "market research": [], "campaign management": [], "google analytics": [], "social media": [],
    "seo": ["search engine optimization"], "sem": [], "branding": [], "lead generation": [],
    "crm": ["customer relationship management"], "content strategy": [],
    "financial modeling": ["financial modelling"], "budgeting": [], "forecasting": [],
    "accounting": [], "valuation": [], "risk analysis": ["risk assessment"],
    "communication": ["communication skills"], "leadership": [], "project management": [],
    "problem solving": ["problem-solving"], "teamwork": ["team player", "team collaboration"],
}

# Ambiguous tokens requiring strict semantic validation
AMBIGUOUS = {
    "c", "r", "go", "rest", "node", "express", "spring", "ml", "ai", "ts", "js",
    "sem", "rag", "apis", "containers", "scaling", "collaboration", "excel"
}

NOISE_SKILLS = {
    "art", "design", "drawing", "construction", "training", "interview", "automotive", "business services",
    "development tools", "databases", "company research", "baseball", "basketball", "soccer", "football", "golf",
    "swimming", "volleyball", "gymnastics", "diving", "email", "energy", "gas", "grocery", "healthcare",
    "history", "hospitals", "insurance", "integration", "journals", "kindergarten", "newsletters", "nutrition",
    "outsourcing", "partnerships", "presentations", "queues", "quotations", "range", "records", "regulations",
    "research", "retail", "reviews", "schedules", "software", "strategy", "teaching", "telecommunications",
    "testing", "validation", "vendors", "video", "volunteering", "windows", "workshops", "writing", "editing",
    "electronics", "finance", "graphics", "investments", "journalism", "facilitation", "fundraising",
    "automation", "interviewing",
}


def normalize(text):
    """Lower-case + simplify punctuation so matching is stable."""
    text = str(text or "").lower()
    text = text.replace("&", " and ").replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"[-/_]", " ", text)
    text = re.sub(r"[^a-z0-9+#.\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _alias_entries():
    entries = {}
    for canon, aliases in CURATED.items():
        for a in [canon] + aliases:
            entries[normalize(a)] = canon

    extras = []
    if os.path.exists(SKILLS_TXT):
        try:
            with open(SKILLS_TXT, "r", encoding="utf-8", errors="ignore") as f:
                extras.extend(line.strip() for line in f if line.strip())
        except Exception:
            pass

    if os.path.exists(CLEAN_RESUME_CSV):
        try:
            df = pd.read_csv(CLEAN_RESUME_CSV, usecols=["Skills"], nrows=1500)
            for raw in df["Skills"].dropna():
                if raw.startswith("["):
                    try:
                        extras.extend(ast.literal_eval(raw))
                    except Exception:
                        pass
                else:
                    extras.extend(raw.split(","))
        except Exception:
            pass

    # K6 fix: use nrows=3000 to avoid loading the full 4.6 MB file
    if os.path.exists(JOB_CSV):
        try:
            df = pd.read_csv(JOB_CSV, usecols=["skills"], nrows=3000)
            for raw in df["skills"].dropna():
                extras.extend(str(raw).split(","))
        except Exception:
            pass

    for s in extras:
        norm = normalize(s)
        if norm and len(norm) >= 2 and norm not in NOISE_SKILLS and norm not in entries:
            entries[norm] = norm

    return entries


@lru_cache(maxsize=1)
def _matcher():
    entries = _alias_entries()
    keys = sorted(entries, key=lambda k: (-len(k), k))
    parts = [re.escape(k).replace(r"\ ", r"\s+") for k in keys]
    regex = re.compile(r"(?<![a-z0-9+#.])(" + "|".join(parts) + r")(?![a-z0-9+#]|\.[a-z0-9])")
    return regex, entries


def display_name(canon):
    """P10/K5: Map canonical skill to proper industry casing."""
    special = {
        "aws": "AWS", "gcp": "GCP", "sql": "SQL", "html": "HTML", "css": "CSS", "nosql": "NoSQL",
        "mysql": "MySQL", "postgresql": "PostgreSQL", "mongodb": "MongoDB", "c++": "C++",
        "c#": "C#", "ci/cd": "CI/CD", "node.js": "Node.js", "next.js": "Next.js",
        "nuxt.js": "Nuxt.js", "express.js": "Express.js", ".net": ".NET", "seo": "SEO",
        "sem": "SEM", "crm": "CRM", "etl": "ETL", "llm": "LLM", "rag": "RAG", "tdd": "TDD",
        "oop": "OOP", "api": "API", "rest api": "REST API", "graphql": "GraphQL",
        "fastapi": "FastAPI", "pytorch": "PyTorch", "tensorflow": "TensorFlow",
        "scikit-learn": "Scikit-Learn", "numpy": "NumPy", "github": "GitHub", "gitlab": "GitLab",
        "github actions": "GitHub Actions", "devops": "DevOps", "sqlite": "SQLite",
        "javascript": "JavaScript", "typescript": "TypeScript", "power bi": "Power BI",
        "hr policies": "HR Policies", "grpc": "gRPC", "dynamodb": "DynamoDB",
        "cockroachdb": "CockroachDB", "rabbitmq": "RabbitMQ", "jquery": "jQuery",
        "large language models": "Large Language Models", "langchain": "LangChain",
        "natural language processing": "Natural Language Processing",
        "object oriented programming": "Object-Oriented Programming", "digitalocean": "DigitalOcean",
        "google analytics": "Google Analytics", "apache kafka": "Apache Kafka",
        "react": "React", "react native": "React Native", "tailwind css": "Tailwind CSS",
        "spring boot": "Spring Boot", "ruby on rails": "Ruby on Rails",
        "vue": "Vue.js", "go": "Go", "r": "R", "c": "C", "kubernetes": "Kubernetes",
        "api development": "API Development",
    }
    return special.get(canon, canon.title() if " " in canon else canon.capitalize())


def find_skills(text):
    """Return canonical skills in order of first appearance (no duplicates)."""
    regex, entries = _matcher()
    norm = normalize(text)
    found = []
    for m in regex.finditer(norm):
        token = m.group(1)
        canon = entries.get(re.sub(r"\s+", " ", token))
        if canon is None:
            continue
        # K1 fix: Guard runs on the matched ambiguous token directly
        if token in AMBIGUOUS:
            if not _ambiguous_ok(token, text):
                continue
        if canon not in found:
            found.append(canon)
    return found


def _ambiguous_ok(token, original):
    """K1/K2 fix: Validate that an ambiguous token is used in genuine technical context."""
    o = str(original or "")
    if token in {"c", "r"}:
        return bool(
            re.search(rf"(?<![A-Za-z0-9]){token.upper()}(?![A-Za-z0-9+#])\s*[,;|/]|[,;|:]\s*{token.upper()}(?![A-Za-z0-9+#])", o) or
            re.search(rf"\b{token.upper()}\s+(?:programming|language|scripting|script|code|developer)\b", o, re.I)
        )
    if token == "go":
        return bool(re.search(r"\bGo(lang)?\b", o)) and bool(
            re.search(r"\b(?:golang|Go)\b\s*[,;|]|[,;|:]\s*Go\b|\bGo\s+(?:developer|programming|language)\b", o)
        )
    if token == "rest":
        # Must be uppercase REST or accompanied by api/services/endpoints, NOT 'the rest of'
        if re.search(r"\bthe\s+rest\s+of\b", o, re.I):
            return False
        return bool(
            re.search(r"\bREST\b") or
            re.search(r"\brest\s*(?:ful)?\s*(?:apis?|services?|endpoints?|architecture|framework)\b", o, re.I) or
            re.search(r"[,;|]\s*rest\s*[,;|]", o, re.I)
        )
    if token in {"node", "express"}:
        return bool(
            re.search(rf"\b{token}(?:\.js|\s*js)\b", o, re.I) or
            re.search(rf"\b{token}\b\s*(?:framework|server|backend|api)\b", o, re.I) or
            re.search(rf"[,;|]\s*{token}\s*[,;|]", o, re.I)
        )
    if token == "spring":
        return bool(re.search(r"\bspring\s*(?:boot|framework|mvc|cloud|security)\b", o, re.I) or
                   re.search(r"\b(?:java|j2ee)\s*[,/]\s*spring\b", o, re.I))
    if token == "containers":
        # Reject 'shipping containers', accept technical containerization
        if re.search(r"\b(?:shipping|freight|cargo|box|storage)\s+containers\b", o, re.I):
            return False
        return bool(re.search(r"\b(?:docker|kubernetes|linux|software|app|cloud)\s+containers\b", o, re.I) or
                   re.search(r"\bcontainers?\s+(?:management|orchestration|deployment|virtualization)\b", o, re.I))
    if token == "excel":
        # Reject 'I excel at', accept Microsoft Excel / spreadsheets
        if re.search(r"\b(?:i|to|we|they|will)\s+excel\s+(?:at|in)\b", o, re.I):
            return False
        return bool(
            re.search(r"\b(?:ms|microsoft)\s*excel\b", o, re.I) or
            re.search(r"\bexcel\s*(?:sheets?|formulas?|vlookup|spreadsheets?|macros?|pivot|dashboards?)\b", o, re.I) or
            re.search(r"\b(?:excel|word|powerpoint)\b", o, re.I) or
            re.search(r"[,;|]\s*excel\s*[,;|]", o, re.I)
        )
    if token == "collaboration":
        return bool(re.search(r"\b(?:team|cross-functional)\s+collaboration\b", o, re.I) or
                   re.search(r"\bcollaboration\s*(?:tools?|skills?)\b", o, re.I))
    if token in {"ml", "ai", "ts", "js", "sem", "rag"}:
        return bool(re.search(rf"\b{token.upper()}\b", o))
    return True
