"""Generate an ATS-friendly, single-column LaTeX (Overleaf-ready) resume.

Professional Harvard/Jake's ATS standard formatting:
- Crisp horizontal dividing lines under every section header
- Grouped technical skills with bold topic headers
- Bullet points with clean word formation and bold topic/action leads
- Single-column ATS parseable structure
- Zero hallucination: strictly retains candidate's authentic wording and chronology
"""
import re
import unicodedata

from resume_parser import target_title
from skills_vocab import display_name, find_skills

_UNICODE_MAP = {
    "\u2022": "-", "\u25cf": "-", "\u25aa": "-", "\u25e6": "-", "\u2043": "-", "\u2219": "-",
    "\u2013": "--", "\u2014": "---", "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u00a0": " ", "\u2192": "->", "\u00b7": "-", "\uf0b7": "-",
}

SKILL_CATEGORIES = [
    ("Programming Languages", [
        "python", "java", "javascript", "typescript", "c++", "c#", "c", "go", "rust",
        "php", "ruby", "scala", "kotlin", "swift", "r", "sql", "bash", "matlab", "html", "css"
    ]),
    ("Frameworks & Libraries", [
        "react", "angular", "vue", "next.js", "nuxt.js", "node.js", "express.js", "redux",
        "django", "flask", "fastapi", "spring boot", "spring", "laravel", "ruby on rails",
        ".net", "pandas", "numpy", "scipy", "scikit-learn", "tensorflow", "pytorch",
        "keras", "langchain", "streamlit", "react native", "flutter", "tailwind css",
        "bootstrap", "material ui", "jquery", "svelte"
    ]),
    ("Cloud & DevOps", [
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible", "jenkins",
        "github actions", "ci/cd", "git", "github", "gitlab", "linux", "heroku",
        "digitalocean", "devops"
    ]),
    ("Databases & Storage", [
        "postgresql", "mysql", "mongodb", "redis", "sqlite", "dynamodb", "cockroachdb",
        "elasticsearch", "apache kafka", "rabbitmq", "nosql", "hadoop", "snowflake"
    ]),
    ("Core Competencies & AI", [
        "rest api", "graphql", "grpc", "microservices", "api development", "machine learning",
        "deep learning", "data science", "data analysis", "data visualization",
        "data engineering", "etl", "artificial intelligence", "natural language processing",
        "computer vision", "generative ai", "large language models", "rag", "spark",
        "airflow", "power bi", "tableau", "unit testing", "integration testing", "tdd",
        "pytest", "junit", "selenium", "agile", "jira", "performance tuning",
        "security best practices", "system design", "software development"
    ]),
    ("Business & Management", [
        "financial modeling", "budgeting", "valuation", "accounting", "risk analysis",
        "excel", "recruitment", "talent acquisition", "onboarding", "payroll", "hr policies",
        "employee engagement", "performance management", "seo", "sem", "google analytics",
        "campaign management", "lead generation", "crm", "market research"
    ]),
]


def latex_escape(value):
    """Escape every LaTeX special character and replace non-ASCII characters."""
    value = str(value or "")
    for k, v in _UNICODE_MAP.items():
        value = value.replace(k, v)
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", value)
    table = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
        "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"
    }
    return re.sub(r"\s+", " ", "".join(table.get(ch, ch) for ch in value)).strip()


def categorize_skills(skills_list):
    """Group a flat list of skills into ATS categories with bold headings.
    
    Fixed G3: Uses exact matching and whole-token word boundaries so single-letter
    languages (c, r, go) never match inside words like Excel or Customer Service.
    """
    if not skills_list:
        return {}
    lookup = {}
    for cat_name, items in SKILL_CATEGORIES:
        for it in items:
            lookup[it.lower()] = cat_name

    categories = {cat_name: [] for cat_name, _ in SKILL_CATEGORIES}
    categories["Tools & Methodologies"] = []

    seen = set()
    for s in skills_list:
        s_str = str(s).strip()
        if not s_str:
            continue
        low = s_str.lower()
        if low in seen:
            continue
        seen.add(low)

        cat = lookup.get(low)
        if not cat:
            # Whole word / token boundary check (no partial substring)
            for cat_name, items in SKILL_CATEGORIES:
                matched_item = False
                for it in items:
                    if len(it) <= 2:
                        if low == it:
                            matched_item = True
                            break
                    else:
                        if re.search(r"\b" + re.escape(it) + r"\b", low):
                            matched_item = True
                            break
                if matched_item:
                    cat = cat_name
                    break

        if cat:
            categories[cat].append(s_str)
        else:
            categories["Tools & Methodologies"].append(s_str)

    res = {k: v for k, v in categories.items() if v}
    return res if res else {"Technical Skills": skills_list}


def format_bullet_parts(raw_line):
    """Clean word formation and return (topic, body, has_colon).

    Fixed G5: Does NOT strip numbers from percentages or quantities (e.g. 99.9% uptime).
    """
    line = str(raw_line or "").strip()
    line = re.sub(r"^[\s\u2022\u25cf\u25aa\u25e6\u2023\u2043\u2219\u00b7*>\-\u2013\u2014o]+\s*", "", line)
    # Strip bullet numbering like '1.' or '(i)' without stripping '99.9%' or '5 years'
    line = re.sub(r"^(?:\([0-9ivxIVX]+\)|[0-9ivxIVX]{1,2}[\.\)])\s+", "", line).strip()
    # Resolve hyphenated word wrapping from PDFs
    line = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", line)
    line = re.sub(r"\s+", " ", line).strip()
    if not line:
        return "", "", False

    # Normalize terminal punctuation
    if line.endswith((",", ";", "-", ":")):
        line = line[:-1].strip() + "."
    elif not line.endswith((".", "!", "?")):
        line = line + "."

    line = line[0].upper() + line[1:]

    # Check for existing "Topic: Details" or "Topic - Details"
    m = re.match(r"^([A-Z][A-Za-z0-9\s/&,]{2,35})\s*[:\-–|]\s*(.+)$", line)
    if m:
        topic = m.group(1).strip()
        body = m.group(2).strip()
        if body:
            body = body[0].upper() + body[1:]
        return topic, body, True

    words = line.split()
    if len(words) <= 3:
        return line, "", False

    # Extract leading action phrase before prepositions
    split_idx = -1
    for i in range(2, min(5, len(words))):
        if words[i].lower() in ("using", "with", "for", "by", "to", "in", "across", "through", "on", "via"):
            split_idx = i
            break
    if split_idx == -1:
        split_idx = min(3, len(words) - 1)

    lead = " ".join(words[:split_idx]).strip()
    rest = " ".join(words[split_idx:]).strip()
    return lead, rest, False


def format_bullet_latex(line):
    topic, body, has_colon = format_bullet_parts(line)
    if not topic and not body:
        return ""
    if topic and body:
        colon_str = ":" if has_colon else ""
        return f"  \\item \\textbf{{{latex_escape(topic)}{colon_str}}} {latex_escape(body)}"
    if topic and not body:
        return f"  \\item {latex_escape(topic)}"
    return f"  \\item {latex_escape(body)}"


def format_education_latex(line):
    line = str(line or "").strip()
    line = re.sub(r"^[\s\u2022\u25cf\u25aa\u25e6\u2023\u2043\u2219\u00b7*>\-\u2013\u2014o]+\s*", "", line)
    line = re.sub(r"\s+", " ", line).strip()
    if not line:
        return ""
    if "," in line:
        parts = line.split(",", 1)
        deg = parts[0].strip()
        rest = parts[1].strip()
        return f"  \\item \\textbf{{{latex_escape(deg)}}}, {latex_escape(rest)}"
    return f"  \\item \\textbf{{{latex_escape(line)}}}"


def _comment(value):
    return re.sub(r"\s+", " ", str(value or "")).replace("%", "").replace("\\", "")


def _section(title, body):
    return f"\\resumeSection{{{title}}}\n{body}\n" if body.strip() else ""


def generate_latex_resume(name="Candidate Name", email="", phone="", links=None, summary="", skills=None,
                          education=None, experience=None, projects=None, certifications=None,
                          missing_skills=None, target_role=""):
    """Generates clean ATS-optimized LaTeX resume preserving candidate's authentic wording (Zero Hallucination)."""
    skills = list(dict.fromkeys(skills or []))
    education = [education] if isinstance(education, str) and education else list(education or [])
    experience = [experience] if isinstance(experience, str) and experience else list(experience or [])
    projects = [projects] if isinstance(projects, str) and projects else list(projects or [])
    certifications = list(certifications or [])
    contact = " $|$ ".join(latex_escape(x) for x in [email, phone, *(links or [])] if x)

    missing_comment = ""
    if missing_skills:
        missing_comment = ("% ---------------------------------------------------------------\n"
                           "% Skills requested by the job but NOT found in your resume.\n"
                           "% Add them to the Technical Skills line ONLY if you truly have them:\n"
                           f"% {_comment(', '.join(missing_skills))}\n"
                           "% ---------------------------------------------------------------\n")

    # Grouped skills with bold category labels
    cats = categorize_skills(skills)
    skill_items = []
    for cat_name, cat_skills in cats.items():
        if cat_skills:
            esc_skills = ", ".join(latex_escape(s) for s in cat_skills)
            skill_items.append(f"  \\item \\textbf{{{latex_escape(cat_name)}:}} {esc_skills}")
    skills_body = "\\begin{itemize}[leftmargin=*, label={}]\n" + "\n".join(skill_items) + "\n\\end{itemize}" if skill_items else ""

    # Experience bullets (preserves candidate chronology - G2)
    exp_items = [format_bullet_latex(v) for v in experience if str(v).strip()]
    exp_items = [it for it in exp_items if it]
    exp_body = "\\begin{itemize}\n" + "\n".join(exp_items) + "\n\\end{itemize}" if exp_items else ""

    # Projects bullets with bold topic leads
    proj_items = [format_bullet_latex(v) for v in projects if str(v).strip()]
    proj_items = [it for it in proj_items if it]
    proj_body = "\\begin{itemize}\n" + "\n".join(proj_items) + "\n\\end{itemize}" if proj_items else ""

    # Education with bold degree
    edu_items = [format_education_latex(v) for v in education if str(v).strip()]
    edu_items = [it for it in edu_items if it]
    edu_body = "\\begin{itemize}\n" + "\n".join(edu_items) + "\n\\end{itemize}" if edu_items else ""

    # Certifications
    cert_items = [format_bullet_latex(v) for v in certifications if str(v).strip()]
    cert_items = [it for it in cert_items if it]
    cert_body = "\\begin{itemize}\n" + "\n".join(cert_items) + "\n\\end{itemize}" if cert_items else ""

    parts = [
        _section("Professional Summary", latex_escape(summary)) if summary.strip() else "",
        _section("Technical Skills", skills_body),
        _section("Experience", exp_body),
        _section("Projects", proj_body),
        _section("Education", edu_body),
        _section("Certifications", cert_body),
    ]
    contact_block = f"\\\\\\vspace{{3pt}}\n  {{\\small {contact}}}\n" if contact else "\n"
    return (
        "\\documentclass[10pt,a4paper]{article}\n"
        "\\usepackage[margin=0.6in]{geometry}\n"
        "\\usepackage[T1]{fontenc}\n"
        "\\usepackage{enumitem}\n"
        "\\usepackage[hidelinks]{hyperref}\n"
        "\\usepackage{titlesec}\n"
        "\\pagenumbering{gobble}\n"
        "\\setlength{\\parindent}{0pt}\n"
        "\\setlist[itemize]{leftmargin=1.2em, itemsep=2.5pt, topsep=2pt}\n\n"
        "% Custom ATS Section Header with crisp horizontal dividing line\n"
        "\\newcommand{\\resumeSection}[1]{%\n"
        "  \\vspace{7pt}%\n"
        "  {\\large\\bfseries\\MakeUppercase{#1}}\\par\\vspace{2pt}%\n"
        "  \\hrule height 0.6pt%\n"
        "  \\vspace{5pt}%\n"
        "}\n\n"
        + missing_comment +
        "\\begin{document}\n"
        "\\begin{center}\n"
        f"  {{\\LARGE \\textbf{{{latex_escape(name)}}}}}{contact_block}"
        "\\end{center}\n\n"
        + "\n".join(p for p in parts if p) +
        "\\end{document}\n"
    )
