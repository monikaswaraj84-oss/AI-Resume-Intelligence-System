"""Resume text extraction (PDF/DOCX) and structure parsing.

Hardened against:
- P1: Substring matches treating normal text as headings
- P2: Short inline headings losing their content
- P3: Lowercase 'o' in bullet regex deleting leading letters
- P4: DOCX list paragraphs fusing into one line or swallowing headers
- P5: DOCX headers and tables losing document order
- P6: PDF ligatures (fi, fl) destroyed by ASCII stripping (uses NFKC)
- P7: Password-protected / encrypted PDF error handling
- P8: Name detection picking job titles, locations or dropping accents
- P9: Phone regex matching year chains (2020 - 2022)
- P10: Links keeping trailing punctuation
- P11: DEGREE_KEYWORDS matching English word 'be'
"""
import io
import re
import unicodedata

from docx import Document
from pypdf import PdfReader

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
LINK_RE = re.compile(r"(?:https?://)?(?:www\.)?(?:linkedin\.com/in|github\.com)/[\w\-]+", re.I)

SECTION_ALIASES = {
    "summary": {
        "summary", "professional summary", "profile", "career objective", "objective",
        "about me", "about", "professional profile", "executive summary", "summary of qualifications",
        "personal profile", "profile summary", "career summary", "background", "synopsis", "overview"
    },
    "skills": {
        "skills", "technical skills", "key skills", "core competencies", "skill set",
        "technical proficiency", "technical proficiencies", "technologies", "tools and technologies",
        "core skills", "programming languages", "it skills", "skills and tools", "areas of expertise",
        "technical expertise", "technical skills & tools", "technical stack", "skills & abilities",
        "skills and abilities", "competencies", "expertise", "technical knowledge", "computer skills"
    },
    "experience": {
        "experience", "work experience", "professional experience", "employment history",
        "internships", "internship", "work history", "internship experience", "relevant experience",
        "professional background", "employment", "career history", "work background", "professional journey",
        "work history & experience", "experience & achievements", "industry experience"
    },
    "projects": {
        "projects", "project", "academic projects", "personal projects", "key projects",
        "technical projects", "capstone project", "capstone projects", "selected projects",
        "relevant projects", "project work", "project details", "notable projects", "major projects",
        "mini projects", "development projects", "projects & contributions"
    },
    "education": {
        "education", "academic background", "academic qualifications", "qualifications",
        "educational qualification", "educational background", "education and training",
        "degrees", "academics", "academic details", "academic record", "academic credentials",
        "educational background & qualifications", "education & certifications", "educational details"
    },
    "certifications": {
        "certifications", "certification", "certificates", "courses",
        "licenses and certifications", "achievements", "awards", "achievements and awards",
        "honors and awards", "honors & awards", "publications", "activities",
        "licenses", "credentials", "accomplishments", "co-curricular activities"
    },
}

# Explicit bullet glyphs (P3: lowercase 'o' REMOVED to protect words like 'Optimized', 'Owned', 'Of')
BULLET_RE = re.compile(r"^[\s\u2022\u25cf\u25aa\u25e6\u2023\u2043\u2219\u00b7*>\-\u2013\u2014]+\s*")

# P11: Exact degree acronyms so 'be' or 'before' is never matched
DEGREE_KEYWORDS = re.compile(
    r"\b(b\.e\.|b\.e\b|b\.tech\b|b\.s\b|b\.s\.|m\.tech\b|m\.s\b|m\.s\.|m\.c\.a\b|b\.c\.a\b|b\.sc\b|m\.sc\b|"
    r"ph\.d\b|bachelor|master|degree|diploma|university|college|institute|cgpa|gpa|graduat)\b",
    re.I
)

ACTION_VERBS = re.compile(
    r"^(developed|built|designed|implemented|created|managed|led|analyzed|automated|configured|"
    r"maintained|deployed|engineered|integrated|worked|tested|optimized|collaborated|prepared|supported|"
    r"solved|delivered|conducted|established|spearheaded|orchestrated|executed|utilized|programmed)\b",
    re.I
)


def extract_pdf_text(file_bytes):
    """Extract text from PDF, normalizing ligatures (P6) and handling encryption (P7)."""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        if reader.is_encrypted:
            res = reader.decrypt("")
            # If decrypt fails or reader remains locked
            if res == 0:
                raise ValueError("The PDF is password protected. Please upload an unprotected PDF.")
        parts = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
        raw_text = "\n".join(parts).strip()
        # P6: NFKC normalizes typographic ligatures: 'fi' -> 'fi', 'fl' -> 'fl'
        return unicodedata.normalize("NFKC", raw_text)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Unable to read PDF file: {exc}")


def extract_docx_text(file_bytes):
    """Extract text from DOCX preserving sequential order of paragraphs and tables (P5)."""
    try:
        document = Document(io.BytesIO(file_bytes))
        parts = []

        # 1. Header paragraphs in natural top order (do not reverse)
        for section in document.sections:
            for p in section.header.paragraphs:
                txt = p.text.strip()
                if txt and txt not in parts:
                    parts.append(txt)

        # 2. Main document flow in natural sequence
        for p in document.paragraphs:
            txt = p.text.strip()
            if txt:
                parts.append(txt)

        # 3. Document tables
        for table in document.tables:
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                # Deduplicate identical adjacent cells (merged cells)
                unique_cells = []
                for c in row_cells:
                    if not unique_cells or c != unique_cells[-1]:
                        unique_cells.append(c)
                line = " | ".join(unique_cells)
                if line and line not in parts:
                    parts.append(line)

        raw_text = "\n".join(parts).strip()
        return unicodedata.normalize("NFKC", raw_text)
    except Exception as exc:
        raise ValueError(f"Unable to read DOCX file: {exc}")


def extract_text(filename, file_bytes):
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        return extract_pdf_text(file_bytes)
    if name.endswith(".docx"):
        return extract_docx_text(file_bytes)
    raise ValueError("Only PDF and DOCX files are supported.")


def _heading_key(line):
    """
    P1 fix: Exact match only against section aliases for clean headings.
    Never match arbitrary text lines as headings via substring search.
    """
    cleaned = BULLET_RE.sub("", line).strip()
    cleaned = re.sub(r"^[0-9ivxIVX]+[\.\)\:\-\]\s]+", "", cleaned).strip()
    cleaned = cleaned.rstrip(":").strip()
    low = re.sub(r"[^a-z& ]", " ", cleaned.lower()).replace("&", "and")
    low = re.sub(r"\s+", " ", low).strip()

    # Headings must be concise: at most 4 words and <= 35 characters
    if not low or len(cleaned) > 35 or len(cleaned.split()) > 4:
        return None

    # Exact match only
    for key, names in SECTION_ALIASES.items():
        if low in names:
            return key

    return None


def split_sections(text):
    """Return {section: [lines]} plus 'header' (lines before the first heading)."""
    sections = {k: [] for k in SECTION_ALIASES}
    sections["header"] = []
    current = "header"

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        # P2 fix: Check inline headings FIRST so content like 'Skills: Python, SQL' is never lost
        inline = re.match(r"^([A-Za-z0-9 &]{3,35})\s*[:|]\s*(.+)$", line)
        if inline:
            cand_key = _heading_key(inline.group(1))
            if cand_key:
                current = cand_key
                val = inline.group(2).strip()
                if val:
                    sections[current].append(val)
                continue

        # Standalone heading match
        key = _heading_key(line)
        if key:
            current = key
            continue

        # P3 fix: Bullet check without matching letter 'o'
        is_explicit_bullet = bool(BULLET_RE.match(raw))
        cleaned_bullet = BULLET_RE.sub("", line).strip()
        if not cleaned_bullet:
            continue

        # P4 fix: Clean continuation merging - do not swallow headers or fuse independent bullets
        is_date_or_header = bool(
            re.search(r"\b(?:19|20)\d\d\b", cleaned_bullet)
            or re.match(r"^[A-Z][A-Za-z0-9\s,.-]{2,35}\s*\|\s*", cleaned_bullet)
        )
        is_continuation_word = bool(
            re.match(r"^(and|with|for|using|to|by|in|on|including|enabling|resulting|across|through|which|that|achieving)\b", cleaned_bullet, re.I)
        )

        if (sections[current]
                and not is_explicit_bullet
                and not is_date_or_header
                and current in ("experience", "projects", "summary")
                and (
                    not sections[current][-1].endswith((".", ":", ";"))
                    and (cleaned_bullet[0].islower() or is_continuation_word)
                )):
            sections[current][-1] = (sections[current][-1] + " " + cleaned_bullet).strip()
        else:
            sections[current].append(cleaned_bullet)

    # Fallback: if experience and projects are empty, extract action bullets from header
    if not sections["experience"] and not sections["projects"] and sections["header"]:
        exp_candidates = []
        for l in sections["header"]:
            l_str = l.strip()
            if 20 <= len(l_str) <= 250 and (ACTION_VERBS.search(l_str) or re.search(r"\b(developed|built|using|designed|managed|implemented)\b", l_str, re.I)):
                exp_candidates.append(l_str)
        if exp_candidates:
            sections["experience"] = exp_candidates[:8]

    # Fallback: if education is empty, extract lines with degree/university mentions
    if not sections["education"]:
        edu_candidates = []
        search_lines = sections["header"] + sections.get("certifications", [])
        for l in search_lines:
            if DEGREE_KEYWORDS.search(l) and len(l.strip()) <= 160:
                edu_candidates.append(l.strip())
        if edu_candidates:
            sections["education"] = edu_candidates[:3]

    return sections


def target_title(job_description, fallback="Software Professional"):
    """Extract clean target role title from job description."""
    if not job_description:
        return fallback
    m = re.search(r"(?:job\s*title|position|role)\s*[:\-]\s*([^\n]{3,50})", job_description, re.I)
    if m:
        return m.group(1).strip(" .")
    for line in job_description.splitlines():
        line = line.strip()
        if 3 <= len(line) <= 50 and not line.endswith((".", ":", ";")) and len(line.split()) <= 6:
            # Skip generic intro headers
            if not re.search(r"\b(job|description|about|company|overview|responsibilities|requirements)\b", line, re.I):
                return line
        if line:
            break
    return fallback


def extract_profile(text, sections=None):
    """Extract contact information and candidate profile cleanly (P8, P9, P10)."""
    sections = sections or split_sections(text)
    email = (EMAIL_RE.search(text) or [""])[0] if EMAIL_RE.search(text) else ""

    # P9 fix: Extract phone while rejecting year ranges (e.g. 2020 - 2022)
    phone = ""
    for segment in re.split(r"[|,;\u2022\u00b7\n]|\s{3,}", text):
        clean_seg = segment.strip()
        if not clean_seg:
            continue
        # Reject year chains
        if re.match(r"^(?:19|20)\d\d(?:\s*[-–/]\s*(?:19|20)?\d\d)+$", clean_seg):
            continue
        m = PHONE_RE.search(clean_seg)
        if m:
            digits_only = re.sub(r"\D", "", m.group(0))
            if 10 <= len(digits_only) <= 15:
                phone = m.group(0).strip()
                break

    # P8 fix: Robust name extraction supporting accented characters, rejecting title words
    name = ""
    m_name = re.search(r"^\s*name\s*[:|-]\s*([^\n\r,;|]+)", text, re.I | re.M)
    if m_name:
        cand = m_name.group(1).strip()
        if cand and len(cand.split()) <= 4:
            name = cand

    if not name:
        stop_name_words = {
            "curriculum", "vitae", "resume", "profile", "contact", "engineer", "developer",
            "scientist", "manager", "lead", "analyst", "india", "california", "bangalore",
            "experience", "education", "skills", "projects", "phone", "email", "address"
        }
        header_pool = list(dict.fromkeys((sections.get("header") or []) + text.splitlines()[:6]))
        for line in header_pool:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in re.split(r"[|•·,]|\s{3,}", line) if p.strip()]
            for part in (parts if parts else [line]):
                words = part.split()
                if 1 <= len(words) <= 4:
                    low_words = set(part.lower().split())
                    if not low_words.intersection(stop_name_words):
                        if not re.search(r"[@\d:/\\_]", part):
                            # Accepts accented European/Latin characters
                            if re.fullmatch(r"[A-Za-zÀ-ÖØ-öø-ÿ.' -]+", part):
                                name = part.title() if part.isupper() else part
                                break
            if name:
                break

    # P10 fix: Clean links without trailing punctuation
    raw_links = LINK_RE.findall(text)
    links = [re.sub(r"[.,;:/]+$", "", l).strip() for l in raw_links if l.strip()]

    education = sections["education"][:3]
    return {
        "name": name or "Candidate Name",
        "email": email,
        "phone": phone,
        "links": list(dict.fromkeys(links))[:2],
        "education": education,
    }
