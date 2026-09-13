#!/usr/bin/env python3

"""
Knowledge Graph Researcher Ranking Pipeline

This single-file script:

1. Collects publications from DBLP.
2. Filters publications by year.
3. Detects KG-related publications using:
   - KG-related conference names
   - KG-related keywords in publication titles
4. Assigns a configurable weight to each conference.
5. Gives the complete paper weight to every co-author.
6. Aggregates researcher scores.
7. Enriches researchers with DBLP profile information.
8. Generates an HTML ranking report.
9. Uses a local cache to reduce repeated DBLP requests.

Run:

    python kg_ranking.py

Examples:

    python kg_ranking.py --start-year 2021 --end-year 2026

    python kg_ranking.py --start-year 2021 --end-year 2026 --top-n 100

    python kg_ranking.py --no-cache

Output:

    output/kg_rankings.html
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
import unicodedata

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup
from jinja2 import Template


# ============================================================
# BASIC CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

OUTPUT_DIR = PROJECT_ROOT / "output"
CACHE_DIR = PROJECT_ROOT / "cache"

OUTPUT_HTML = OUTPUT_DIR / "kg_rankings.html"

DEFAULT_YEAR_RANGE = 5
DEFAULT_TOP_N = 200
DEFAULT_RECENT_PUBLICATIONS_LIMIT = 10

REQUEST_TIMEOUT = 30
DBLP_BATCH_SIZE = 1000

USE_CACHE_BY_DEFAULT = True

# If a venue is not listed in CONFERENCE_WEIGHTS,
# this default weight will be used.
DEFAULT_CONFERENCE_WEIGHT = 1.0


# ============================================================
# CONFERENCE-SPECIFIC WEIGHTS
# ============================================================
#
# Every paper receives the weight of its conference.
#
# If all papers should count equally, keep every value as 1.0.
#
# You can change each conference independently later.
#
# Example:
#
#     "ISWC": 3.0,
#     "ESWC": 2.0,
#     "WWW": 1.5,
#
# A paper from ISWC will contribute 3.0 points to every author.
# A paper from ESWC will contribute 2.0 points to every author.
#

CONFERENCE_WEIGHTS = {
    # Knowledge Graph and Semantic Web
    "ISWC": 1.0,
    "ESWC": 1.0,
    "K-CAP": 1.0,
    "KR": 1.0,
    "RULEML+RR": 1.0,

    # Web and Information Retrieval
    "WWW": 1.0,
    "SIGIR": 1.0,
    "WSDM": 1.0,
    "CIKM": 1.0,

    # Data Mining and Databases
    "KDD": 1.0,
    "ICDM": 1.0,
    "PAKDD": 1.0,
    "ICDE": 1.0,
    "EDBT": 1.0,
    "DASFAA": 1.0,
    "ASONAM": 1.0,

    # Artificial Intelligence and Machine Learning
    "AAAI": 1.0,
    "IJCAI": 1.0,
    "ECAI": 1.0,
    "UAI": 1.0,
    "ICML": 1.0,
    "ICLR": 1.0,
    "NEURIPS": 1.0,
    "NIPS": 1.0,
    "AISTATS": 1.0,
    "ECML": 1.0,

    # Natural Language Processing
    "ACL": 1.0,
    "EMNLP": 1.0,
    "COLING": 1.0,
    "EACL": 1.0,
    "NAACL": 1.0,
    "AACL": 1.0,
    "IJCNLP": 1.0,
    "ICWSM": 1.0,
}


# ============================================================
# KG-RELATED CONFERENCES
# ============================================================

KG_VENUES = {
    # Knowledge Graph and Semantic Web
    "ISWC",
    "ESWC",
    "K-CAP",
    "KR",
    "RULEML+RR",

    # Web and Information Retrieval
    "WWW",
    "SIGIR",
    "WSDM",
    "CIKM",

    # Data Mining and Databases
    "KDD",
    "ICDM",
    "PAKDD",
    "ICDE",
    "EDBT",
    "DASFAA",
    "ASONAM",

    # Artificial Intelligence and Machine Learning
    "AAAI",
    "IJCAI",
    "ECAI",
    "UAI",
    "ICML",
    "ICLR",
    "NEURIPS",
    "NIPS",
    "AISTATS",
    "ECML",

    # Natural Language Processing
    "ACL",
    "EMNLP",
    "COLING",
    "EACL",
    "NAACL",
    "AACL",
    "IJCNLP",
    "ICWSM",
}


# ============================================================
# VENUE ALIASES
# ============================================================
#
# DBLP may use different names for the same conference.
# These aliases convert them into a common canonical name.
#

CONFERENCE_ALIASES = {
    "WWW": "WWW",
    "THE WEB CONFERENCE": "WWW",
    "INTERNATIONAL WORLD WIDE WEB CONFERENCE": "WWW",
    "WORLD WIDE WEB CONFERENCE": "WWW",

    "ISWC": "ISWC",
    "INTERNATIONAL SEMANTIC WEB CONFERENCE": "ISWC",

    "ESWC": "ESWC",
    "EUROPEAN SEMANTIC WEB CONFERENCE": "ESWC",

    "K CAP": "K-CAP",
    "K-CAP": "K-CAP",
    "KNOWLEDGE CAPTURE": "K-CAP",

    "RULEML": "RULEML+RR",
    "RULEML RR": "RULEML+RR",
    "RULEML+RR": "RULEML+RR",

    "NEURIPS": "NEURIPS",
    "NIPS": "NIPS",
    "NEURAL INFORMATION PROCESSING SYSTEMS": "NEURIPS",

    "AAAI": "AAAI",
    "IJCAI": "IJCAI",
    "ECAI": "ECAI",
    "UAI": "UAI",
    "ICML": "ICML",
    "ICLR": "ICLR",
    "AISTATS": "AISTATS",
    "ECML": "ECML",

    "ACL": "ACL",
    "EMNLP": "EMNLP",
    "COLING": "COLING",
    "EACL": "EACL",
    "NAACL": "NAACL",
    "AACL": "AACL",
    "IJCNLP": "IJCNLP",
    "ICWSM": "ICWSM",

    "SIGIR": "SIGIR",
    "WSDM": "WSDM",
    "CIKM": "CIKM",
    "KDD": "KDD",
    "ICDM": "ICDM",
    "PAKDD": "PAKDD",
    "ICDE": "ICDE",
    "EDBT": "EDBT",
    "DASFAA": "DASFAA",
    "ASONAM": "ASONAM",
    "KR": "KR",
}


# ============================================================
# KG KEYWORDS
# ============================================================
#
# These keywords are mainly matched against publication titles.
#
# Be careful with broad terms such as:
# - graph embedding
# - information extraction
# - RAG
#
# They can include papers that are not specifically about
# Knowledge Graphs.
#

KG_KEYWORDS = {
    # Core Knowledge Graph concepts
    "knowledge graph",
    "knowledge graphs",
    "knowledge base",
    "knowledge bases",
    "knowledge representation",
    "knowledge reasoning",
    "semantic web",
    "linked data",
    "linked open data",
    "web of data",
    "graph data",
    "graph-based knowledge",

    # RDF and Semantic Web
    "rdf",
    "rdfs",
    "owl",
    "sparql",
    "shacl",
    "ontology",
    "ontologies",
    "ontology learning",
    "ontology matching",
    "ontology alignment",
    "ontology engineering",
    "semantic annotation",
    "semantic reasoning",
    "description logic",

    # KG construction and completion
    "knowledge graph construction",
    "knowledge graph completion",
    "knowledge graph population",
    "knowledge graph refinement",
    "knowledge graph curation",
    "knowledge graph extraction",
    "knowledge graph induction",
    "knowledge graph generation",
    "knowledge graph augmentation",
    "knowledge graph building",

    "knowledge base completion",
    "knowledge base construction",
    "knowledge base population",
    "knowledge base refinement",

    # KG embeddings and representation learning
    "knowledge graph embedding",
    "knowledge graph embeddings",
    "knowledge graph representation",
    "knowledge representation learning",
    "relational embedding",
    "relational embeddings",
    "relation prediction",
    "link prediction",
    "entity prediction",
    "graph embedding",

    # Entity-related tasks
    "entity linking",
    "entity disambiguation",
    "entity alignment",
    "entity resolution",
    "entity matching",
    "cross-lingual entity",
    "entity typing",
    "entity discovery",
    "entity representation",

    # Relation and information extraction
    "relation extraction",
    "relation learning",
    "open information extraction",
    "open ie",
    "information extraction",
    "triple extraction",
    "fact extraction",
    "fact checking",
    "fact verification",
    "knowledge extraction",

    # Reasoning
    "knowledge graph reasoning",
    "knowledge base reasoning",
    "multi-hop reasoning",
    "logical reasoning",
    "neural-symbolic",
    "neuro-symbolic",
    "symbolic reasoning",
    "rule reasoning",
    "rule learning",
    "automated reasoning",
    "inductive reasoning",
    "deductive reasoning",
    "abductive reasoning",
    "commonsense reasoning",

    # Question answering
    "knowledge graph question answering",
    "knowledge graph question-answering",
    "knowledge base question answering",
    "knowledge base question-answering",
    "kgqa",
    "kbqa",
    "semantic parsing",
    "question answering over knowledge graphs",
    "question answering over knowledge bases",

    # Retrieval and generation
    "knowledge graph retrieval",
    "knowledge graph search",
    "knowledge-aware retrieval",
    "knowledge-enhanced retrieval",
    "knowledge-grounded",
    "knowledge augmented",
    "knowledge-augmented",
    "graph retrieval",
    "retrieval augmented generation",
    "retrieval-augmented generation",
    "rag",

    # Common KG datasets and resources
    "wikidata",
    "dbpedia",
    "freebase",
    "yago",
    "nell",
    "conceptnet",
    "wordnet",
    "babelnet",
    "geonames",
    "schema.org",

    # Types of knowledge graphs
    "temporal knowledge graph",
    "temporal knowledge graphs",
    "dynamic knowledge graph",
    "dynamic knowledge graphs",
    "multilingual knowledge graph",
    "multilingual knowledge graphs",
    "cross-lingual knowledge graph",
    "cross-lingual knowledge graphs",
    "multimodal knowledge graph",
    "multimodal knowledge graphs",
    "probabilistic knowledge graph",
    "probabilistic knowledge graphs",
    "event knowledge graph",
    "event knowledge graphs",
    "biomedical knowledge graph",
    "biomedical knowledge graphs",
    "enterprise knowledge graph",
    "enterprise knowledge graphs",
    "domain knowledge graph",
    "domain-specific knowledge graph",
}


# ============================================================
# DBLP API ENDPOINTS
# ============================================================

DBLP_SEARCH_API = "https://dblp.org/search/publ/api"
DBLP_AUTHOR_API = "https://dblp.org/search/author/api"
DBLP_PERSON_API = "https://dblp.org/pid"


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

LOGGER = logging.getLogger("kg-ranking")


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class Publication:
    title: str
    venue: str
    year: int
    url: str = ""
    key: str = ""
    authors: List[Dict[str, str]] = field(default_factory=list)
    weight: float = 1.0
    relevance_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "venue": self.venue,
            "year": self.year,
            "url": self.url,
            "key": self.key,
            "authors": self.authors,
            "weight": self.weight,
            "relevance_reason": self.relevance_reason,
        }


@dataclass
class Author:
    pid: str
    name: str
    score: float = 0.0
    publication_count: int = 0
    publications: List[Dict[str, Any]] = field(default_factory=list)
    venues: Dict[str, float] = field(default_factory=dict)
    years: Dict[str, float] = field(default_factory=dict)
    affiliations: List[str] = field(default_factory=list)
    homepage_urls: List[str] = field(default_factory=list)
    dblp_url: str = ""
    profile_url: str = ""
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pid": self.pid,
            "name": self.name,
            "score": round(self.score, 3),
            "publication_count": self.publication_count,
            "publications": self.publications,
            "venues": self.venues,
            "years": self.years,
            "affiliations": self.affiliations,
            "homepage_urls": self.homepage_urls,
            "dblp_url": self.dblp_url,
            "profile_url": self.profile_url,
            "rank": self.rank,
        }


# ============================================================
# HTML TEMPLATE
# ============================================================

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>Knowledge Graph Researcher Rankings</title>

    <style>
        :root {
            --background: #f5f7fb;
            --surface: #ffffff;
            --border: #dfe4ea;
            --text: #1f2937;
            --muted: #6b7280;
            --primary: #2563eb;
            --primary-light: #eff6ff;
            --success: #047857;
        }

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            padding: 32px;
            background: var(--background);
            color: var(--text);
            font-family:
                Arial,
                Helvetica,
                sans-serif;
        }

        .container {
            max-width: 1400px;
            margin: auto;
        }

        h1 {
            margin: 0 0 8px;
        }

        .subtitle {
            color: var(--muted);
            margin-bottom: 24px;
        }

        .summary {
            display: grid;
            grid-template-columns:
                repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }

        .summary-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 18px;
        }

        .summary-label {
            color: var(--muted);
            font-size: 13px;
        }

        .summary-value {
            font-size: 28px;
            font-weight: 700;
            margin-top: 6px;
        }

        .researcher {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            margin-bottom: 18px;
            padding: 22px;
        }

        .researcher-header {
            display: flex;
            justify-content: space-between;
            gap: 16px;
            flex-wrap: wrap;
        }

        .rank {
            color: var(--primary);
            font-size: 22px;
            font-weight: 800;
        }

        .name {
            font-size: 22px;
            font-weight: 700;
        }

        .name a {
            color: var(--text);
            text-decoration: none;
        }

        .name a:hover {
            color: var(--primary);
        }

        .metrics {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 12px;
        }

        .metric {
            background: var(--primary-light);
            border-radius: 8px;
            color: var(--primary);
            padding: 7px 10px;
            font-size: 13px;
            font-weight: 600;
            text-decoration: none;
        }

        .section {
            margin-top: 18px;
        }

        .section-title {
            font-size: 14px;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .tags {
            display: flex;
            gap: 7px;
            flex-wrap: wrap;
        }

        .tag {
            background: #f3f4f6;
            border-radius: 6px;
            color: #374151;
            font-size: 12px;
            padding: 5px 8px;
        }

        .publication {
            border-top: 1px solid var(--border);
            padding: 12px 0;
        }

        .publication:first-child {
            border-top: 0;
        }

        .publication-title {
            font-weight: 600;
        }

        .publication-title a {
            color: var(--primary);
            text-decoration: none;
        }

        .publication-meta {
            color: var(--muted);
            font-size: 13px;
            margin-top: 4px;
        }

        .weight {
            color: var(--success);
            font-weight: 700;
        }

        .footer {
            color: var(--muted);
            font-size: 13px;
            margin-top: 32px;
        }

        @media (max-width: 700px) {
            body {
                padding: 16px;
            }

            .name {
                font-size: 19px;
            }
        }
    </style>
</head>

<body>
<div class="container">

    <h1>Knowledge Graph Researcher Rankings</h1>

    <div class="subtitle">
        DBLP-based ranking for the period
        {{ start_year }}–{{ end_year }}.
    </div>

    <div class="summary">
        <div class="summary-card">
            <div class="summary-label">Researchers</div>
            <div class="summary-value">{{ authors|length }}</div>
        </div>

        <div class="summary-card">
            <div class="summary-label">KG publications</div>
            <div class="summary-value">{{ publication_count }}</div>
        </div>

        <div class="summary-card">
            <div class="summary-label">Total weighted score</div>
            <div class="summary-value">{{ total_score }}</div>
        </div>

        <div class="summary-card">
            <div class="summary-label">Generated</div>
            <div class="summary-value" style="font-size: 18px;">
                {{ generated_at }}
            </div>
        </div>
    </div>

    {% for author in authors %}
    <article class="researcher">

        <div class="researcher-header">
            <div>
                <div class="rank">#{{ author.rank }}</div>

                <div class="name">
                    {% if author.profile_url %}
                    <a
                        href="{{ author.profile_url }}"
                        target="_blank"
                        rel="noopener noreferrer"
                    >
                        {{ author.name }}
                    </a>
                    {% else %}
                        {{ author.name }}
                    {% endif %}
                </div>
            </div>

            <div class="metrics">
                <span class="metric">
                    Score: {{ "%.2f"|format(author.score) }}
                </span>

                <span class="metric">
                    Papers: {{ author.publication_count }}
                </span>

                {% if author.dblp_url %}
                <a
                    class="metric"
                    href="{{ author.dblp_url }}"
                    target="_blank"
                    rel="noopener noreferrer"
                >
                    DBLP Profile
                </a>
                {% endif %}
            </div>
        </div>

        {% if author.affiliations %}
        <div class="section">
            <div class="section-title">Affiliations</div>

            <div class="tags">
                {% for affiliation in author.affiliations %}
                <span class="tag">
                    {{ affiliation }}
                </span>
                {% endfor %}
            </div>
        </div>
        {% endif %}

        <div class="section">
            <div class="section-title">Conference scores</div>

            <div class="tags">
                {% for venue, score in author.venues|dictsort %}
                <span class="tag">
                    {{ venue }}:
                    {{ "%.2f"|format(score) }}
                </span>
                {% endfor %}
            </div>
        </div>

        <div class="section">
            <div class="section-title">Publication years</div>

            <div class="tags">
                {% for year, score in author.years|dictsort %}
                <span class="tag">
                    {{ year }}:
                    {{ "%.2f"|format(score) }}
                </span>
                {% endfor %}
            </div>
        </div>

        <div class="section">
            <div class="section-title">
                KG-related publications
            </div>

            {% for publication in author.publications[:recent_limit] %}
            <div class="publication">

                <div class="publication-title">
                    {% if publication.url %}
                    <a
                        href="{{ publication.url }}"
                        target="_blank"
                        rel="noopener noreferrer"
                    >
                        {{ publication.title }}
                    </a>
                    {% else %}
                        {{ publication.title }}
                    {% endif %}
                </div>

                <div class="publication-meta">
                    {{ publication.venue }}
                    ·
                    {{ publication.year }}
                    ·
                    <span class="weight">
                        Weight: {{ publication.weight }}
                    </span>
                    ·
                    {{ publication.relevance_reason }}
                </div>

            </div>
            {% endfor %}
        </div>

    </article>
    {% endfor %}

    <div class="footer">
        Each co-author receives the complete weight of the paper.
        Scores are based on DBLP metadata and title-based KG detection.
    </div>

</div>
</body>
</html>
"""


# ============================================================
# DIRECTORY AND TEXT UTILITIES
# ============================================================

def ensure_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(value: Any) -> str:
    """
    Convert text into a normalized form for matching.
    """
    if value is None:
        return ""

    value = str(value)
    value = unicodedata.normalize("NFKC", value)
    value = value.lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[-_/]+", " ", value)
    value = re.sub(r"[^\w\s+]", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def normalize_venue(venue: str) -> str:
    """
    Convert DBLP venue names to canonical conference names.
    """
    normalized = normalize_text(venue).upper()

    if normalized in CONFERENCE_ALIASES:
        return CONFERENCE_ALIASES[normalized]

    if "WORLD WIDE WEB" in normalized:
        return "WWW"

    if "WEB CONFERENCE" in normalized:
        return "WWW"

    if "NEURAL INFORMATION PROCESSING SYSTEM" in normalized:
        return "NEURIPS"

    if "EUROPEAN SEMANTIC WEB" in normalized:
        return "ESWC"

    if "SEMANTIC WEB" in normalized:
        return "ISWC"

    if "KNOWLEDGE CAPTURE" in normalized:
        return "K-CAP"

    if "RULEML" in normalized:
        return "RULEML+RR"

    return normalized


def clean_title(title: str) -> str:
    return re.sub(r"\s+", " ", title or "").strip()


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def deduplicate(values: Iterable[str]) -> List[str]:
    result = []
    seen = set()

    for value in values:
        value = value.strip()

        if not value:
            continue

        key = value.lower()

        if key not in seen:
            seen.add(key)
            result.append(value)

    return result


def make_cache_file(url: str) -> Path:
    digest = hashlib.sha256(
        url.encode("utf-8")
    ).hexdigest()

    return CACHE_DIR / f"{digest}.json"


# ============================================================
# HTTP CACHE
# ============================================================

def cached_get(
    session: requests.Session,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    use_cache: bool = True,
) -> Optional[requests.Response]:
    """
    GET request with a local JSON cache.
    """
    full_url = url

    if params:
        full_url = (
            f"{url}?{urlencode(sorted(params.items()))}"
        )

    cache_file = make_cache_file(full_url)

    if use_cache and cache_file.exists():
        try:
            cached_data = json.loads(
                cache_file.read_text(encoding="utf-8")
            )

            response = requests.Response()
            response.status_code = cached_data["status_code"]
            response.url = cached_data["url"]
            response._content = cached_data["content"].encode(
                "utf-8"
            )
            response.headers["Content-Type"] = (
                cached_data.get(
                    "content_type",
                    "application/json",
                )
            )

            return response

        except Exception as exc:
            LOGGER.warning(
                "Could not read cache %s: %s",
                cache_file,
                exc,
            )

    headers = {
        "User-Agent": (
            "KGResearcherRanking/1.0 "
            "(academic research; contact: your-email@example.com)"
        ),
        "Accept": (
            "application/json, "
            "application/xml, "
            "text/xml, "
            "text/html"
        ),
    }

    try:
        response = session.get(
            url,
            params=params,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        if use_cache:
            cache_file.write_text(
                json.dumps(
                    {
                        "status_code": response.status_code,
                        "url": response.url,
                        "content_type": response.headers.get(
                            "Content-Type",
                            "",
                        ),
                        "content": response.text,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        # Avoid sending requests too quickly.
        time.sleep(0.2)

        return response

    except requests.RequestException as exc:
        LOGGER.error(
            "Request failed for %s: %s",
            full_url,
            exc,
        )
        return None


# ============================================================
# DBLP SEARCH FUNCTIONS
# ============================================================

def parse_dblp_search_response(
    data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    result = data.get("result", {})
    hits = result.get("hits", {})
    hit_list = hits.get("hit", [])

    if isinstance(hit_list, dict):
        hit_list = [hit_list]

    return hit_list or []


def extract_publication_from_hit(
    hit: Dict[str, Any],
) -> Optional[Publication]:
    info = hit.get("info", {})

    title = clean_title(info.get("title", ""))

    venue = clean_title(
        info.get("venue")
        or info.get("journal")
        or info.get("booktitle")
        or ""
    )

    year = safe_int(info.get("year"))

    if not title or not year:
        return None

    authors = []

    raw_authors = (
        info.get("authors", {})
        .get("author", [])
    )

    if isinstance(raw_authors, dict):
        raw_authors = [raw_authors]

    for author in raw_authors:
        if isinstance(author, str):
            authors.append(
                {
                    "name": author,
                    "pid": "",
                }
            )
        else:
            authors.append(
                {
                    "name": author.get("text", ""),
                    "pid": author.get("@pid", ""),
                }
            )

    return Publication(
        title=title,
        venue=venue,
        year=year,
        url=info.get("ee", "")
        or info.get("url", ""),
        key=info.get("key", ""),
        authors=authors,
    )


def search_dblp_publications(
    session: requests.Session,
    query: str,
    start_year: int,
    end_year: int,
    use_cache: bool,
) -> List[Publication]:
    """
    Search DBLP publications for one query.
    """
    publications = []
    offset = 0

    LOGGER.info(
        "Searching DBLP for query: %s",
        query,
    )

    while True:
        params = {
            "q": query,
            "h": DBLP_BATCH_SIZE,
            "f": offset,
            "format": "json",
        }

        response = cached_get(
            session,
            DBLP_SEARCH_API,
            params=params,
            use_cache=use_cache,
        )

        if response is None:
            break

        try:
            data = response.json()
        except ValueError:
            LOGGER.error(
                "DBLP returned invalid JSON for query '%s'.",
                query,
            )
            break

        hits = parse_dblp_search_response(data)

        if not hits:
            break

        for hit in hits:
            publication = extract_publication_from_hit(hit)

            if publication is None:
                continue

            if publication.year < start_year:
                continue

            if publication.year > end_year:
                continue

            publications.append(publication)

        total = safe_int(
            data.get("result", {})
            .get("hits", {})
            .get("@total", 0)
        )

        offset += len(hits)

        LOGGER.info(
            "Collected approximately %d/%d records for '%s'.",
            min(offset, total),
            total,
            query,
        )

        if offset >= total:
            break

        if len(hits) < DBLP_BATCH_SIZE:
            break

    return publications


# ============================================================
# KG RELEVANCE DETECTION
# ============================================================

def find_matching_keywords(title: str) -> List[str]:
    normalized_title = normalize_text(title)

    matches = []

    for keyword in KG_KEYWORDS:
        normalized_keyword = normalize_text(keyword)

        if normalized_keyword in normalized_title:
            matches.append(keyword)

    return sorted(
        matches,
        key=len,
        reverse=True,
    )


def is_kg_venue(venue: str) -> bool:
    normalized_venue = normalize_venue(venue)
    return normalized_venue in KG_VENUES


def classify_publication(
    publication: Publication,
) -> Tuple[bool, str]:
    """
    A publication is considered KG-related if:

    1. Its venue is in KG_VENUES, or
    2. Its title contains a KG keyword.
    """
    normalized_venue = normalize_venue(
        publication.venue
    )

    if is_kg_venue(publication.venue):
        return True, f"venue:{normalized_venue}"

    matching_keywords = find_matching_keywords(
        publication.title
    )

    if matching_keywords:
        return (
            True,
            "title:" + ", ".join(
                matching_keywords[:5]
            ),
        )

    return False, ""


def get_publication_weight(
    publication: Publication,
) -> float:
    """
    Return the conference-specific weight.

    Unknown venues receive DEFAULT_CONFERENCE_WEIGHT.
    """
    normalized_venue = normalize_venue(
        publication.venue
    )

    return CONFERENCE_WEIGHTS.get(
        normalized_venue,
        DEFAULT_CONFERENCE_WEIGHT,
    )


def filter_kg_publications(
    publications: Iterable[Publication],
) -> List[Publication]:
    """
    Filter KG publications and remove duplicates.
    """
    retained = []
    seen = set()

    for publication in publications:
        is_relevant, reason = classify_publication(
            publication
        )

        if not is_relevant:
            continue

        deduplication_key = (
            publication.key
            or (
                f"{publication.title.lower()}|"
                f"{publication.venue.lower()}|"
                f"{publication.year}"
            )
        )

        if deduplication_key in seen:
            continue

        seen.add(deduplication_key)

        publication.weight = get_publication_weight(
            publication
        )

        publication.relevance_reason = reason

        retained.append(publication)

    return retained


# ============================================================
# AUTHOR IDENTIFICATION AND ENRICHMENT
# ============================================================

def parse_author_search_response(
    data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    result = data.get("result", {})
    hits = result.get("hits", {})
    authors = hits.get("author", [])

    if isinstance(authors, dict):
        authors = [authors]

    return authors or []


def find_author_pid(
    session: requests.Session,
    author_name: str,
    use_cache: bool,
) -> str:
    """
    Find a DBLP PID using an author name.

    This is only used when a publication record does not
    already contain an author PID.
    """
    params = {
        "q": author_name,
        "h": 10,
        "format": "json",
    }

    response = cached_get(
        session,
        DBLP_AUTHOR_API,
        params=params,
        use_cache=use_cache,
    )

    if response is None:
        return ""

    try:
        data = response.json()
    except ValueError:
        return ""

    authors = parse_author_search_response(data)

    if not authors:
        return ""

    first_result = authors[0]

    author_info = first_result.get(
        "author",
        first_result,
    )

    if isinstance(author_info, dict):
        return (
            author_info.get("@pid", "")
            or author_info.get("pid", "")
        )

    return ""


def parse_dblp_person_xml(
    session: requests.Session,
    pid: str,
    use_cache: bool,
) -> Dict[str, Any]:
    """
    Read author information from DBLP's XML profile.
    """
    if not pid:
        return {}

    url = f"{DBLP_PERSON_API}/{pid}.xml"

    response = cached_get(
        session,
        url,
        use_cache=use_cache,
    )

    if response is None:
        return {}

    try:
        soup = BeautifulSoup(
            response.text,
            "xml",
        )
    except Exception:
        return {}

    person = soup.find("person")

    if person is None:
        return {}

    author_node = person.find("author")

    name = (
        author_node.get_text(
            " ",
            strip=True,
        )
        if author_node
        else ""
    )

    affiliations = [
        node.get_text(
            " ",
            strip=True,
        )
        for node in person.find_all("affiliation")
    ]

    homepage_urls = [
        node.get_text(
            " ",
            strip=True,
        )
        for node in person.find_all("url")
    ]

    return {
        "name": name,
        "affiliations": deduplicate(
            affiliations
        ),
        "homepage_urls": deduplicate(
            homepage_urls
        ),
        "dblp_url": f"https://dblp.org/pid/{pid}",
    }


# ============================================================
# AUTHOR AGGREGATION
# ============================================================

def make_author_key(
    author_data: Dict[str, str],
) -> str:
    """
    Prefer DBLP PID as the author identity.
    Fall back to normalized name if PID is unavailable.
    """
    pid = author_data.get("pid", "").strip()

    if pid:
        return f"pid:{pid}"

    name = normalize_text(
        author_data.get("name", "")
    )

    return f"name:{name}"


def aggregate_authors(
    publications: Iterable[Publication],
) -> Dict[str, Author]:
    """
    Aggregate all KG-related publications by author.

    Every co-author receives the complete paper weight.

    Example:
        Paper weight = 3.0
        Number of authors = 4

    Each of the four authors receives 3.0 points.
    The score is not divided among the authors.
    """
    authors = {}

    for publication in publications:
        for author_data in publication.authors:
            author_name = clean_title(
                author_data.get("name", "")
            )

            if not author_name:
                continue

            author_pid = author_data.get(
                "pid",
                "",
            ).strip()

            author_key = make_author_key(
                author_data
            )

            if author_key not in authors:
                authors[author_key] = Author(
                    pid=author_pid,
                    name=author_name,
                )

            author = authors[author_key]

            # ------------------------------------------------
            # Main scoring operation
            # ------------------------------------------------
            #
            # Every co-author receives the complete weight.
            #
            author.score += publication.weight
            author.publication_count += 1

            publication_data = {
                "title": publication.title,
                "venue": publication.venue,
                "normalized_venue": normalize_venue(
                    publication.venue
                ),
                "year": publication.year,
                "url": publication.url,
                "key": publication.key,
                "weight": publication.weight,
                "relevance_reason": (
                    publication.relevance_reason
                ),
            }

            author.publications.append(
                publication_data
            )

            normalized_venue = normalize_venue(
                publication.venue
            )

            author.venues[normalized_venue] = (
                author.venues.get(
                    normalized_venue,
                    0.0,
                )
                + publication.weight
            )

            year_key = str(publication.year)

            author.years[year_key] = (
                author.years.get(
                    year_key,
                    0.0,
                )
                + publication.weight
            )

    return authors


def merge_duplicate_authors(
    authors: Dict[str, Author],
) -> List[Author]:
    """
    Merge author records that share the same DBLP PID.

    Name-only records are not aggressively merged because two
    different researchers can have the same name.
    """
    authors_by_pid = {}
    authors_without_pid = []

    for author in authors.values():
        if not author.pid:
            authors_without_pid.append(author)
            continue

        if author.pid not in authors_by_pid:
            authors_by_pid[author.pid] = author
            continue

        existing = authors_by_pid[author.pid]

        existing.score += author.score
        existing.publication_count += (
            author.publication_count
        )

        existing.publications.extend(
            author.publications
        )

        for venue, score in author.venues.items():
            existing.venues[venue] = (
                existing.venues.get(
                    venue,
                    0.0,
                )
                + score
            )

        for year, score in author.years.items():
            existing.years[year] = (
                existing.years.get(
                    year,
                    0.0,
                )
                + score
            )

        existing.affiliations = deduplicate(
            existing.affiliations
            + author.affiliations
        )

        existing.homepage_urls = deduplicate(
            existing.homepage_urls
            + author.homepage_urls
        )

    return (
        list(authors_by_pid.values())
        + authors_without_pid
    )


def enrich_authors(
    session: requests.Session,
    authors: List[Author],
    use_cache: bool,
) -> None:
    """
    Enrich authors with DBLP profile information.
    """
    for index, author in enumerate(
        authors,
        start=1,
    ):
        LOGGER.info(
            "Enriching author %d/%d: %s",
            index,
            len(authors),
            author.name,
        )

        if not author.pid:
            author.pid = find_author_pid(
                session,
                author.name,
                use_cache,
            )

        metadata = parse_dblp_person_xml(
            session,
            author.pid,
            use_cache,
        )

        if metadata:
            if metadata.get("name"):
                author.name = metadata["name"]

            author.affiliations = deduplicate(
                author.affiliations
                + metadata.get(
                    "affiliations",
                    [],
                )
            )

            author.homepage_urls = deduplicate(
                author.homepage_urls
                + metadata.get(
                    "homepage_urls",
                    [],
                )
            )

            author.dblp_url = metadata.get(
                "dblp_url",
                author.dblp_url,
            )

        if author.pid:
            author.dblp_url = (
                author.dblp_url
                or f"https://dblp.org/pid/{author.pid}"
            )

        author.profile_url = (
            author.homepage_urls[0]
            if author.homepage_urls
            else author.dblp_url
        )


# ============================================================
# RANKING
# ============================================================

def sort_and_rank_authors(
    authors: List[Author],
    top_n: int,
) -> List[Author]:
    """
    Ranking order:

    1. Weighted KG score descending
    2. Unweighted KG publication count descending
    3. Researcher name ascending
    """
    authors.sort(
        key=lambda author: (
            -author.score,
            -author.publication_count,
            author.name.lower(),
        )
    )

    ranked_authors = authors[:top_n]

    for rank, author in enumerate(
        ranked_authors,
        start=1,
    ):
        author.rank = rank

    return ranked_authors


# ============================================================
# HTML GENERATION
# ============================================================

def render_html(
    authors: List[Author],
    publications: List[Publication],
    start_year: int,
    end_year: int,
) -> Path:
    template = Template(HTML_TEMPLATE)

    author_data = [
        author.to_dict()
        for author in authors
    ]

    total_score = sum(
        publication.weight
        for publication in publications
    )

    rendered_html = template.render(
        authors=author_data,
        publication_count=len(publications),
        total_score=round(
            total_score,
            3,
        ),
        start_year=start_year,
        end_year=end_year,
        recent_limit=DEFAULT_RECENT_PUBLICATIONS_LIMIT,
        generated_at=datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_HTML.write_text(
        rendered_html,
        encoding="utf-8",
    )

    return OUTPUT_HTML


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(
    start_year: int,
    end_year: int,
    top_n: int,
    use_cache: bool,
) -> None:
    ensure_directories()

    session = requests.Session()

    # Multiple queries improve recall because DBLP's search
    # API does not provide complete abstracts and keywords
    # for every publication.
    search_queries = [
        "knowledge graph",
        "knowledge base",
        "semantic web",
        "linked data",
        "ontology",
        "entity linking",
        "knowledge representation",
        "relation extraction",
        "knowledge reasoning",
        "question answering",
        "knowledge embedding",
        "knowledge graph embedding",
        "Wikidata",
        "DBpedia",
        "Freebase",
    ]

    all_publications = []

    for query in search_queries:
        query_publications = search_dblp_publications(
            session=session,
            query=query,
            start_year=start_year,
            end_year=end_year,
            use_cache=use_cache,
        )

        all_publications.extend(
            query_publications
        )

    LOGGER.info(
        "Collected %d DBLP records before filtering.",
        len(all_publications),
    )

    kg_publications = filter_kg_publications(
        all_publications
    )

    LOGGER.info(
        "Retained %d KG-related publications.",
        len(kg_publications),
    )

    authors_by_key = aggregate_authors(
        kg_publications
    )

    authors = merge_duplicate_authors(
        authors_by_key
    )

    LOGGER.info(
        "Found %d researchers before enrichment.",
        len(authors),
    )

    enrich_authors(
        session=session,
        authors=authors,
        use_cache=use_cache,
    )

    ranked_authors = sort_and_rank_authors(
        authors=authors,
        top_n=top_n,
    )

    output_file = render_html(
        authors=ranked_authors,
        publications=kg_publications,
        start_year=start_year,
        end_year=end_year,
    )

    LOGGER.info(
        "HTML report generated at: %s",
        output_file,
    )


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

def parse_arguments() -> argparse.Namespace:
    current_year = datetime.now().year

    parser = argparse.ArgumentParser(
        description=(
            "Generate Knowledge Graph researcher rankings "
            "using DBLP."
        )
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=(
            current_year
            - DEFAULT_YEAR_RANGE
            + 1
        ),
        help="First publication year.",
    )

    parser.add_argument(
        "--end-year",
        type=int,
        default=current_year,
        help="Last publication year.",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help="Number of researchers to show.",
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable local DBLP response caching.",
    )

    return parser.parse_args()


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

def main() -> None:
    args = parse_arguments()

    if args.start_year > args.end_year:
        raise ValueError(
            "The start year cannot be greater than the end year."
        )

    if args.top_n <= 0:
        raise ValueError(
            "The value of --top-n must be greater than zero."
        )

    run_pipeline(
        start_year=args.start_year,
        end_year=args.end_year,
        top_n=args.top_n,
        use_cache=(
            USE_CACHE_BY_DEFAULT
            and not args.no_cache
        ),
    )


if __name__ == "__main__":
    main()
