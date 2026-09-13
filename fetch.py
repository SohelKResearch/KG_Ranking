#!/usr/bin/env python3

import argparse
import json
import os
import re
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import requests


# ============================================================
# Configuration
# ============================================================

DBLP_API_URL = "https://dblp.org/search/publ/api"

IEEE_API_URL = "https://ieeexploreapi.ieee.org/api/v1/search/articles"

DEFAULT_START_YEAR = 2015
DEFAULT_END_YEAR = 2026

REQUEST_TIMEOUT = 30
SLEEP_SECONDS = 0.2

CACHE_DIR = Path("cache")
PUBLICATIONS_FILE = Path("publications.json")
AUTHORS_FILE = Path("authors.json")


# ============================================================
# Knowledge Graph-related conferences
# ============================================================

CONFERENCES = [
    "ACL",
    "IJCAI",
    "SIGIR",
    "ICWSM",
    "ECML",
    "KDD",
    "UAI",
    "AACL",
    "ECAI",
    "ICLR",
    "AISTATS",
    "NeurIPS",
    "ISWC",
    "IJCNLP",
    "EMNLP",
    "ICDM",
    "ICML",
    "AAAI",
    "WSDM",
    "COLING",
    "WWW",
    "EACL",
    "NAACL",
    "PAKDD",
    "ESWC",
    "K-CAP",
    "KR",
    "RuleML+RR",
    "ICDE",
    "EDBT",
    "DASFAA",
    "ASONAM",
]


# ============================================================
# Knowledge Graph keywords
# ============================================================

KG_KEYWORDS = [
    # General KG concepts
    "knowledge graph",
    "knowledge graphs",
    "knowledge base",
    "knowledge bases",
    "knowledge representation",
    "knowledge engineering",
    "knowledge acquisition",
    "knowledge integration",
    "knowledge fusion",
    "knowledge discovery",

    # Semantic Web
    "semantic web",
    "linked data",
    "open linked data",
    "semantic knowledge graph",
    "semantic graph",

    # Semantic Web technologies
    "rdf",
    "rdfs",
    "owl",
    "sparql",
    "triplestore",
    "triple store",
    "named graph",
    "named graphs",
    "shacl",
    "swrl",

    # Ontologies
    "ontology",
    "ontologies",
    "ontology engineering",
    "ontology learning",
    "ontology matching",
    "ontology alignment",
    "ontology mapping",
    "ontology integration",
    "ontology population",
    "ontology evolution",

    # KG construction and management
    "knowledge graph construction",
    "knowledge graph generation",
    "knowledge graph extraction",
    "knowledge graph population",
    "knowledge graph completion",
    "knowledge graph enrichment",
    "knowledge graph evolution",
    "knowledge graph fusion",
    "knowledge graph integration",
    "knowledge graph refinement",
    "knowledge graph management",

    # Entities and relations
    "entity linking",
    "entity disambiguation",
    "entity resolution",
    "entity alignment",
    "entity matching",
    "relation extraction",
    "relation discovery",
    "triple extraction",
    "open information extraction",
    "openie",

    # KG representation and reasoning
    "knowledge graph embedding",
    "knowledge graph embeddings",
    "kg embedding",
    "kg embeddings",
    "knowledge graph representation",
    "knowledge representation learning",
    "link prediction",
    "knowledge graph inference",
    "knowledge graph reasoning",
    "multi-hop reasoning",
    "multi hop reasoning",
    "relational reasoning",
    "graph reasoning",

    # KG question answering and retrieval
    "knowledge graph question answering",
    "knowledge graph question-answering",
    "kgqa",
    "knowledge base question answering",
    "kbqa",
    "text to sparql",
    "text-to-sparql",
    "knowledge graph retrieval",
    "knowledge graph search",
    "knowledge retrieval",

    # KG and language models
    "knowledge graph augmented generation",
    "knowledge graph-augmented generation",
    "kg-rag",
    "knowledge graph rag",
    "knowledge-enhanced language model",
    "knowledge enhanced language model",
    "knowledge-grounded language model",
    "knowledge grounded language model",
    "knowledge-enhanced generation",
    "knowledge grounded generation",

    # Existing knowledge graphs and resources
    "wikidata",
    "dbpedia",
    "freebase",
    "yago",
    "conceptnet",
    "babelnet",
    "nell",
    "wordnet",

    # KG variants
    "temporal knowledge graph",
    "dynamic knowledge graph",
    "probabilistic knowledge graph",
    "multimodal knowledge graph",
    "multilingual knowledge graph",
    "domain-specific knowledge graph",
    "biomedical knowledge graph",
    "scientific knowledge graph",
]


# ============================================================
# Utility functions
# ============================================================

def normalize_text(value: Any) -> str:
    """
    Convert a value to normalized lowercase text.
    """
    if value is None:
        return ""

    text = str(value).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text(value: Any) -> str:
    """
    Clean text while preserving readable content.
    """
    if value is None:
        return ""

    text = str(value)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_filename(value: str) -> str:
    """
    Convert a string into a safe cache filename.
    """
    value = normalize_text(value)
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")[:120]


def make_cache_key(prefix: str, value: str) -> Path:
    """
    Generate a deterministic cache path.
    """
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]
    filename = f"{safe_filename(prefix)}_{digest}.json"
    return CACHE_DIR / filename


def load_json(path: Path, default: Any = None) -> Any:
    """
    Load JSON from disk.
    """
    if not path.exists():
        return default

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as error:
        print(f"[WARNING] Could not read {path}: {error}")
        return default


def save_json(path: Path, data: Any) -> None:
    """
    Save JSON to disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    temporary_path.replace(path)


def request_json(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    cache_prefix: Optional[str] = None,
    cache_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Request JSON with local caching.
    """

    if cache_prefix and cache_key:
        cache_path = make_cache_key(cache_prefix, cache_key)
        cached_data = load_json(cache_path)

        if cached_data is not None:
            return cached_data
    else:
        cache_path = None

    try:
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()
        data = response.json()

        if cache_path:
            save_json(cache_path, data)

        time.sleep(SLEEP_SECONDS)
        return data

    except requests.RequestException as error:
        print(f"[WARNING] Request failed: {url}")
        print(f"          {error}")
        return None

    except ValueError as error:
        print(f"[WARNING] Invalid JSON response from {url}: {error}")
        return None


# ============================================================
# DBLP functions
# ============================================================

def search_dblp(
    query: str,
    start_year: int,
    end_year: int,
    max_results: int = 1000,
) -> List[Dict[str, Any]]:
    """
    Search DBLP publications using the DBLP API.
    """

    cache_key = f"{query}|{start_year}|{end_year}|{max_results}"

    params = {
        "q": query,
        "h": max_results,
        "f": 0,
        "format": "json",
    }

    data = request_json(
        DBLP_API_URL,
        params=params,
        cache_prefix="dblp",
        cache_key=cache_key,
    )

    if not data:
        return []

    result = data.get("result", {})
    hits = result.get("hits", {})
    hit_list = hits.get("hit", [])

    if isinstance(hit_list, dict):
        hit_list = [hit_list]

    publications = []

    for hit in hit_list:
        info = hit.get("info", {})

        title = clean_text(info.get("title"))
        year_value = info.get("year")
        venue = clean_text(
            info.get("venue")
            or info.get("booktitle")
            or info.get("journal")
        )

        try:
            year = int(year_value)
        except (TypeError, ValueError):
            year = None

        if year is not None:
            if year < start_year or year > end_year:
                continue

        authors = []

        author_data = info.get("authors", {}).get("author", [])

        if isinstance(author_data, dict):
            author_data = [author_data]

        for author in author_data:
            if isinstance(author, dict):
                author_name = clean_text(author.get("text"))
            else:
                author_name = clean_text(author)

            if author_name:
                authors.append(author_name)

        publication = {
            "title": title,
            "year": year,
            "venue": venue,
            "authors": authors,
            "url": clean_text(info.get("ee") or info.get("url")),
            "doi": clean_text(info.get("doi")),
            "source": "DBLP",
            "type": clean_text(info.get("type")),
        }

        publications.append(publication)

    return publications


def collect_dblp_publications(
    conferences: List[str],
    start_year: int,
    end_year: int,
) -> List[Dict[str, Any]]:
    """
    Collect DBLP publications for all configured conferences.
    """

    all_publications = []

    for conference in conferences:
        print(f"[DBLP] Searching: {conference}")

        publications = search_dblp(
            query=conference,
            start_year=start_year,
            end_year=end_year,
        )

        all_publications.extend(publications)

        print(
            f"[DBLP] {conference}: "
            f"{len(publications)} publications"
        )

    return all_publications


# ============================================================
# IEEE functions
# ============================================================

def search_ieee(
    query: str,
    start_year: int,
    end_year: int,
    api_key: Optional[str] = None,
    max_results: int = 100,
) -> List[Dict[str, Any]]:
    """
    Search IEEE Xplore metadata.

    An IEEE API key is optional. If no key is provided,
    IEEE collection is skipped.
    """

    if not api_key:
        print("[IEEE] API key not provided. Skipping IEEE search.")
        return []

    cache_key = f"{query}|{start_year}|{end_year}|{max_results}"

    params = {
        "apikey": api_key,
        "querytext": query,
        "start_year": start_year,
        "end_year": end_year,
        "max_records": max_results,
        "start_record": 1,
        "sort_order": "asc",
        "sort_field": "publication_year",
    }

    data = request_json(
        IEEE_API_URL,
        params=params,
        cache_prefix="ieee",
        cache_key=cache_key,
    )

    if not data:
        return []

    articles = data.get("articles", [])

    if not isinstance(articles, list):
        return []

    publications = []

    for article in articles:
        title = clean_text(article.get("title"))
        year_value = (
            article.get("publication_year")
            or article.get("year")
        )

        try:
            year = int(year_value)
        except (TypeError, ValueError):
            year = None

        if year is not None:
            if year < start_year or year > end_year:
                continue

        authors = []

        author_data = article.get("authors", {}).get("authors", [])

        if isinstance(author_data, dict):
            author_data = [author_data]

        for author in author_data:
            if isinstance(author, dict):
                author_name = clean_text(
                    author.get("full_name")
                    or author.get("name")
                )
            else:
                author_name = clean_text(author)

            if author_name:
                authors.append(author_name)

        keywords = []

        for keyword_group in [
            article.get("index_terms", {}),
            article.get("keywords", {}),
        ]:
            if isinstance(keyword_group, dict):
                for value in keyword_group.values():
                    if isinstance(value, list):
                        keywords.extend(clean_text(item) for item in value)
                    elif value:
                        keywords.append(clean_text(value))

        publication = {
            "title": title,
            "year": year,
            "venue": clean_text(
                article.get("publication_title")
                or article.get("conference_name")
            ),
            "authors": authors,
            "url": clean_text(
                article.get("html_url")
                or article.get("pdf_url")
            ),
            "doi": clean_text(article.get("doi")),
            "keywords": keywords,
            "abstract": clean_text(article.get("abstract")),
            "source": "IEEE",
            "type": "article",
        }

        publications.append(publication)

    return publications


def collect_ieee_publications(
    conferences: List[str],
    start_year: int,
    end_year: int,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Collect IEEE publications for all configured conferences.
    """

    if not api_key:
        return []

    all_publications = []

    for conference in conferences:
        print(f"[IEEE] Searching: {conference}")

        publications = search_ieee(
            query=conference,
            start_year=start_year,
            end_year=end_year,
            api_key=api_key,
        )

        all_publications.extend(publications)

        print(
            f"[IEEE] {conference}: "
            f"{len(publications)} publications"
        )

    return all_publications


# ============================================================
# Interspeech functions
# ============================================================

def collect_interspeech_publications(
    start_year: int,
    end_year: int,
) -> List[Dict[str, Any]]:
    """
    Placeholder for Interspeech collection.

    Interspeech does not provide a stable public API equivalent
    to DBLP. This function is kept so the pipeline can be
    extended later.

    The current KG ranking pipeline does not require Interspeech,
    so this function returns an empty list.
    """

    print("[Interspeech] No public API configured. Skipping.")
    return []


# ============================================================
# Deduplication
# ============================================================

def publication_key(publication: Dict[str, Any]) -> str:
    """
    Generate a stable key for duplicate detection.
    """

    doi = normalize_text(publication.get("doi"))

    if doi:
        return f"doi:{doi}"

    title = normalize_text(publication.get("title"))
    year = publication.get("year")

    return f"title:{title}|year:{year}"


def deduplicate_publications(
    publications: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Remove duplicate publications collected from different sources.
    """

    unique_publications = {}
    duplicate_count = 0

    for publication in publications:
        key = publication_key(publication)

        if key in unique_publications:
            duplicate_count += 1

            existing = unique_publications[key]

            # Merge missing fields from the duplicate record.
            for field, value in publication.items():
                if not existing.get(field) and value:
                    existing[field] = value

            # Merge author lists.
            existing_authors = set(existing.get("authors", []))
            new_authors = publication.get("authors", [])

            for author in new_authors:
                if author not in existing_authors:
                    existing.setdefault("authors", []).append(author)

            # Merge keywords.
            existing_keywords = set(existing.get("keywords", []))
            new_keywords = publication.get("keywords", [])

            for keyword in new_keywords:
                if keyword not in existing_keywords:
                    existing.setdefault("keywords", []).append(keyword)

        else:
            unique_publications[key] = publication

    print(f"[DEDUPLICATION] Removed duplicates: {duplicate_count}")

    return list(unique_publications.values())


# ============================================================
# Knowledge Graph relevance detection
# ============================================================

def contains_kg_keyword(text: str) -> bool:
    """
    Check whether text contains a KG-related keyword.
    """

    normalized = normalize_text(text)

    for keyword in KG_KEYWORDS:
        if keyword in normalized:
            return True

    return False


def is_kg_related(publication: Dict[str, Any]) -> bool:
    """
    Determine whether a publication is KG-related.

    A publication is considered KG-related when:
    1. Its title contains a KG keyword; or
    2. Its abstract contains a KG keyword; or
    3. Its keywords contain a KG keyword; or
    4. Its venue is a core KG venue.

    Every retained publication has equal weight: 1.
    """

    title = publication.get("title", "")
    abstract = publication.get("abstract", "")
    venue = publication.get("venue", "")
    keywords = publication.get("keywords", [])

    combined_text = " ".join(
        [
            clean_text(title),
            clean_text(abstract),
            clean_text(venue),
            " ".join(clean_text(item) for item in keywords),
        ]
    )

    if contains_kg_keyword(combined_text):
        return True

    normalized_venue = normalize_text(venue)

    core_kg_venues = [
        "iswc",
        "eswc",
        "k-cap",
        "kr",
        "ruleml",
        "ruleml+rr",
    ]

    for core_venue in core_kg_venues:
        if core_venue in normalized_venue:
            return True

    return False


def filter_kg_publications(
    publications: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Keep only KG-related publications.
    """

    kg_publications = []

    for publication in publications:
        if is_kg_related(publication):
            publication["kg_related"] = True
            publication["weight"] = 1
            kg_publications.append(publication)

    print(
        f"[KG FILTER] Retained {len(kg_publications)} "
        f"out of {len(publications)} publications"
    )

    return kg_publications


# ============================================================
# Author aggregation
# ============================================================

def aggregate_authors(
    publications: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Aggregate retained publications by author.

    Every KG-related publication contributes one point.
    """

    authors: Dict[str, Dict[str, Any]] = {}

    for publication in publications:
        publication_title = clean_text(publication.get("title"))
        publication_year = publication.get("year")
        publication_venue = clean_text(publication.get("venue"))
        publication_url = clean_text(publication.get("url"))

        for author_name in publication.get("authors", []):
            author_name = clean_text(author_name)

            if not author_name:
                continue

            author_key = normalize_text(author_name)

            if author_key not in authors:
                authors[author_key] = {
                    "name": author_name,
                    "pubs": [],
                    "score": 0,
                    "kg_publications": 0,
                    "years": [],
                    "venues": [],
                }

            author = authors[author_key]

            author["pubs"].append(
                {
                    "title": publication_title,
                    "year": publication_year,
                    "venue": publication_venue,
                    "url": publication_url,
                    "doi": clean_text(publication.get("doi")),
                    "weight": 1,
                }
            )

            author["score"] += 1
            author["kg_publications"] += 1

            if publication_year:
                author["years"].append(publication_year)

            if publication_venue:
                author["venues"].append(publication_venue)

    return list(authors.values())


# ============================================================
# Main pipeline
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect KG-related publications and author rankings."
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=DEFAULT_START_YEAR,
        help="First publication year.",
    )

    parser.add_argument(
        "--end-year",
        type=int,
        default=DEFAULT_END_YEAR,
        help="Last publication year.",
    )

    parser.add_argument(
        "--ieee-api-key",
        type=str,
        default=os.getenv("IEEE_API_KEY"),
        help="IEEE Xplore API key. Can also be provided through IEEE_API_KEY.",
    )

    parser.add_argument(
        "--skip-ieee",
        action="store_true",
        help="Skip IEEE collection.",
    )

    parser.add_argument(
        "--skip-interspeech",
        action="store_true",
        help="Skip Interspeech collection.",
    )

    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("KG RESEARCHER RANKING DATA COLLECTION")
    print("=" * 70)

    print(f"Start year: {args.start_year}")
    print(f"End year:   {args.end_year}")
    print(f"Conferences: {len(CONFERENCES)}")
    print()

    # --------------------------------------------------------
    # Collect DBLP data
    # --------------------------------------------------------

    dblp_publications = collect_dblp_publications(
        conferences=CONFERENCES,
        start_year=args.start_year,
        end_year=args.end_year,
    )

    # --------------------------------------------------------
    # Collect IEEE data
    # --------------------------------------------------------

    if args.skip_ieee:
        ieee_publications = []
        print("[IEEE] Skipped by command-line option.")
    else:
        ieee_publications = collect_ieee_publications(
            conferences=CONFERENCES,
            start_year=args.start_year,
            end_year=args.end_year,
            api_key=args.ieee_api_key,
        )

    # --------------------------------------------------------
    # Collect Interspeech data
    # --------------------------------------------------------

    if args.skip_interspeech:
        interspeech_publications = []
        print("[Interspeech] Skipped by command-line option.")
    else:
        interspeech_publications = collect_interspeech_publications(
            start_year=args.start_year,
            end_year=args.end_year,
        )

    all_publications = (
        dblp_publications
        + ieee_publications
        + interspeech_publications
    )

    print()
    print(f"[TOTAL] Collected publications: {len(all_publications)}")

    # --------------------------------------------------------
    # Deduplicate
    # --------------------------------------------------------

    unique_publications = deduplicate_publications(
        all_publications
    )

    # --------------------------------------------------------
    # Filter KG-related publications
    # --------------------------------------------------------

    kg_publications = filter_kg_publications(
        unique_publications
    )

    # Sort publications by year and title.
    kg_publications.sort(
        key=lambda publication: (
            -(publication.get("year") or 0),
            normalize_text(publication.get("title")),
        )
    )

    # --------------------------------------------------------
    # Aggregate authors
    # --------------------------------------------------------

    authors = aggregate_authors(kg_publications)

    authors.sort(
        key=lambda author: (
            -author["score"],
            normalize_text(author["name"]),
        )
    )

    # Assign ranks.
    current_rank = 0
    previous_score = None

    for index, author in enumerate(authors, start=1):
        score = author["score"]

        if score != previous_score:
            current_rank = index
            previous_score = score

        author["rank"] = current_rank

    # --------------------------------------------------------
    # Save output files
    # --------------------------------------------------------

    save_json(PUBLICATIONS_FILE, kg_publications)
    save_json(AUTHORS_FILE, authors)

    print()
    print("=" * 70)
    print("PIPELINE COMPLETED")
    print("=" * 70)
    print(f"KG publications saved: {PUBLICATIONS_FILE}")
    print(f"Authors saved:         {AUTHORS_FILE}")
    print(f"KG publications:       {len(kg_publications)}")
    print(f"Ranked authors:        {len(authors)}")
    print()

    if authors:
        print("Top researchers:")

        for author in authors[:10]:
            print(
                f"{author['rank']:>3}. "
                f"{author['name']} — "
                f"{author['score']} KG publications"
            )


if __name__ == "__main__":
    main()
