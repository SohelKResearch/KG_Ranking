#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set

from jinja2 import Environment, FileSystemLoader


# ============================================================
# Configuration
# ============================================================

PUBLICATIONS_FILE = Path("publications.json")
AUTHORS_FILE = Path("authors.json")
TEMPLATE_FILE = "template.html"

DEFAULT_OUTPUT_FILE = "kg_rankings.html"
DEFAULT_START_YEAR = 2015
DEFAULT_END_YEAR = 2026


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
    Normalize text for comparison.
    """
    if value is None:
        return ""

    text = str(value).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text(value: Any) -> str:
    """
    Convert a value to clean readable text.
    """
    if value is None:
        return ""

    text = str(value)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_json(path: Path, default: Any) -> Any:
    """
    Load JSON data from a file.
    """
    if not path.exists():
        print(f"[WARNING] File not found: {path}")
        return default

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    except json.JSONDecodeError as error:
        print(f"[ERROR] Invalid JSON in {path}: {error}")
        return default

    except Exception as error:
        print(f"[ERROR] Could not read {path}: {error}")
        return default


def save_json(path: Path, data: Any) -> None:
    """
    Save JSON data.
    """
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


# ============================================================
# KG relevance detection
# ============================================================

def contains_kg_keyword(text: str) -> bool:
    """
    Return True when the text contains at least one
    KG-related keyword.
    """

    normalized_text = normalize_text(text)

    for keyword in KG_KEYWORDS:
        if keyword in normalized_text:
            return True

    return False


def is_core_kg_venue(venue: str) -> bool:
    """
    Check whether a publication belongs to a core KG venue.
    """

    normalized_venue = normalize_text(venue)

    core_kg_venues = [
        "iswc",
        "eswc",
        "k-cap",
        "kr",
        "ruleml",
        "ruleml+rr",
    ]

    return any(
        core_venue in normalized_venue
        for core_venue in core_kg_venues
    )


def is_kg_related(publication: Dict[str, Any]) -> bool:
    """
    Determine whether a publication is KG-related.

    A paper is retained if:
    - Its title contains a KG keyword;
    - Its abstract contains a KG keyword;
    - Its keywords contain a KG keyword; or
    - It belongs to a core KG venue.

    Each retained paper contributes exactly one point.
    """

    title = clean_text(publication.get("title"))
    abstract = clean_text(publication.get("abstract"))
    venue = clean_text(publication.get("venue"))

    keywords = publication.get("keywords", [])

    if not isinstance(keywords, list):
        keywords = [keywords]

    keyword_text = " ".join(
        clean_text(keyword)
        for keyword in keywords
    )

    combined_text = " ".join(
        [
            title,
            abstract,
            venue,
            keyword_text,
        ]
    )

    if contains_kg_keyword(combined_text):
        return True

    if is_core_kg_venue(venue):
        return True

    return False


# ============================================================
# Publication filtering
# ============================================================

def parse_year(value: Any) -> int:
    """
    Convert a year value into an integer.
    Returns 0 if conversion fails.
    """

    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def filter_publications(
    publications: List[Dict[str, Any]],
    start_year: int,
    end_year: int,
    excluded_venues: Set[str],
) -> List[Dict[str, Any]]:
    """
    Filter publications by year, excluded venues,
    and KG relevance.
    """

    filtered_publications = []

    for publication in publications:
        year = parse_year(publication.get("year"))

        if year < start_year or year > end_year:
            continue

        venue = normalize_text(publication.get("venue"))

        if any(
            excluded_venue in venue
            for excluded_venue in excluded_venues
        ):
            continue

        if not is_kg_related(publication):
            continue

        publication["kg_related"] = True
        publication["weight"] = 1

        filtered_publications.append(publication)

    return filtered_publications


# ============================================================
# Author aggregation
# ============================================================

def aggregate_authors(
    publications: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Aggregate KG-related publications by author.

    Score:
        number of KG-related publications

    Every publication has weight 1.
    """

    author_map: Dict[str, Dict[str, Any]] = {}

    for publication in publications:
        title = clean_text(publication.get("title"))
        year = parse_year(publication.get("year"))
        venue = clean_text(publication.get("venue"))
        url = clean_text(publication.get("url"))
        doi = clean_text(publication.get("doi"))

        authors = publication.get("authors", [])

        if not isinstance(authors, list):
            authors = [authors]

        for author_name in authors:
            author_name = clean_text(author_name)

            if not author_name:
                continue

            author_key = normalize_text(author_name)

            if author_key not in author_map:
                author_map[author_key] = {
                    "name": author_name,
                    "score": 0,
                    "kg_publications": 0,
                    "pubs": [],
                    "years": [],
                    "venues": [],
                }

            author = author_map[author_key]

            # Every KG-related paper contributes exactly 1.
            author["score"] += 1
            author["kg_publications"] += 1

            author["pubs"].append(
                {
                    "title": title,
                    "year": year,
                    "venue": venue,
                    "url": url,
                    "doi": doi,
                    "weight": 1,
                }
            )

            if year:
                author["years"].append(year)

            if venue:
                author["venues"].append(venue)

    return list(author_map.values())


# ============================================================
# Author metadata
# ============================================================

def merge_existing_author_metadata(
    ranked_authors: List[Dict[str, Any]],
    existing_authors: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Merge optional metadata from authors.json.

    This preserves fields such as:
    - affiliation
    - homepage
    - scholar_url
    - orcid
    - country
    """

    existing_map = {}

    for author in existing_authors:
        name = clean_text(
            author.get("name")
            or author.get("author")
        )

        if name:
            existing_map[normalize_text(name)] = author

    metadata_fields = [
        "affiliation",
        "institution",
        "homepage",
        "url",
        "profile_url",
        "scholar_url",
        "orcid",
        "country",
        "image",
    ]

    for author in ranked_authors:
        author_key = normalize_text(author["name"])
        old_author = existing_map.get(author_key)

        if not old_author:
            continue

        for field in metadata_fields:
            if field in old_author and old_author[field]:
                author[field] = old_author[field]

    return ranked_authors


# ============================================================
# Ranking
# ============================================================

def assign_ranks(
    authors: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Sort authors by score and assign competition ranks.

    Example:
        10 papers -> rank 1
         8 papers -> rank 2
         8 papers -> rank 2
         5 papers -> rank 4
    """

    authors.sort(
        key=lambda author: (
            -author.get("score", 0),
            normalize_text(author.get("name")),
        )
    )

    previous_score = None
    current_rank = 0

    for index, author in enumerate(authors, start=1):
        score = author.get("score", 0)

        if score != previous_score:
            current_rank = index
            previous_score = score

        author["rank"] = current_rank

    return authors


# ============================================================
# Template rendering
# ============================================================

def render_html(
    authors: List[Dict[str, Any]],
    publications: List[Dict[str, Any]],
    output_file: Path,
    start_year: int,
    end_year: int,
) -> None:
    """
    Render the HTML website using template.html.
    """

    environment = Environment(
        loader=FileSystemLoader("."),
        autoescape=True,
    )

    template = environment.get_template(TEMPLATE_FILE)

    html = template.render(
        authors=authors,
        publications=publications,
        conferences=CONFERENCES,
        kg_keywords=KG_KEYWORDS,
        start_year=start_year,
        end_year=end_year,
        total_authors=len(authors),
        total_publications=len(publications),
        generated_title="Knowledge Graph Researcher Rankings",
    )

    output_file.write_text(
        html,
        encoding="utf-8",
    )

    print(f"[HTML] Generated: {output_file}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Knowledge Graph researcher rankings."
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=DEFAULT_START_YEAR,
        help="Minimum publication year.",
    )

    parser.add_argument(
        "--end-year",
        type=int,
        default=DEFAULT_END_YEAR,
        help="Maximum publication year.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help="Output HTML filename.",
    )

    parser.add_argument(
        "--excluded-venues",
        type=str,
        default="",
        help=(
            "Comma-separated venue names to exclude. "
            "Example: arxiv,workshop"
        ),
    )

    args = parser.parse_args()

    output_file = Path(args.output)

    excluded_venues = {
        normalize_text(venue)
        for venue in args.excluded_venues.split(",")
        if normalize_text(venue)
    }

    print("=" * 70)
    print("KNOWLEDGE GRAPH RESEARCHER RANKING")
    print("=" * 70)

    print(f"Start year: {args.start_year}")
    print(f"End year:   {args.end_year}")
    print(f"Output:     {output_file}")
    print()

    # --------------------------------------------------------
    # Load input files
    # --------------------------------------------------------

    publications_data = load_json(
        PUBLICATIONS_FILE,
        default=[],
    )

    authors_data = load_json(
        AUTHORS_FILE,
        default=[],
    )

    if isinstance(publications_data, dict):
        publications = publications_data.get(
            "publications",
            publications_data.get("data", []),
        )
    else:
        publications = publications_data

    if isinstance(authors_data, dict):
        existing_authors = authors_data.get(
            "authors",
            authors_data.get("data", []),
        )
    else:
        existing_authors = authors_data

    if not isinstance(publications, list):
        print("[ERROR] publications.json must contain a list.")
        return

    if not isinstance(existing_authors, list):
        existing_authors = []

    print(f"[INPUT] Publications loaded: {len(publications)}")

    # --------------------------------------------------------
    # Filter KG publications
    # --------------------------------------------------------

    kg_publications = filter_publications(
        publications=publications,
        start_year=args.start_year,
        end_year=args.end_year,
        excluded_venues=excluded_venues,
    )

    print(
        f"[FILTER] KG publications retained: "
        f"{len(kg_publications)}"
    )

    # --------------------------------------------------------
    # Aggregate authors
    # --------------------------------------------------------

    ranked_authors = aggregate_authors(
        kg_publications
    )

    # Merge optional metadata.
    ranked_authors = merge_existing_author_metadata(
        ranked_authors=ranked_authors,
        existing_authors=existing_authors,
    )

    # --------------------------------------------------------
    # Assign ranks
    # --------------------------------------------------------

    ranked_authors = assign_ranks(
        ranked_authors
    )

    # --------------------------------------------------------
    # Render website
    # --------------------------------------------------------

    render_html(
        authors=ranked_authors,
        publications=kg_publications,
        output_file=output_file,
        start_year=args.start_year,
        end_year=args.end_year,
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RANKING COMPLETED")
    print("=" * 70)
    print(f"KG publications: {len(kg_publications)}")
    print(f"Researchers:      {len(ranked_authors)}")
    print(f"HTML file:        {output_file}")
    print()

    print("Top researchers:")

    for author in ranked_authors[:20]:
        print(
            f"{author['rank']:>3}. "
            f"{author['name']} — "
            f"{author['score']} KG publications"
        )


if __name__ == "__main__":
    main()
