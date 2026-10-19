#!/usr/bin/env python3
"""
Load the Neo4j Recommendations movie dataset and prepare it for LlamaIndex retrieval.

This script handles the full pipeline:
1. Downloads the denormalized CSV from data.neo4j.com
2. Creates Movie, Person, Genre nodes and ACTED_IN, DIRECTED, IN_GENRE relationships
3. Generates retrieval_text for ALL node types
4. Assigns string id and __Entity__ label to ALL nodes (required by LlamaIndex)
5. Embeds ALL nodes using LlamaIndex HuggingFaceEmbedding (BAAI/bge-small-en-v1.5)
6. Creates the vector index 'entity' in Neo4j for fast similarity search

Usage:
    python load_movies_dataset.py          # loads into existing DB
    python load_movies_dataset.py --clean  # wipes DB first, then loads
"""

from __future__ import annotations

import csv
import io
import os
import re
import sys
from pathlib import Path
from urllib.request import urlopen

from neo4j import GraphDatabase

# ------------------------------
# Load .env
# ------------------------------

def _load_env_file() -> None:
    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^export\s+", "", line)
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

_load_env_file()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4jpassword")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Denormalized CSV: each row = one movie-person pair with role/work type
DATA_URL = "https://data.neo4j.com/importing/test.csv"

BATCH = 500

# ------------------------------
# Helpers
# ------------------------------

def _int(v):
    if v is None: return None
    v = str(v).strip()
    if not v: return None
    try: return int(v)
    except ValueError: return None

def _float(v):
    if v is None: return None
    v = str(v).strip()
    if not v: return None
    try: return float(v)
    except ValueError: return None

def run(driver, cypher, params=None):
    with driver.session(database=NEO4J_DATABASE) as s:
        s.run(cypher, params or {}).consume()

def run_read(driver, cypher, params=None):
    with driver.session(database=NEO4J_DATABASE) as s:
        return [dict(r) for r in s.run(cypher, params or {})]

def batched(driver, cypher, rows):
    for i in range(0, len(rows), BATCH):
        run(driver, cypher, {"rows": rows[i:i+BATCH]})

# ------------------------------
# Main
# ------------------------------

def main():
    clean = "--clean" in sys.argv

    print(f"Connecting to {NEO4J_URI} as {NEO4J_USER} (db={NEO4J_DATABASE})...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    if not run_read(driver, "RETURN 1 AS ok"):
        print("ERROR: Cannot connect"); return 1
    print("Connected OK.\n")

    if clean:
        print("Cleaning database...")
        run(driver, "MATCH (n) DETACH DELETE n")
        print("  Done.\n")

    # ---- Download CSV ----
    print(f"Downloading {DATA_URL} ...")
    with urlopen(DATA_URL) as resp:
        text = resp.read().decode("utf-8")
    raw = list(csv.DictReader(io.StringIO(text)))
    print(f"  {len(raw)} rows downloaded.\n")

    # ---- De-duplicate movies ----
    movies_map = {}
    for r in raw:
        mid = _int(r.get("movieId"))
        if mid is None or mid in movies_map:
            continue
        movies_map[mid] = {
            "tmdbId":     _int(r.get("movie_tmdbId")) or mid,
            "imdbId":     _int(r.get("movie_imdbId")),
            "title":      r.get("title"),
            "year":       _int(r.get("year")),
            "released":   r.get("released"),
            "runtime":    _int(r.get("runtime")),
            "budget":     _float(r.get("budget")),
            "revenue":    _float(r.get("revenue")),
            "imdbRating": _float(r.get("imdbRating")),
            "imdbVotes":  _int(r.get("imdbVotes")),
            "url":        r.get("movie_url"),
            "plot":       r.get("plot"),
            "poster":     r.get("movie_poster"),
            "languages":  [x.strip() for x in (r.get("languages") or "").split("|") if x.strip()],
            "countries":  [x.strip() for x in (r.get("countries") or "").split("|") if x.strip()],
            "genres":     [x.strip() for x in (r.get("genres") or "").split("|") if x.strip()],
        }
    mid_to_tmdb = {mid: m["tmdbId"] for mid, m in movies_map.items()}

    # ---- De-duplicate persons ----
    persons_map = {}
    for r in raw:
        ptmdb = _int(r.get("person_tmdbId"))
        if ptmdb is None or ptmdb in persons_map:
            continue
        persons_map[ptmdb] = {
            "tmdbId":  ptmdb,
            "imdbId":  _int(r.get("person_imdbId")),
            "name":    r.get("name"),
            "born":    r.get("born"),
            "bornIn":  r.get("bornIn"),
            "died":    r.get("died"),
            "bio":     r.get("bio"),
            "url":     r.get("person_url"),
            "poster":  r.get("person_poster"),
        }

    # ---- Extract relationships ----
    acted_in = []
    directed = []
    for r in raw:
        mid = _int(r.get("movieId"))
        ptmdb = _int(r.get("person_tmdbId"))
        m_tmdb = mid_to_tmdb.get(mid)
        if m_tmdb is None or ptmdb is None:
            continue
        work = (r.get("work") or "").strip().upper()
        if "ACT" in work:
            acted_in.append({"personTmdb": ptmdb, "movieTmdb": m_tmdb, "role": r.get("role")})
        elif "DIRECT" in work:
            directed.append({"personTmdb": ptmdb, "movieTmdb": m_tmdb})

    # ---- Constraints ----
    print("Creating constraints...")
    for c in [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Movie) REQUIRE m.tmdbId IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.tmdbId IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE",
    ]:
        try: run(driver, c)
        except Exception as e: print(f"  Warning: {e}")
    print("  Done.\n")

    # ---- Load Movies ----
    print("Loading Movie nodes...")
    batched(driver, """
        UNWIND $rows AS row
        MERGE (m:Movie {tmdbId: row.tmdbId})
        SET m += {
            imdbId: row.imdbId, title: row.title, year: row.year,
            released: row.released, runtime: row.runtime, budget: row.budget,
            revenue: row.revenue, imdbRating: row.imdbRating,
            imdbVotes: row.imdbVotes, url: row.url, plot: row.plot,
            poster: row.poster, languages: row.languages,
            countries: row.countries, genres: row.genres
        }
    """, list(movies_map.values()))
    print(f"  {len(movies_map)} movies.\n")

    # ---- Load Persons ----
    print("Loading Person nodes...")
    batched(driver, """
        UNWIND $rows AS row
        MERGE (p:Person {tmdbId: row.tmdbId})
        SET p += {
            imdbId: row.imdbId, name: row.name, born: row.born,
            bornIn: row.bornIn, died: row.died, bio: row.bio,
            url: row.url, poster: row.poster
        }
    """, list(persons_map.values()))
    print(f"  {len(persons_map)} persons.\n")

    # ---- ACTED_IN ----
    print("Loading ACTED_IN relationships...")
    batched(driver, """
        UNWIND $rows AS row
        MATCH (p:Person {tmdbId: row.personTmdb})
        MATCH (m:Movie  {tmdbId: row.movieTmdb})
        MERGE (p)-[r:ACTED_IN]->(m)
        SET r.role = row.role
    """, acted_in)
    print(f"  {len(acted_in)} ACTED_IN rows sent.\n")

    # ---- DIRECTED ----
    print("Loading DIRECTED relationships...")
    batched(driver, """
        UNWIND $rows AS row
        MATCH (p:Person {tmdbId: row.personTmdb})
        MATCH (m:Movie  {tmdbId: row.movieTmdb})
        MERGE (p)-[:DIRECTED]->(m)
    """, directed)
    print(f"  {len(directed)} DIRECTED rows sent.\n")

    # ---- Genres ----
    print("Creating Genre nodes and IN_GENRE relationships...")
    run(driver, """
        MATCH (m:Movie) WHERE m.genres IS NOT NULL
        UNWIND m.genres AS gn
        WITH m, trim(gn) AS g WHERE g <> ''
        MERGE (genre:Genre {name: g})
        MERGE (m)-[:IN_GENRE]->(genre)
    """)
    cnt = run_read(driver, "MATCH (g:Genre) RETURN count(g) AS c")[0]["c"]
    print(f"  {cnt} genres.\n")

    # ---- retrieval_text for ALL node types ----
    print("Generating retrieval_text for Movie nodes...")
    run(driver, """
        MATCH (m:Movie)
        WITH m,
             'title: ' + coalesce(m.title, '') +
             '\nplot: ' + coalesce(m.plot, '') +
             '\ngenres: ' + coalesce(reduce(s='', g IN m.genres | s + CASE WHEN s='' THEN '' ELSE ', ' END + g), '') +
             '\nlanguages: ' + coalesce(reduce(s='', l IN m.languages | s + CASE WHEN s='' THEN '' ELSE ', ' END + l), '') +
             '\nyear: ' + coalesce(toString(m.year), '') AS text
        SET m.retrieval_text = text
    """)
    print("  Done.\n")

    print("Generating retrieval_text for Person nodes...")
    run(driver, """
        MATCH (p:Person)
        WITH p,
             'name: ' + coalesce(p.name, '') +
             '\nborn: ' + coalesce(p.born, '') +
             '\nbornIn: ' + coalesce(p.bornIn, '') +
             '\nbio: ' + coalesce(p.bio, '') AS text
        SET p.retrieval_text = text
    """)
    print("  Done.\n")

    print("Generating retrieval_text for Genre nodes...")
    run(driver, """
        MATCH (g:Genre)
        SET g.retrieval_text = 'genre: ' + coalesce(g.name, '')
    """)
    print("  Done.\n")

    # ---- String id for ALL nodes (required by LlamaIndex) ----
    print("Adding string id to all nodes...")
    run(driver, """
        MATCH (m:Movie)
        SET m.id = 'movie_' + toString(m.tmdbId)
    """)
    run(driver, """
        MATCH (p:Person)
        SET p.id = 'person_' + toString(p.tmdbId)
    """)
    run(driver, """
        MATCH (g:Genre)
        SET g.id = 'genre_' + replace(toLower(g.name), ' ', '_')
    """)
    print("  Done.\n")

    # ---- __Entity__ label for ALL nodes (required by LlamaIndex) ----
    print("Adding __Entity__ label to all nodes...")
    run(driver, "MATCH (n) WHERE n:Movie OR n:Person OR n:Genre SET n:__Entity__")
    print("  Done.\n")

    # ---- Embedding ALL nodes via LlamaIndex ----
    embed_provider = os.getenv("EMBED_PROVIDER", "huggingface")
    embed_model_name = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    print(f"Loading embedding model ({embed_model_name})...")

    if embed_provider == "huggingface":
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        embed_model = HuggingFaceEmbedding(
            model_name=embed_model_name,
            normalize=os.getenv("EMBED_NORMALIZE", "true").lower() in ("1", "true", "yes"),
            cache_folder=os.getenv("HF_HOME") or None,
        )
    elif embed_provider == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding
        embed_model = OpenAIEmbedding(model=embed_model_name)
    else:
        print(f"ERROR: Unknown EMBED_PROVIDER: {embed_provider}")
        driver.close()
        return 1

    print("  Model loaded.\n")

    nodes_to_embed = run_read(driver, """
        MATCH (n:__Entity__)
        WHERE n.retrieval_text IS NOT NULL AND n.embedding IS NULL
        RETURN n.id AS id, n.retrieval_text AS text
    """)
    total = len(nodes_to_embed)
    print(f"Embedding {total} nodes...")

    EMBED_BATCH = 64
    for i in range(0, total, EMBED_BATCH):
        batch = nodes_to_embed[i:i + EMBED_BATCH]
        texts = [n["text"] for n in batch]
        ids = [n["id"] for n in batch]

        embeddings = embed_model.get_text_embedding_batch(texts)

        params = [{"id": id_, "embedding": emb}
                  for id_, emb in zip(ids, embeddings)]
        run(driver, """
            UNWIND $rows AS row
            MATCH (n:__Entity__ {id: row.id})
            SET n.embedding = row.embedding
        """, {"rows": params})

        done = min(i + EMBED_BATCH, total)
        print(f"  {done}/{total} ({done * 100 // total}%)")

    print("  Embedding complete.\n")

    # ---- Vector index ----
    print("Creating vector index 'entity' on __Entity__.embedding...")
    dim = len(embed_model.get_text_embedding("test"))
    try:
        run(driver, f"""
            CREATE VECTOR INDEX entity IF NOT EXISTS
            FOR (n:__Entity__) ON (n.embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {dim},
                `vector.similarity_function`: 'cosine'
            }}}}
        """)
        print(f"  Done (dimension={dim}).\n")
    except Exception as e:
        print(f"  Index may already exist: {e}\n")

    # ---- Summary ----
    print("=" * 50)
    print("DATASET LOADED AND EMBEDDED SUCCESSFULLY")
    print("=" * 50)
    for r in run_read(driver, "MATCH (n) UNWIND labels(n) AS l RETURN DISTINCT l AS label, count(*) AS c ORDER BY c DESC"):
        print(f"  :{r['label']}  ->  {r['c']} nodes")
    for r in run_read(driver, "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c ORDER BY c DESC"):
        print(f"  [{r['t']}]  ->  {r['c']} rels")

    embedded = run_read(driver, """
        MATCH (n:__Entity__)
        RETURN count(n) AS total,
               count(CASE WHEN n.embedding IS NOT NULL THEN 1 END) AS with_embedding
    """)[0]
    print(f"\n  Embedded: {embedded['with_embedding']}/{embedded['total']} nodes")
    print(f"  Vector index: entity (dim={dim}, cosine)")

    print(f"\nNext step:")
    print(f'  python neo4j_kg_retrieval_llamaindex.py retrieve --query "romantic films in New York" --out trace.json')

    driver.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
