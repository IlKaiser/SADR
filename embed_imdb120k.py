#!/usr/bin/env python3
"""
Prepare a loaded Neo4j recommendations database for LlamaIndex retrieval.

Run it after loading the public recommendations dump (see README):
1. Generates retrieval_text for every node (JSON of its properties, APOC required)
2. Assigns a string id to every node and the __Entity__ label to Movie, Person
   and Genre nodes (required by LlamaIndex); User nodes get an id only
3. Embeds every node with LlamaIndex HuggingFaceEmbedding (BAAI/bge-small-en-v1.5)
4. Creates the vector index 'entity' on __Entity__.embedding

Usage:
    python embed_imdb120k.py
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
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")



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

    
    # ---- String id for every node, __Entity__ label for the retrievable ones ----
    print("Adding string ids and the __Entity__ label...")
    run(driver, "MATCH (m:Movie) SET m.id = 'movie_' + coalesce(toString(m.tmdbId), 'm' + toString(m.movieId))")
    run(driver, "MATCH (p:Person) SET p.id = 'person_' + toString(p.tmdbId)")
    run(driver, "MATCH (g:Genre) SET g.id = 'genre_' + replace(toLower(g.name), ' ', '_')")
    run(driver, "MATCH (u:User) SET u.id = 'user_' + toString(u.userId)")
    run(driver, "MATCH (n) WHERE n:Movie OR n:Person OR n:Genre SET n:__Entity__")
    print("  Done.\n")

    # ---- For each node type create retrieval_text as a json with all attributes but id ----
    print("Creating retrieval_text for all nodes...")
    run(driver, """
        MATCH (n)
        WHERE n.retrieval_text IS NULL
        SET n.retrieval_text = apoc.convert.toJson(
            apoc.map.removeKeys(properties(n), ['id', 'embedding'])
        )
    """)
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
        MATCH (n)
        WHERE n.retrieval_text IS NOT NULL AND n.embedding IS NULL
        RETURN elementId(n) AS id, n.retrieval_text AS text
    """)
    total = len(nodes_to_embed)
    print(f"Embedding {total} nodes...")

    EMBED_BATCH = 8192
    for i in range(0, total, EMBED_BATCH):
        batch = nodes_to_embed[i:i + EMBED_BATCH]
        texts = [n["text"] for n in batch]
        ids = [n["id"] for n in batch]

        embeddings = embed_model.get_text_embedding_batch(texts)

        params = [{"id": id_, "embedding": emb}
                  for id_, emb in zip(ids, embeddings)]
        run(driver, """
            UNWIND $rows AS row
            MATCH (n) WHERE elementId(n) = row.id
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
            FOR n ON (n.embedding)
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
        MATCH (n)
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
