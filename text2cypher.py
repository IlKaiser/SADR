#!/usr/bin/env python3
"""
Text2Cypher retrieval: converts a natural language query into a Cypher query
using an Ollama-hosted LLM, runs it against Neo4j, and saves the results as
trace.json in the same format expected by definability.py.

Usage:
    python text2cypher.py --query "Director of fantasy movies" \
                          --model llama3.2 \
                          --out trace.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import requests
from neo4j import READ_ACCESS, GraphDatabase, basic_auth, unit_of_work


# ---------------------------------------------------------------------------
# JSON encoder for Neo4j temporal types (Date, DateTime, Time, Duration)
# ---------------------------------------------------------------------------

class Neo4jEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'iso_format'):   # Date, DateTime, Time
            return obj.iso_format()
        if hasattr(obj, '__str__'):      # Duration and other Neo4j types
            return str(obj)
        return super().default(obj)


# ---------------------------------------------------------------------------
# Load .env (same logic as definability.py)
# ---------------------------------------------------------------------------

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

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
OLLAMA_BASE    = os.getenv("OLLAMA_BASE",    "http://localhost:11434")

driver   = GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD))
database = NEO4J_DATABASE


# ---------------------------------------------------------------------------
# Graph schema (injected into the LLM prompt)
# ---------------------------------------------------------------------------

GRAPH_SCHEMA = """
Node labels and key properties:
  Movie    : id (string), title (string), year (int), imdbRating (float), runtime (int),
             plot (string), imdbId (string),
             countries (LIST of strings, e.g. ["USA", "Japan"]; some entries have a leading space),
             languages (LIST of strings, e.g. ["English", "French"])
  Person   : id (string), name (string), born (date)   (every Actor and Director is also a Person)
  Actor    : id (string), name (string)
  Director : id (string), name (string)
  Genre    : id (string), name (string), e.g. "Horror", "Animation", "Sci-Fi"

Relationships:
  (Director)-[:DIRECTED]->(Movie)
  (Actor)-[:ACTED_IN]->(Movie)
  (Movie)-[:IN_GENRE]->(Genre)

Critical property rules - wrong syntax causes empty results:
  - countries and languages are LIST properties; match them with
      any(c IN m.countries WHERE trim(c) = 'Japan')
  - Country names are as in IMDb: "USA", "UK", "France", "Germany", "Japan", "South Korea".
  - Genres are nodes: MATCH (m:Movie)-[:IN_GENRE]->(:Genre {name: 'Horror'})
  - year is an integer: WHERE m.year = 2005  (no quotes)
  - imdbRating is a float: ORDER BY m.imdbRating DESC
"""

CYPHER_SYSTEM = f"""You are a Neo4j Cypher expert for a movie knowledge graph.

{GRAPH_SCHEMA}
Given a natural language question, write a single valid Cypher query that retrieves
the most relevant nodes.

STRICT output rules:
1. The RETURN clause MUST use this exact pattern for every node variable n:
       n.id AS id, labels(n)[0] AS label, properties(n) AS properties
2. Always add LIMIT 10.
3. Use only MATCH, OPTIONAL MATCH, WHERE, WITH, RETURN, LIMIT — no APOC, no writes.
4. Output ONLY the raw Cypher query. No explanations, no markdown, no code fences.
"""


# ---------------------------------------------------------------------------
# LLM → Cypher
# ---------------------------------------------------------------------------

def generate_cypher(query: str, model: str) -> str:
    """Ask Ollama to convert a natural language query into a Cypher statement."""
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": CYPHER_SYSTEM},
            {"role": "user",   "content": query},
        ],
    }
    resp = requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=60)
    resp.raise_for_status()
    raw = resp.json()["message"]["content"].strip()
    # Strip markdown fences if the LLM added them despite instructions
    match = re.search(r"```(?:cypher)?\s*\n?(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else raw


def _validate_return_clause(cypher: str) -> str | None:
    """Return a feedback string if the RETURN clause is missing required aliases, else None."""
    lower = cypher.lower()
    if "return" not in lower:
        return "The query has no RETURN clause."
    missing = []
    if not re.search(r"\bas\s+id\b", lower):
        missing.append("'AS id'")
    if not re.search(r"\bas\s+label\b", lower):
        missing.append("'AS label'")
    if not re.search(r"\bas\s+properties\b", lower):
        missing.append("'AS properties'")
    if missing:
        return (
            f"The RETURN clause is missing required aliases: {', '.join(missing)}. "
            "Use exactly: <var>.id AS id, labels(<var>)[0] AS label, properties(<var>) AS properties"
        )
    return None


# ---------------------------------------------------------------------------
# Cypher execution
# ---------------------------------------------------------------------------

# Clause keywords of Cypher. Only the read-only subset below may appear in a
# generated query (outside string literals, identifiers in backticks, and comments).
_CLAUSE_KEYWORDS = {
    "MATCH", "OPTIONAL", "WHERE", "WITH", "RETURN", "ORDER", "SKIP", "OFFSET", "LIMIT",
    "UNWIND", "UNION", "CREATE", "MERGE", "DELETE", "DETACH", "SET", "REMOVE", "DROP",
    "LOAD", "CALL", "FOREACH", "USE", "SHOW", "TERMINATE", "ALTER", "GRANT", "DENY",
    "REVOKE", "START", "STOP", "ENABLE", "RENAME", "FINISH", "INSERT", "IMPORT",
}
_ALLOWED_CLAUSES = {"MATCH", "OPTIONAL", "WHERE", "WITH", "RETURN", "ORDER", "SKIP",
                    "OFFSET", "LIMIT", "UNWIND", "UNION"}
_ALLOWED_FIRST = {"MATCH", "OPTIONAL", "WITH", "UNWIND"}
# Plan operators that must never appear, even in a read-only plan.
_FORBIDDEN_OPERATOR = re.compile(r"LoadCSV|ProcedureCall|^Show|Terminate|SystemProcedure", re.IGNORECASE)
QUERY_TIMEOUT_S = 30


def _strip_literals(cypher: str) -> str:
    """Remove string literals, backtick-quoted names, and comments from a query."""
    out, i, n = [], 0, len(cypher)
    while i < n:
        c = cypher[i]
        if c in "'\"`":
            j = i + 1
            while j < n and cypher[j] != c:
                j += 2 if cypher[j] == "\\" else 1
            out.append(" ")
            i = j + 1
        elif cypher.startswith("//", i):
            j = cypher.find("\n", i)
            i = n if j < 0 else j
        elif cypher.startswith("/*", i):
            j = cypher.find("*/", i + 2)
            i = n if j < 0 else j + 2
            out.append(" ")
        else:
            out.append(c)
            i += 1
    return "".join(out)


def validate_read_query(cypher: str) -> None:
    """Accept only read-only MATCH queries; raise ValueError otherwise.

    The query is written by an LLM and may be influenced by the user prompt, so
    it is checked against an allowlist of clauses (outside literals and comments)
    instead of a list of forbidden keywords.
    """
    tokens = [t.upper() for t in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", _strip_literals(cypher))]
    if not tokens or tokens[0] not in _ALLOWED_FIRST:
        raise ValueError("Generated query rejected: it must start with MATCH, OPTIONAL MATCH, WITH or UNWIND.")
    if "apoc" in (t.lower() for t in tokens):
        raise ValueError("Generated query rejected: APOC procedures and functions are not allowed.")
    disallowed = sorted((set(tokens) & _CLAUSE_KEYWORDS) - _ALLOWED_CLAUSES)
    if disallowed:
        raise ValueError(f"Generated query rejected: clauses not allowed: {', '.join(disallowed)}.")


def _plan_operators(plan) -> list[str]:
    """Operator names of an EXPLAIN plan, recursively."""
    if not plan:
        return []
    ops = [plan.get("operatorType", "")]
    for child in plan.get("children", []):
        ops += _plan_operators(child)
    return ops


def run_cypher(cypher: str) -> list[dict]:
    """Execute LLM-generated Cypher read-only and return raw records.

    Three layers: the clause allowlist of validate_read_query; an EXPLAIN check
    that Neo4j classifies the query as read-only ('r') and that its plan has no
    LOAD CSV, procedure, or administration operators; and execution in a
    read-only transaction with a server-side timeout.
    """
    validate_read_query(cypher)

    def _explain(tx):
        result = tx.run("EXPLAIN " + cypher)
        summary = result.consume()
        return summary.query_type, summary.plan

    def _read(tx):
        return tx.run(cypher).data()

    with driver.session(database=database, default_access_mode=READ_ACCESS) as session:
        query_type, plan = session.execute_read(_explain)
        operators = _plan_operators(plan)
        if query_type != "r" or any(_FORBIDDEN_OPERATOR.search(op or "") for op in operators):
            raise ValueError(f"Generated query rejected by the query planner (type {query_type!r}).")
        return session.execute_read(unit_of_work(timeout=QUERY_TIMEOUT_S)(_read))


# ---------------------------------------------------------------------------
# Convert Neo4j records → trace.json seed_nodes format
# ---------------------------------------------------------------------------

def records_to_seed_nodes(records: list[dict]) -> list[dict]:
    """
    Convert query records into the seed_nodes list expected by definability.py.

    Each record must have keys: id, label, properties (as returned by the
    generated Cypher RETURN clause).
    """
    seen: set = set()
    nodes: list[dict] = []

    for rec in records:
        node_id    = rec.get("id")
        label      = rec.get("label", "")
        properties = dict(rec.get("properties") or {})

        if not node_id or node_id in seen:
            continue
        seen.add(node_id)

        properties.pop("embedding", None)   # strip large vector
        properties.pop("retrieval_text", None)

        nodes.append({
            "id":               node_id,
            "label":            label,
            "properties":       properties,
            # Text2Cypher is a direct structural match — no cosine score.
            # Use 1.0 so downstream code that reads similarity_score still works.
            "similarity_score": 1.0,
        })

    return nodes


# ---------------------------------------------------------------------------
# Graph expansion — 1-hop neighbours of seed nodes
# ---------------------------------------------------------------------------

TARGET_RELATIONS = ["DIRECTED", "ACTED_IN", "IN_GENRE"]

def expand_seed_nodes(seed_nodes: list[dict]) -> list[dict]:
    """
    For each seed node fetch its 1-hop neighbours via TARGET_RELATIONS and
    return a list of triplets in the same format used by neo4j_kg_retrieval_llamaindex.py:

        {"source": {id, label, properties},
         "relationship": {source_id, target_id, type, properties},
         "target": {id, label, properties},
         "score": <seed similarity_score>}
    """
    if not seed_nodes:
        return []

    seed_ids = [n["id"] for n in seed_nodes]
    score_by_id = {n["id"]: n.get("similarity_score", 1.0) for n in seed_nodes}

    rel_filter = " OR ".join(f'type(r) = "{r}"' for r in TARGET_RELATIONS)
    query = f"""
    MATCH (src)-[r]-(tgt)
    WHERE src.id IN $ids AND ({rel_filter})
    RETURN src.id          AS src_id,
           labels(src)[0]  AS src_label,
           properties(src) AS src_props,
           type(r)         AS rel_type,
           properties(r)   AS rel_props,
           tgt.id          AS tgt_id,
           labels(tgt)[0]  AS tgt_label,
           properties(tgt) AS tgt_props
    """

    try:
        with driver.session(database=database, default_access_mode=READ_ACCESS) as session:
            records = session.run(query, ids=seed_ids).data()
    except Exception as exc:
        print(f"  Warning: graph expansion failed ({exc}).")
        return []

    triplets = []
    seen: set = set()
    for row in records:
        key = (row["src_id"], row["rel_type"], row["tgt_id"])
        if key in seen:
            continue
        seen.add(key)

        src_props = dict(row["src_props"] or {})
        tgt_props = dict(row["tgt_props"] or {})
        src_props.pop("embedding", None);  src_props.pop("retrieval_text", None)
        tgt_props.pop("embedding", None);  tgt_props.pop("retrieval_text", None)

        triplets.append({
            "source": {
                "id":         row["src_id"],
                "label":      row["src_label"],
                "properties": src_props,
            },
            "relationship": {
                "source_id":  row["src_id"],
                "target_id":  row["tgt_id"],
                "type":       row["rel_type"],
                "properties": dict(row["rel_props"] or {}),
            },
            "target": {
                "id":         row["tgt_id"],
                "label":      row["tgt_label"],
                "properties": tgt_props,
            },
            "score": score_by_id.get(row["src_id"], 1.0),
        })

    return triplets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Text2Cypher retrieval — writes seed nodes to trace.json."
    )
    parser.add_argument("--query", required=True, help="Natural language query")
    parser.add_argument("--model", default="llama3.2", help="Ollama model name")
    parser.add_argument("--out",   default="trace.json",
                        help="Output path (default: trace.json)")
    args = parser.parse_args()

    print(f"Generating Cypher for: {args.query!r}")
    cypher = generate_cypher(args.query, args.model)
    print(f"Generated Cypher:\n{cypher}\n")

    print("Running against Neo4j…")
    records = []
    for attempt in range(3):
        # Validate RETURN clause format before hitting Neo4j
        format_err = _validate_return_clause(cypher)
        if format_err:
            feedback = format_err
            print(f"  Attempt {attempt+1}: invalid RETURN format — {feedback}")
        else:
            try:
                records = run_cypher(cypher)
            except Exception as exc:
                feedback = f"Cypher syntax/runtime error: {exc}"
                print(f"  Attempt {attempt+1} failed: {feedback}")
            else:
                if records:
                    break
                feedback = (
                    "The query executed but returned zero results. "
                    "Common causes:\n"
                    "  • genres is a plain string — use m.genres CONTAINS 'Fantasy' (not 'Fantasy' IN m.genres)\n"
                    "  • countries/languages are lists — use 'United States' IN m.countries (not CONTAINS)\n"
                    "  • year is an integer — use WHERE m.year = 2005 (no quotes)\n"
                    "  • Relationship direction matters: (Director)-[:DIRECTED]->(Movie), "
                    "(Actor)-[:ACTED_IN]->(Movie), (Movie)-[:IN_GENRE]->(Genre)\n"
                    "  • Node labels: Movie, Director, Actor, Person, Genre (case-sensitive)\n"
                    "Try relaxing or broadening the filters."
                )
                print(f"  Attempt {attempt+1}: zero results.")

        if attempt < 2:
            print("  Asking LLM to fix the Cypher…")
            fix_prompt = (
                f"The following Cypher query failed or returned no results:\n```\n{cypher}\n```\n\n"
                f"Problem: {feedback}\n\n"
                "Rewrite the query to fix the issue. "
                "Remember the mandatory RETURN format: "
                "<var>.id AS id, labels(<var>)[0] AS label, properties(<var>) AS properties. "
                "Output ONLY the corrected Cypher."
            )
            try:
                cypher = generate_cypher(fix_prompt, args.model)
                print(f"  Revised Cypher:\n{cypher}\n")
            except Exception as exc:
                print(f"  Could not get revised Cypher: {exc}", file=sys.stderr)
                break

    if not records:
        print("All attempts returned no results.", file=sys.stderr)
        return 1

    # --- Raw cypher results (full node properties, embedding stripped) ---
    cypher_results = []
    for rec in records:
        node_id    = rec.get("id")
        label      = rec.get("label", "")
        properties = dict(rec.get("properties") or {})
        properties.pop("embedding", None)
        properties.pop("retrieval_text", None)
        if node_id:
            cypher_results.append({
                "id":         node_id,
                "label":      label,
                "properties": properties,
            })
    print(f"  → {len(cypher_results)} nodes returned by Cypher")

    seed_nodes = records_to_seed_nodes(records)

    print("Expanding 1 hop from seed nodes…")
    triplets = expand_seed_nodes(seed_nodes)
    print(f"  → {len(triplets)} triplets")

    trace = {
        "query":             args.query,
        "retrieval_mode":    "text2cypher",
        "generated_cypher":  cypher,
        "cypher_results": {
            "count": len(cypher_results),
            "nodes": cypher_results,
        },
        "seed_nodes": {
            "count": len(seed_nodes),
            "nodes": seed_nodes,
        },
        "graph_expansion": {
            "mode":     "paths_up_to_depth_1",
            "count":    len(triplets),
            "triplets": triplets,
        },
    }

    Path(args.out).write_text(
        json.dumps(trace, indent=2, ensure_ascii=False, cls=Neo4jEncoder), encoding="utf-8"
    )
    print(f"  → written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
