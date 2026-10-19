#!/usr/bin/env python3
"""
Dense retrieval over an existing Neo4j Knowledge Graph with LlamaIndex
PropertyGraphIndex and VectorContextRetriever.

The vector search and the graph expansion are delegated to LlamaIndex rather
than to hand-written Cypher queries.

Pipeline:
  1. Connect to Neo4j through Neo4jPropertyGraphStore.
  2. Attach to the existing graph with PropertyGraphIndex.from_existing()
     (no data is inserted).
  3. Embed the user query.
  4. VectorContextRetriever finds the top-k most similar nodes in the Neo4j
     vector index and expands the graph around them up to `path_depth` hops.
  5. Return a debug JSON with every node and relationship found.

Requirements:
  - Neo4j with the graph loaded (Movie, Person, Genre nodes with the
    __Entity__ label and a string 'id' property).
  - Embeddings stored in the 'embedding' property of the nodes.
  - The vector index 'entity'.
  - A .env file (see .env.example) with the Neo4j credentials.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_index.core.indices import PropertyGraphIndex
from llama_index.core.indices.property_graph import (
    ImplicitPathExtractor,
    PGRetriever,
    VectorContextRetriever,
)
from llama_index.core.graph_stores.types import (
    EntityNode,
    Relation,
    KG_SOURCE_REL,
)
from llama_index.core.schema import QueryBundle
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore


# ----------------------------------------------------------------------
# .env loading (bash format: "export KEY=VALUE")
# ----------------------------------------------------------------------
# The .env file next to this script holds the variables needed to connect
# to Neo4j (URI, user, password, database) and to configure the embedding
# model. It is parsed directly to avoid a dependency on python-dotenv and
# to work on shells without 'source'.
# ----------------------------------------------------------------------


def _load_env_file(env_path: Optional[str] = None) -> None:
    """
    Read a bash-style .env file and add its variables to os.environ.

    Lines may look like:
        export NEO4J_URI="bolt://localhost:7687"
        NEO4J_USER=neo4j
        # comment

    Variables already set in the environment take precedence.

    Args:
        env_path: optional path to the file; defaults to '.env' next to this script.
    """
    import re as _re

    if env_path is None:
        candidate = Path(__file__).with_name(".env")
        if candidate.exists():
            env_path = str(candidate)
        else:
            return

    path = Path(env_path)
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = _re.sub(r"^export\s+", "", line)
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# Load at import time, before Config reads the variables with os.getenv().
_load_env_file()


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
# Every parameter is read from an environment variable; when a variable is
# not set, the default below is used.
# ----------------------------------------------------------------------


def _env_list(name: str, default: str) -> List[str]:
    """Read an environment variable as a comma-separated list."""
    raw = os.getenv(name, default)
    return [x.strip() for x in raw.split(",") if x.strip()]


@dataclass
class Config:
    """
    Configuration of the retriever.

    Each field maps to an environment variable. The defaults match the standard
    setup (local Neo4j, HuggingFace bge-small, top_k=5, one-hop expansion).
    """

    # --- Neo4j connection ---
    # NEO4J_URI: Bolt address of the DBMS (e.g. bolt://localhost:7687)
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    # NEO4J_USER: Neo4j user (usually "neo4j")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    # NEO4J_PASSWORD: DBMS password
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")
    # NEO4J_DATABASE: database name (usually "neo4j")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")

    # --- Embedding model ---
    # EMBED_PROVIDER: "huggingface" (local) or "openai" (requires OPENAI_API_KEY)
    embed_provider: str = os.getenv("EMBED_PROVIDER", "huggingface")
    # EMBED_MODEL_NAME: HuggingFace repository (e.g. "BAAI/bge-small-en-v1.5")
    # or OpenAI model name (e.g. "text-embedding-3-small")
    embed_model_name: str = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    # EMBED_NORMALIZE: normalize vectors to unit length (needed for cosine similarity)
    normalize_embeddings: bool = os.getenv("EMBED_NORMALIZE", "true").lower() in (
        "1", "true", "yes",
    )
    # HF_HOME: optional cache folder for HuggingFace models
    hf_cache_folder: Optional[str] = os.getenv("HF_HOME") or None

    # --- Retrieval parameters ---
    # RETRIEVAL_TOP_K: number of seed nodes taken from the vector search.
    similarity_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    # RETRIEVAL_PATH_DEPTH: depth of the graph expansion (1 = one hop).
    path_depth: int = int(os.getenv("RETRIEVAL_PATH_DEPTH", "1"))

    # INCLUDE_TEXT: also include the original text chunks in the results.
    include_text: bool = os.getenv("INCLUDE_TEXT", "false").lower() in (
        "1", "true", "yes",
    )


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------


def json_dumps(data: Any) -> str:
    """Serialize an object to indented JSON."""
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def build_embed_model(cfg: Config):
    """
    Build the embedding model from the configuration.

    Providers:
    - "huggingface": local model, downloaded and cached on first use.
    - "openai": cloud model, requires OPENAI_API_KEY.

    The model embeds the user query and is used by LlamaIndex for the vector search.

    Args:
        cfg: configuration with embed_provider and embed_model_name.

    Returns:
        A LlamaIndex-compatible embedding model.
    """
    provider = cfg.embed_provider.strip().lower()

    if provider == "huggingface":
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        return HuggingFaceEmbedding(
            model_name=cfg.embed_model_name,
            normalize=cfg.normalize_embeddings,
            cache_folder=cfg.hf_cache_folder,
        )

    if provider == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding
        return OpenAIEmbedding(model=cfg.embed_model_name)

    raise ValueError(
        f"Unsupported EMBED_PROVIDER: {cfg.embed_provider!r}. "
        "Use 'huggingface' or 'openai'."
    )


# ----------------------------------------------------------------------
# LlamaIndexKGRetriever
# ----------------------------------------------------------------------
# Connects to Neo4j through LlamaIndex, attaches to the existing graph
# without inserting data, builds the vector retriever, and runs queries
# with debug output.
# ----------------------------------------------------------------------


class LlamaIndexKGRetriever:
    """
    Wrapper around LlamaIndex PropertyGraphIndex + VectorContextRetriever.

    Components:
    1. Embedding model (text to vectors)
    2. Neo4jPropertyGraphStore (connection to Neo4j)
    3. PropertyGraphIndex (LlamaIndex index over the existing graph)
    4. VectorContextRetriever (vector search + graph expansion)
    5. PGRetriever (container that can combine several retrievers)
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

        # --- Step 1: embedding model (e.g. 384 dimensions) ---
        self.embed_model = build_embed_model(cfg)

        # --- Step 2: connection to Neo4j ---
        self.graph_store = Neo4jPropertyGraphStore(
            username=cfg.neo4j_user,
            password=cfg.neo4j_password,
            url=cfg.neo4j_uri,
            database=cfg.neo4j_database,
        )

        # --- Step 3: attach to the existing graph ---
        # from_existing() does not insert data.
        #
        # kg_extractors=[ImplicitPathExtractor()] avoids a LlamaIndex pitfall:
        # an empty list is falsy, so LlamaIndex would fall back to
        # SimpleLLMPathExtractor, which requires an OpenAI LLM.
        #
        # embed_kg_nodes=False because the embeddings are already stored in
        # the Neo4j nodes (see embed_imdb120k.py / load_movies_dataset.py).
        self.index = PropertyGraphIndex.from_existing(
            property_graph_store=self.graph_store,
            embed_model=self.embed_model,
            embed_kg_nodes=False,
            kg_extractors=[ImplicitPathExtractor()],
        )

        # --- Step 4: VectorContextRetriever ---
        # For a query it:
        #   a) embeds the query;
        #   b) finds the top-k most similar nodes in the vector index 'entity'
        #      (db.index.vector.queryNodes);
        #   c) expands the graph around each seed node up to path_depth hops;
        #   d) returns a list of NodeWithScore.
        self.vector_retriever = VectorContextRetriever(
            self.index.property_graph_store,
            embed_model=self.embed_model,
            include_text=cfg.include_text,
            similarity_top_k=cfg.similarity_top_k,
            path_depth=cfg.path_depth,
        )

        # --- Step 5: PGRetriever container (one sub-retriever for now) ---
        self.retriever = PGRetriever(
            sub_retrievers=[self.vector_retriever],
        )

    def close(self) -> None:
        """Close the Neo4j connection."""
        self.graph_store.close()

    # ---------- Graph inspection ----------

    def get_schema(self) -> str:
        """Return the graph schema (labels, relationship types, properties) as seen by LlamaIndex."""
        return self.graph_store.get_schema_str()

    # ---------- Retrieval with debug output ----------

    @staticmethod
    def _entity_node_to_dict(node: EntityNode) -> Dict[str, Any]:
        """Convert a LlamaIndex EntityNode into a readable dict."""
        return {
            "id": node.name,          # logical id (e.g. "movie_123")
            "label": node.label,      # node type (e.g. "Movie")
            "properties": dict(node.properties) if node.properties else {},
        }

    @staticmethod
    def _relation_to_dict(rel: Relation) -> Dict[str, Any]:
        """Convert a LlamaIndex Relation into a readable dict."""
        return {
            "source_id": rel.source_id,
            "target_id": rel.target_id,
            "type": rel.label,
            "properties": dict(rel.properties) if rel.properties else {},
        }

    def retrieve_debug(self, query_str: str) -> Dict[str, Any]:
        """
        Run the retrieval step by step, as VectorContextRetriever does
        internally, and expose every intermediate result.

        Pipeline:
        1. QUERY EMBEDDING
        2. VECTOR SEARCH: top-k most similar nodes (the seed nodes, with scores)
        3. GRAPH EXPANSION: paths up to path_depth hops around each seed node
           (triples source, relationship, target)

        Args:
            query_str: the user question in natural language.

        Returns:
            Dict with query, embedding, config, seed_nodes, graph_expansion.
        """

        # ============================================================
        # Phase 1: query embedding
        # ============================================================
        # Aggregate embedding of the QueryBundle, as VectorContextRetriever does.
        query_bundle = QueryBundle(query_str=query_str)
        query_embedding = self.embed_model.get_agg_embedding_from_queries(
            query_bundle.embedding_strs
        )

        # Preview of the first 8 dimensions
        embedding_preview = query_embedding[:8] if query_embedding else []
        embedding_dim = len(query_embedding) if query_embedding else 0

        # ============================================================
        # Phase 2: vector search, i.e. the seed nodes
        # ============================================================
        # graph_store.vector_query() is what VectorContextRetriever calls internally.
        # It returns the most similar EntityNodes and their similarity scores.
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self.cfg.similarity_top_k,
        )

        kg_nodes, scores = self.graph_store.vector_query(vector_store_query)

        seed_nodes_output: List[Dict[str, Any]] = []
        for node, score in zip(kg_nodes, scores):
            seed_nodes_output.append({
                **self._entity_node_to_dict(node),
                "similarity_score": score,
            })

        # ============================================================
        # Phase 3: graph expansion around the seeds
        # ============================================================
        # get_rel_map() runs MATCH p=(e)-[r*1..{depth}]-(other) ... and
        # returns triples [EntityNode, Relation, EntityNode].
        triplets = self.graph_store.get_rel_map(
            kg_nodes,
            depth=self.cfg.path_depth,
            limit=30,
            ignore_rels=[KG_SOURCE_REL],
        )

        # Score of a triple: the highest score of the seed nodes it contains,
        # as in LlamaIndex.
        kg_ids = [node.id for node in kg_nodes]
        expansion_output: List[Dict[str, Any]] = []
        for triplet in triplets:
            source_node, relation, target_node = triplet
            score_src = scores[kg_ids.index(source_node.id)] if source_node.id in kg_ids else 0.0
            score_tgt = scores[kg_ids.index(target_node.id)] if target_node.id in kg_ids else 0.0
            triplet_score = max(score_src, score_tgt)

            expansion_output.append({
                "source": self._entity_node_to_dict(source_node),
                "relationship": self._relation_to_dict(relation),
                "target": self._entity_node_to_dict(target_node),
                "score": triplet_score,
            })

        # Sort by decreasing score, as LlamaIndex does
        expansion_output.sort(key=lambda x: x["score"], reverse=True)

        return {
            "query": query_str,
            "query_embedding": {
                "preview": embedding_preview,
                "dimension": embedding_dim,
            },
            "retrieval_config": {
                "similarity_top_k": self.cfg.similarity_top_k,
                "path_depth": self.cfg.path_depth,
                "include_text": self.cfg.include_text,
                "embed_provider": self.cfg.embed_provider,
                "embed_model_name": self.cfg.embed_model_name,
            },
            "seed_nodes": {
                "count": len(seed_nodes_output),
                "nodes": seed_nodes_output,
            },
            "graph_expansion": {
                "mode": f"paths_up_to_depth_{self.cfg.path_depth}",
                "count": len(expansion_output),
                "triplets": expansion_output,
            },
        }


# ----------------------------------------------------------------------
# Command-line interface
# ----------------------------------------------------------------------
#   schema   - print the graph schema as seen by LlamaIndex
#   retrieve - run the retrieval and print or save the debug JSON
# ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser.

    Commands:
        schema:   print the graph schema (labels, relationships, properties).
        retrieve: run vector retrieval + graph expansion.
                  --query (required) the question in natural language
                  --out   (optional) output JSON path; prints to stdout if omitted.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Retrieval from an existing Neo4j KG with LlamaIndex "
            "PropertyGraphIndex + VectorContextRetriever."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "schema",
        help="Print the graph schema as seen by LlamaIndex.",
    )

    p_retrieve = sub.add_parser(
        "retrieve",
        help="Run vector retrieval + graph expansion and write the debug JSON.",
    )
    p_retrieve.add_argument("--query", type=str, required=True)
    p_retrieve.add_argument(
        "--out", type=str, default="",
        help="Optional path of the output JSON file.",
    )

    return parser


def main() -> int:
    """Parse the arguments, connect to Neo4j, run the command, and close the connection."""
    parser = build_parser()
    args = parser.parse_args()

    cfg = Config()
    retriever = LlamaIndexKGRetriever(cfg)

    try:
        if args.command == "schema":
            print(retriever.get_schema())
            return 0

        if args.command == "retrieve":
            result = retriever.retrieve_debug(args.query)
            payload = json_dumps(result)
            if args.out:
                Path(args.out).write_text(payload, encoding="utf-8")
                print(f"Wrote {args.out}")
            else:
                print(payload)
            return 0

        parser.print_help()
        return 2

    finally:
        retriever.close()


# Usage:
#   python neo4j_kg_retrieval_llamaindex.py retrieve --query "..."
if __name__ == "__main__":
    raise SystemExit(main())
