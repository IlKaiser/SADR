# Symbolic-Assisted Dense Retriever (SADR)

SADR treats the output of a dense retriever over a Knowledge Graph as a set of noisy
examples of an implicit query, synthesizes a symbolic query (a union of Cypher patterns)
that approximates it, and runs that query over the graph. The result is a retrieved
context that an expert can inspect and that extends the top-k nodes along the attributes
they share.

This repository contains the implementation used in the paper *Enhancing
Retrieval-Augmented Generators with Symbolic Query Definitions* (FLoC 2026 Workshops),
a demo app that compares naive, dense, and SADR-based retrieval-augmented answers, and a
benchmark against dense retrieval and Text2Cypher.

## How it works

1. **Dense retrieval** (`neo4j_kg_retrieval_llamaindex.py`): the question is embedded with
   `BAAI/bge-small-en-v1.5` and the top-k nodes (default k = 5) are taken from the Neo4j
   vector index; the result is written to `trace.json`.
2. **Query synthesis** (`definability.py`): for each seed node, a conjunctive query is built
   from its labels, its attributes, and its one-hop neighbours. Constants shared by more
   than a fraction beta of the seeds (default 0.45) are kept, the others are dropped. The
   union of these queries approximates the implicit query of the dense retriever
   (`results/generated_query.txt`).
3. **Symbolic retrieval**: each disjunct is executed over the graph and the results are
   merged (`results/retrieved_nodes.json`).

`text2cypher.py` is a baseline that asks a local LLM (through Ollama) to write the Cypher
query directly. `app.py` is a Streamlit app that answers a question in three ways (no
context, dense context, SADR context) with a local LLM.

## Requirements

- Python 3.10 or later
- Docker (for Neo4j 5.x Community with the APOC plugin)
- [Ollama](https://ollama.com) for `text2cypher.py`, `app.py`, and the benchmark
  (tested with `qwen2.5:7b-instruct`)

```bash
pip install -r requirements.txt
cp .env.example .env   # then set NEO4J_PASSWORD
```

## Setting up the graph

The scripts use the public Neo4j
[recommendations](https://github.com/neo4j-graph-examples/recommendations) database
(9,125 movies, 28,863 nodes).

```bash
# 1. Download the dump and load it into a Docker volume
mkdir -p dump && curl -L -o dump/neo4j.dump \
  https://github.com/neo4j-graph-examples/recommendations/raw/main/data/recommendations-50.dump
docker volume create sadr-neo4j-data
docker run --rm -v sadr-neo4j-data:/data -v "$PWD/dump":/dumps:ro neo4j:5.26-community \
  neo4j-admin database load neo4j --from-path=/dumps --overwrite-destination=true

# 2. Start Neo4j with APOC (use the password of your .env)
docker run -d --name sadr-neo4j -p 7474:7474 -p 7687:7687 -v sadr-neo4j-data:/data \
  -e NEO4J_AUTH=neo4j/change-me -e NEO4J_PLUGINS='["apoc"]' neo4j:5.26-community

# 3. Add ids, the __Entity__ label, embeddings, and the vector index
python embed_imdb120k.py
```

`load_movies_dataset.py --clean` builds a smaller graph from a public CSV instead, if you
prefer not to use the dump.

## Usage

```bash
# Dense retrieval: writes trace.json
python neo4j_kg_retrieval_llamaindex.py retrieve \
  --query "What are the best animated movies from Japan?" --out trace.json

# SADR: reads trace.json, writes results/generated_query.txt and results/retrieved_nodes.json
python definability.py            # or: --trace path/to/trace.json --out-dir path/to/results

# Text2Cypher baseline
python text2cypher.py --query "Japanese horror movies" --model qwen2.5:7b-instruct

# Demo app
streamlit run app.py
```

The SADR parameters (attributes and relationships that may appear in the query, k, beta,
number of hops, neighbours per relationship) are set in `setup_paramenters()` in
`definability.py`.

## Security

`text2cypher.py` executes Cypher written by an LLM from user input, so a generated query
must pass three checks: it may only contain read clauses (`MATCH`, `OPTIONAL MATCH`,
`WHERE`, `WITH`, `UNWIND`, `RETURN`, `ORDER BY`, `SKIP`, `LIMIT`, `UNION`) and no APOC
calls; `EXPLAIN` must classify it as read-only, with no `LOAD CSV`, procedure, or
administration operator in its plan; and it runs in a read-only transaction with a
30-second timeout, which Neo4j enforces on the server. All other queries in this
repository also run in read-only transactions. For any deployment beyond local
experiments, also connect with a Neo4j user that has read privileges only and set
`dbms.security.allow_csv_import_from_file_urls=false`. The Streamlit app keeps the files
of each browser session in a separate temporary folder.

## Benchmark

`benchmark/` compares SADR with dense retrieval and Text2Cypher on 38 questions with
hand-written gold queries. On these questions, SADR raises recall from 0.05 (its five
seeds) to 0.34, but does not outperform a dense ranking of the same size (0.45) or a
schema-aware Text2Cypher model (F1 0.66 against 0.14). See
[benchmark/README.md](benchmark/README.md) for the setup, the results, and the threats to
validity.

```bash
PYTHONPATH=. python benchmark/run_benchmark.py
python benchmark/summarize.py
```

## Examples from the paper

`examples/` contains the traces of the two running examples of the paper. They were
produced on an extended version of the recommendations graph (about 120,000 nodes) that is
not distributed here, so rerunning the questions on the public graph gives different
nodes.

## Citation

```bibtex
@inproceedings{calamo2026sadr,
  title     = {Enhancing Retrieval-Augmented Generators with Symbolic Query Definitions},
  author    = {Calamo, Marco and Papi, Laura and Console, Marco and Mecella, Massimo and Bianchini, Filippo},
  booktitle = {Workshop on Neuro-Symbolic Intelligence for {LLMs} and Autonomous Agents (FLoC 2026 Workshops)},
  year      = {2026}
}
```

## License

Apache License 2.0, see [LICENSE](LICENSE). The recommendations dataset is distributed by
Neo4j under its own terms.
