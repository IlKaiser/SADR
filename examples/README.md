# Running examples of the paper

Traces of the two questions discussed in the paper, produced on an extended version of the
recommendations graph (about 120,000 nodes, not distributed here):

| Folder | Question |
|---|---|
| `q1/` | "What are the best animated movies from Japan?" |
| `q2/` | "Horror Japanese Movies" |

Each folder contains the dense retrieval trace (`trace.json`), the synthesized query
(`generated_query.txt`), and the nodes it retrieves (`retrieved_nodes.json`); `v2/` holds
the outputs of a later version of the query synthesis. In that graph genres were stored in
`Genre.genre`; the public graph stores them in `Genre.name`.
