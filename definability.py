import json
import os
from neo4j import READ_ACCESS, GraphDatabase, basic_auth
import re
from pathlib import Path

def _cypher_literal(value) -> str:
    """Render a graph value as a Cypher literal, escaping strings.

    The synthesized query is kept as readable text, so constants read from the
    graph are written into it; escaping prevents them from altering the query.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


# Custom JSON encoder for Neo4j types
class Neo4jEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'iso_format'):  # DateTime objects
            return obj.iso_format()
        return super().default(obj)

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

driver = GraphDatabase.driver(
  NEO4J_URI,
  auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD))
database = NEO4J_DATABASE

# ---------------------------------------
# Helper functions for definability
# ---------------------------------------

def getFreshVariable(variables_in_cq, new_id):
    # Given the variables already used in the CQ and the id of a node, return the variable
    # already assigned to that node, or a new variable with the next progressive number.
    maxIndex = -1
    for pair in variables_in_cq:
        varId = pair[0]
        varName = pair[1]
        index = varName.lstrip('n')
        if varId == new_id:
            return varName
        if index.isdigit() and int(index) > maxIndex:
            maxIndex = int(index)
    if maxIndex == -1: new_index = 0
    else: new_index = int(maxIndex) + 1
    return f"n{new_index}"

_node_details_cache: dict = {}
_node_relations_cache: dict = {}

def get_node_undirected_relations(node_id):
    if node_id in _node_relations_cache:
        return _node_relations_cache[node_id]
    with driver.session(database=database, default_access_mode=READ_ACCESS) as session:
        query = """
        MATCH (n)-[r]-(target)
        WHERE n.id = $node_id AND type(r) IN $rel_types
        RETURN type(r) AS rel_type, collect(DISTINCT target.id)[0..$limit] AS target_ids
        """
        result = session.run(query, node_id=node_id,
                             rel_types=target_relations,
                             limit=superconnector_threshold)
        data = result.data()
    _node_relations_cache[node_id] = data
    return data
    
def get_node_exiting_relations(node_id):
    # Return the outgoing relationships of a node: the relationship type and the ids of the
    # related nodes (collect(DISTINCT target.id) groups them by relationship type).
    with driver.session(database=database, default_access_mode=READ_ACCESS) as session:
        query = """
        MATCH (n)-[r]->(target)
        WHERE n.id = $node_id
        RETURN type(r) AS rel_type, collect(DISTINCT target.id) AS target_ids
        """
        result = session.run(query, node_id=node_id)
        return result.data()

def get_node_details(node_id):
    if node_id not in _node_details_cache:
        with driver.session(database=database, default_access_mode=READ_ACCESS) as session:
            query = """
            MATCH (n)
            WHERE n.id = $node_id
            RETURN n.id AS id, labels(n) AS labels, properties(n) AS properties
            """
            result = session.run(query, node_id=node_id)
            _node_details_cache[node_id] = result.data()[0]
    cached = _node_details_cache[node_id]
    # Return a fresh properties dict each time — callers delete keys from it
    return {"id": cached["id"], "labels": list(cached["labels"]), "properties": dict(cached["properties"])}



# ----------------------------
# Compute Definability (UCQ)
# ----------------------------
def compute_disjoint(node, target_constants):
    cq = ""
    found_constants = set()  # constants appearing in this disjunct
    
    # ------- STEP 0: the seed node -------
    node_id = node.get("id")
    node_details = get_node_details(node_id) # dict with keys "id", "labels", "properties"
    
    node_labels = set(node_details["labels"]) & set(target_classes) # keep only the labels of interest
    
    node_properties = node_details["properties"]
    discard_keys = set()
    for key in node_properties.keys():
        isRelevant = False
        for label in node_labels:
            if label in target_attributes.keys() and key in target_attributes.get(label):
                isRelevant = True
        if not isRelevant:
            discard_keys.add(key)
    for key in discard_keys:
        del node_properties[key]

    # build the conjunct for this node
    cq += f" MATCH (n:{'&'.join(node_labels)})\n"

    if node_properties:
        cq += " WHERE"
        noConstant = True
        for key, val in node_properties.items():
            if isinstance(val, list):
                for v in val:
                    found_constants.add(str(v))
            else: found_constants.add(str(val))
            if target_constants == "ALL" or (str(val) in target_constants) or (isinstance(val, list) and any(str(v) in target_constants for v in val)):
                noConstant = False
                if isinstance(val, list) and len(val) > 1:
                    for v in val:
                        if target_constants == "ALL" or v in target_constants:
                            formatted_v = _cypher_literal(v)
                            cq += f" {formatted_v} IN n.{key} AND"
                elif isinstance(val, list) and len(val) == 1:
                    formatted_v = _cypher_literal(val[0])
                    cq += f" {formatted_v} IN n.{key} AND"
                else:
                    formatted_val = _cypher_literal(val)
                    cq += f" n.{key} = {formatted_val} AND"
        cq = cq.rstrip(" AND") # drop the trailing AND
        if noConstant:
            cq = cq.rstrip(" WHERE")
        cq += "\n "

    current_neighborhood = [[node_id, "n", node_id, "n", "__SELF__"]] # neighbourhood of the top-k node
    assigned_variables = [[node_id, "n"]] # one variable per top-k node, initially "n"
    currentVariable = "n"


    
    # ------- STEP N: neighbours up to n_hops -------
    for hop in range(n_hops):
        new_neighbors = []
        for tupla in current_neighborhood:
            old_node_id = tupla[0]
            old_variable = tupla[1]
            node_id = tupla[2]
            currentVariable = tupla[3]
            neighborhood = get_node_undirected_relations(node_id) # dict with keys "rel_type" and "target_ids" (ids of the related nodes)

            for rel in neighborhood:
                rel_label = rel["rel_type"]
                target_ids = rel["target_ids"]
                if rel_label in target_relations:
                    rel_count = 0
                    for id in target_ids:
                        if rel_count >= max_neighbors_per_rel:
                            break
                        if id != old_node_id or rel_label != tupla[4]: # do not go back to the previous node through the same relationship
                            newVariable = getFreshVariable(assigned_variables, id)
                            if [id, newVariable] not in assigned_variables:
                                assigned_variables.append([id, newVariable])
                            cq += f" MATCH ({currentVariable})-[:{rel_label}]-({newVariable})"
                            new_neighbors.append([old_node_id, old_variable, id, newVariable, rel_label])
                            rel_count += 1


        for tupla in new_neighbors:
            
            old_node_id = tupla[0]
            old_variable = tupla[1]
            node_id = tupla[2]
            currentVariable = tupla[3]
            node_details = get_node_details(node_id) # dict with keys "id", "labels", "properties"
            
            node_labels = set(node_details["labels"]) & set(target_classes) # keep only the labels of interest
            
            node_properties = node_details["properties"]
            discard_keys = set()
            for key in node_properties.keys():
                isRelevant = False
                for label in node_labels:
                    if label in target_attributes.keys() and key in target_attributes.get(label):
                        isRelevant = True
                if not isRelevant:
                    discard_keys.add(key)
            for key in discard_keys:
                del node_properties[key]

            # build the conjunct for this node
            cq += f" MATCH ({currentVariable}:{'&'.join(node_labels)})\n"

            if node_properties:
                cq += " WHERE"
                noConstant = True
                for key, val in node_properties.items():
                    if isinstance(val, list):
                        for v in val:
                            found_constants.add(str(v))
                    else: found_constants.add(str(val))
                    if target_constants == "ALL" or (str(val) in target_constants) or (isinstance(val, list) and any(str(v) in target_constants for v in val)):
                        noConstant = False
                        if isinstance(val, list) and len(val) > 1:
                            for v in val:
                                if target_constants == "ALL" or v in target_constants:
                                    formatted_v = _cypher_literal(v)
                                    cq += f" {formatted_v} IN {currentVariable}.{key} AND"
                        elif isinstance(val, list) and len(val) == 1:
                            formatted_v = _cypher_literal(val[0])
                            cq += f" {formatted_v} IN {currentVariable}.{key} AND"
                        else:
                            formatted_val = _cypher_literal(val)
                            cq += f" {currentVariable}.{key} = {formatted_val} AND"
                cq = cq.rstrip(" AND") # drop the trailing AND
                if noConstant: cq = cq.rstrip(" WHERE")
        current_neighborhood = new_neighbors

    return cq, found_constants

def compute_qbe_ucq(ktop_nodes_path, out_dir="results"):
    '''
    Compute the query-by-example (QBE) as a union of conjunctive queries (UCQ) for a given set of nodes in the graph
    Args:
        ktop_nodes_path: Path to the JSON file containing the retrieval result
        n_hops: Maximum exploration depth for query
    Returns:
        str: minimally-complete UCQ for the given nodes, with n_hops max exploration depth
    '''
    # load the JSON file with the top-k nodes
    with open(ktop_nodes_path, 'r', encoding='utf-8') as f:
        retrieval_result = json.load(f)
    # extract the top-k nodes according to the retrieval mode
    retrieval_mode = retrieval_result.get("retrieval_mode", "embedding")
    if retrieval_mode == "text2cypher":
        ktop_nodes = retrieval_result.get("cypher_results", {}).get("nodes", [])
        if not ktop_nodes:
            ktop_nodes = retrieval_result.get("seed_nodes", {}).get("nodes", [])
    else:
        ktop_nodes = retrieval_result.get("seed_nodes", {}).get("nodes", [])

    query_path = Path(out_dir) / "generated_query.txt"
    query_path.parent.mkdir(parents=True, exist_ok=True)
    with open(query_path, 'w') as f:
        pass
    common_constants = set()  # constants appearing in at least beta * k disjuncts
    all_found_constants = {}  # constant -> number of disjuncts in which it appears

    for node in ktop_nodes:
        _, cq_constants = compute_disjoint(node, "ALL")
        for const in cq_constants:
            if const not in all_found_constants.keys():
                all_found_constants[const] = 1
            else: all_found_constants[const]+=1
    for key, val in all_found_constants.items():
        if val > len(ktop_nodes)*filter_constants_threshold: # keep constants shared by more than beta * k disjuncts
            common_constants.add(key)

    for node in ktop_nodes:
        cq, _ = compute_disjoint(node, common_constants)
        with open(query_path, 'a') as f:
            f.write(cq)
            f.write("\n RETURN n")
            if node != ktop_nodes[-1]: # UNION between disjuncts
                f.write("\n\nUNION\n\n")


                
def _simplify_disjunct(cq: str) -> str:
    """
    Keep only the seed-node MATCH+WHERE from a CQ, dropping all neighbor joins.
    Neighbor MATCH chains (n)-[:R]-(nX) cause multi-way Cartesian joins that
    time out; the label+property filter on n alone is fast and sufficient for RAG.
    """
    import re
    seed_match_line = ""
    seed_where_line = ""
    for raw_line in cq.strip().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("RETURN"):
            continue
        # Seed MATCH: "MATCH (n:Label)" — n has a label, not n0/n1/...
        if re.match(r"^MATCH \(n:[A-Za-z]", line):
            seed_match_line = raw_line
        # Seed WHERE: references n. properties and no neighbour variable n<digit>.
        elif line.startswith("WHERE") and not re.search(r"\bn\d+\.", line):
            seed_where_line = raw_line
    parts = [seed_match_line] if seed_match_line else [" MATCH (n)"]
    if seed_where_line:
        parts.append(seed_where_line)
    parts.append(" RETURN n LIMIT 200")
    return "\n".join(parts)


def compute_definability_retrieval(ucq):
    disjuncts = [d.strip() for d in ucq.split("UNION") if d.strip()]
    seen_ids = set()
    result = []
    for cq in disjuncts:
        simple_cq = _simplify_disjunct(cq)
        with driver.session(database=database, default_access_mode=READ_ACCESS) as session:
            raw = session.run(simple_cq).data()
        for record in raw:
            entry = {}
            for key, val in record.items():
                try:
                    node_dict = dict(val)
                    node_dict.pop("embedding", None)
                    node_dict.pop("retrieval_text", None)
                    entry[key] = node_dict
                except (TypeError, ValueError):
                    entry[key] = val
            node_id = entry.get("n", {}).get("id")
            if node_id and node_id not in seen_ids:
                seen_ids.add(node_id)
                result.append(entry)
    return result


def setup_paramenters():
    global target_attributes, target_classes, target_relations, n_hops, superconnector_threshold, filter_constants_threshold, max_neighbors_per_rel
    # Attributes and relationships that may appear in the synthesized query.
    # The extended graph used for the examples in the paper stored genres in
    # Genre.genre; the public recommendations graph stores them in Genre.name.
    # RATED is excluded because it leads to anonymous User nodes.
    target_attributes = {
        "Genre": ["name"],
        "Movie": ["countries", "year"],
        "Director": ["name"],
    }
    target_classes = ["Movie", "Person", "Genre", "Director", "Actor"]
    target_relations = ["DIRECTED", "IN_GENRE", "ACTED_IN"]
    n_hops = 1
    superconnector_threshold = 50
    filter_constants_threshold = 0.45 # threshold 1 for hard filtering, 0 for no filtering
    max_neighbors_per_rel = 2  # max neighbors per relation type per node in a CQ


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Synthesize and run the SADR query from a dense retrieval trace.")
    parser.add_argument("--trace", default="trace.json", help="dense retrieval trace (default: trace.json)")
    parser.add_argument("--out-dir", default="results", help="output folder (default: results)")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    setup_paramenters()
    compute_qbe_ucq(args.trace, out_dir)
    print(f"Computed QBE for the given top-k nodes -> query written to {out_dir / 'generated_query.txt'}")

    ucq = (out_dir / "generated_query.txt").read_text()
    retrieved_nodes = compute_definability_retrieval(ucq)
    with open(out_dir / "retrieved_nodes.json", "w") as f:
        json.dump(retrieved_nodes, f, indent=2, cls=Neo4jEncoder)
    print(f"Computed new retrieval based on the QBE -> retrieval result written to {out_dir / 'retrieved_nodes.json'}")


if __name__ == "__main__":
    main()