"""Summarize the benchmark: reconstruction of the request, expansion, reference systems.

Reports:
- constraint recovery: share of the constants of each gold query (country, genre,
  director, year) that appear in the query SADR reconstructs from the seeds;
- expansion: relevant movies among the seeds and in the SADR result;
- precision, recall and F1 per system and question group, with paired tests;
- a LaTeX table with the same layout as in the thesis.

Usage:
    python benchmark/summarize.py --results benchmark/results --tex benchmark/results/sadr_benchmark.tex
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

logger = logging.getLogger(__name__)

SADR_SYSTEMS = ["dense@5", "sadr_impl", "sadr_full"]
REFERENCE_SYSTEMS = ["dense@N", "text2cypher"]
SYSTEMS = SADR_SYSTEMS + REFERENCE_SYSTEMS
LABELS = {
    "dense@5": "Dense, top-5 (SADR seeds)",
    "dense@N": "Dense, size-matched to SADR",
    "sadr_impl": "SADR as implemented",
    "sadr_full": "SADR with one-hop constraints",
    "text2cypher": "Text2Cypher (Qwen2.5 7B)",
}
GROUPS = ["country+genre", "director", "time", "not-expressible"]
SHORT = {"country+genre": "CG", "director": "Dir", "time": "Time", "not-expressible": "NE"}
GROUP_NAMES = {"country+genre": "country and genre", "director": "director", "time": "time",
               "not-expressible": "not expressible by SADR"}
PAIRS = [("sadr_full", "dense@5"), ("sadr_full", "dense@N"), ("sadr_full", "sadr_impl"),
         ("text2cypher", "sadr_full")]


def paired(df: pd.DataFrame, a: str, b: str, metric: str) -> dict:
    """Mean paired difference, wins/losses and Wilcoxon p-value."""
    x = df[df.system == a].set_index("id")[metric]
    y = df[df.system == b].set_index("id")[metric]
    d = (x - y).dropna()
    p = wilcoxon(x[d.index], y[d.index], zero_method="zsplit").pvalue if (d != 0).any() else 1.0
    return {"a": a, "b": b, "metric": metric, "mean_diff": round(d.mean(), 3),
            "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "p": round(float(p), 4)}


def constraint_recovery(gold: str, ucq: str) -> tuple[int, int]:
    """Constraints of the gold query found in the reconstructed query: (recovered, total).

    String constants (country, genre, director, actor, ...) must appear as constants of
    the reconstructed query; a year constraint (exact, range or upper bound) is recovered
    when the reconstructed query fixes a year that satisfies it.
    """
    ucq_strings = {s.strip() for s in re.findall(r'"([^"]+)"', ucq)}
    ucq_years = [int(y) for y in re.findall(r"year\s*=\s*(\d{4})", ucq)]
    checks = [s in ucq_strings for s in set(re.findall(r"'([^']+)'", gold))]
    checks += [int(y) in ucq_years for y in re.findall(r"m\.year\s*=\s*(\d{4})", gold)]
    checks += [any(int(a) <= y <= int(b) for y in ucq_years)
               for a, b in re.findall(r"(\d{4})\s*<=\s*m\.year\s*<=\s*(\d{4})", gold)]
    checks += [any(y < int(b) for y in ucq_years) for b in re.findall(r"m\.year\s*<\s*(\d{4})", gold)]
    return sum(checks), len(checks)


def main() -> None:
    """Print the summary and write summary.json and the LaTeX table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("benchmark/results"))
    parser.add_argument("--questions", type=Path, default=Path("benchmark/questions.json"))
    parser.add_argument("--tex", type=Path, default=Path("benchmark/results/sadr_benchmark.tex"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    r = pd.DataFrame(json.loads((args.results / "per_question.json").read_text()))
    traces = {t["id"]: t for t in json.loads((args.results / "traces.json").read_text())}
    questions = {q["id"]: q for q in json.loads(args.questions.read_text())}
    main_df = r[r.system.isin(SYSTEMS)]

    # Reconstruction of the request
    rec = pd.DataFrame([{"id": qid, "group": questions[qid]["group"],
                         **dict(zip(("recovered", "total"),
                                    constraint_recovery(questions[qid]["gold"], traces[qid]["ucq"])))}
                        for qid in traces])
    rec["all"] = rec.recovered == rec.total
    rec_group = rec.groupby("group")[["recovered", "total", "all"]].sum().reindex(GROUPS)
    counts = rec.groupby("group").size().reindex(GROUPS)

    # Expansion: relevant movies among the seeds and in the SADR result
    hits = (main_df.assign(hits=(main_df.recall * main_df.gold).round())
            .pivot(index="id", columns="system", values="hits"))
    added = hits["sadr_full"] - hits["dense@5"]

    overall = main_df.groupby("system")[["precision", "recall", "f1", "returned_movies"]].mean().reindex(SYSTEMS)
    by_group = main_df.pivot_table(index="system", columns="group", values="recall").reindex(SYSTEMS)[GROUPS]
    tests = pd.DataFrame([paired(main_df, a, b, m) for a, b in PAIRS for m in ("recall", "f1")])
    errors = sum(1 for t in traces.values() if t["text2cypher_error"])

    logger.info("Constraints recovered by the reconstructed query: %d/%d (%.0f%%); all recovered in %d of %d questions",
                rec.recovered.sum(), rec.total.sum(), 100 * rec.recovered.sum() / rec.total.sum(),
                rec["all"].sum(), len(rec))
    logger.info("%s", rec_group.assign(questions=counts).to_string())
    logger.info("\nRelevant movies per question: seeds %.1f, SADR %.1f; SADR adds relevant movies in %d of %d questions",
                hits["dense@5"].mean(), hits["sadr_full"].mean(), int((added > 0).sum()), len(added))
    logger.info("\nOverall means (%d questions):\n%s", main_df.id.nunique(), overall.round(3))
    logger.info("\nRecall by question group:\n%s", by_group.round(2))
    logger.info("\nPaired comparisons (Wilcoxon signed-rank):\n%s", tests.to_string(index=False))
    logger.info("\nText2Cypher queries failing to execute: %d of %d", errors, len(traces))

    summary = {
        "constraint_recovery": {"recovered": int(rec.recovered.sum()), "total": int(rec.total.sum()),
                                "questions_all_recovered": int(rec["all"].sum()),
                                "by_group": rec_group.assign(questions=counts).reset_index().to_dict("records")},
        "expansion": {"relevant_in_seeds": round(float(hits["dense@5"].mean()), 2),
                      "relevant_in_sadr_full": round(float(hits["sadr_full"].mean()), 2),
                      "questions_with_added_relevant": int((added > 0).sum())},
        "overall": overall.round(4).reset_index().to_dict("records"),
        "recall_by_group": by_group.round(4).reset_index().to_dict("records"),
        "paired_tests": tests.to_dict("records"),
        "text2cypher_errors": errors, "questions": len(traces),
    }
    (args.results / "summary.json").write_text(json.dumps(summary, indent=1, default=int))

    def row(s: str) -> str:
        o, g = overall.loc[s], by_group.loc[s]
        return (f"{LABELS[s]} & {o.precision:.2f} & {o.recall:.2f} & {o.f1:.2f} & {o.returned_movies:.0f} & "
                + " & ".join(f"{g[k]:.2f}" for k in GROUPS) + r" \\")

    legend = (f"{SHORT[GROUPS[0]]}: {GROUP_NAMES[GROUPS[0]]}, {counts[GROUPS[0]]} questions; "
              + "; ".join(f"{SHORT[g]}: {GROUP_NAMES[g]}, {counts[g]}" for g in GROUPS[1:]))
    head = " & ".join(rf"\textbf{{{SHORT[g]}}}" for g in GROUPS)
    rec_cells = " & ".join(f"{int(rec_group.loc[g, 'recovered'])}/{int(rec_group.loc[g, 'total'])}" for g in GROUPS)
    all_cells = " & ".join(f"{int(rec_group.loc[g, 'all'])}/{counts[g]}" for g in GROUPS)
    share = 100 * rec.recovered.sum() / rec.total.sum()
    tex = (
        "% Results from benchmark/summarize.py in https://github.com/IlKaiser/SADR.\n"
        "\\begin{table}[htbp]\n\\centering\n"
        f"\\caption{{Benchmark on the public Neo4j recommendations graph ({len(traces)} questions with "
        "hand-written gold queries). Top: precision, recall and $F_1$ of the retrieved Movie nodes, averaged "
        "over questions, the mean number of movies returned, and the mean recall per question group "
        f"({legend}). Bottom: constraints of the gold query recovered by the query that SADR reconstructs "
        "from the five seeds.}\n"
        "\\label{tab:sadr_benchmark}\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{lcccc|cccc}\n\\toprule\n"
        " & \\multicolumn{4}{c|}{\\textbf{All questions}} & \\multicolumn{4}{c}{\\textbf{Recall per group}} \\\\\n"
        f"\\textbf{{System}} & \\textbf{{P}} & \\textbf{{R}} & \\textbf{{$F_1$}} & \\textbf{{Movies}} & {head} \\\\\n"
        "\\midrule\n" + "\n".join(row(s) for s in SADR_SYSTEMS) + "\n\\midrule\n"
        + "\n".join(row(s) for s in REFERENCE_SYSTEMS) + "\n\\bottomrule\n\\end{tabular}\n\n"
        "\\vspace{0.8em}\n\\begin{tabular}{lcccc|c}\n\\toprule\n"
        f"\\textbf{{Reconstructed query}} & {head} & \\textbf{{All}} \\\\\n\\midrule\n"
        f"Constraints recovered & {rec_cells} & {int(rec.recovered.sum())}/{int(rec.total.sum())} ({share:.0f}\\%) \\\\\n"
        f"Questions with all constraints recovered & {all_cells} & {int(rec['all'].sum())}/{len(rec)} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    args.tex.write_text(tex)
    logger.info("Wrote %s", args.tex)


if __name__ == "__main__":
    main()
