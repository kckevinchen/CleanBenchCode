"""One-off: inspect list-shape benchmark cells."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else (
    "label_result/benchmark_inconsistency_20260513T055952Z_gpt5-family.jsonl"
)
list_cells = []
null_cells = []
with open(path) as fh:
    for line in fh:
        rec = json.loads(line)
        if isinstance(rec["corrected_value"], list):
            list_cells.append(rec)
        elif rec["corrected_value"] is None:
            null_cells.append(rec)

print(f"{len(list_cells)} list-shape cells:")
for rec in list_cells:
    print(
        f"  {rec['source']}/{rec['column']} row={rec['row']} "
        f"value={rec['original_value']!r} -> {rec['corrected_value']}"
    )

print(f"\n{len(null_cells)} null corrected_value cells:")
for rec in null_cells:
    print(
        f"  {rec['source']}/{rec['column']} row={rec['row']} "
        f"value={rec['original_value']!r} label={rec['label']} "
        f"clusters={rec['cluster_ids']}"
    )
