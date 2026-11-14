"""Emit paper/tables/results_table.tex from results/statistical_analysis/summary_metrics_with_ci.csv.
Use \\input{tables/results_table.tex} in manuscript.tex so no number is ever typed by hand."""
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
s = pd.read_csv(ROOT / "results/statistical_analysis/summary_metrics_with_ci.csv")
s = s[~s.system_config.str.contains("_no_")]
label = {"E1_Conventional_RAG": "E1: Conventional RAG", "E2_Single_Verifier": "E2: Single Verifier",
         "E3_Multi_Agent": "E3: Multi-Agent RAG", "E4_Proposed_Evidence_Aware": "E4: Evidence-Aware (proposed)"}
def ci(r, m, scale=1, d=1):
    return f"{r[m+'_mean']*scale:.{d}f} [{r[m+'_ci_low']*scale:.{d}f}, {r[m+'_ci_high']*scale:.{d}f}]"
lines = [r"\begin{tabular}{lcccc}", r"\toprule",
         r"System & Accuracy (\%) & Citation prec. ($n$) & Latency (s) & Cost/query (\$) \\", r"\midrule"]
for _, r in s.iterrows():
    lines.append(f"{label.get(r.system_config, r.system_config)} & {ci(r,'answer_accuracy',100)} & "
                 f"{ci(r,'citation_precision',1,3)} ({int(r.citation_defined_n)}) & {ci(r,'latency_sec',1,2)} & "
                 f"{ci(r,'token_cost_usd',1,4)} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
out = ROOT / "paper/tables"; out.mkdir(exist_ok=True); (out / "results_table.tex").write_text("\n".join(lines) + "\n")
print("wrote", out / "results_table.tex")
