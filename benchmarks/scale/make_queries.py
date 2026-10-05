"""Ground-truth queries for the real pool: the 98 PubMed/Zotero tool requests + 12 chit-chat from
benchmarks/data/queries_hard.jsonl, plus requests for every other server. Several tools are acceptable
where servers genuinely overlap."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from pool import load  # noqa: E402

NEW = [
    ("figures_generate_figure|figures_plan_figure", "generate a publication-ready figure illustrating the mechanism of action of remimazolam"),
    ("figures_evaluate_figure|figures_verify_figure", "check my figure against the 8-domain quality checklist"),
    ("figures_composite_figure", "把這三張 panel 合成一張投稿用的圖"),
    ("figures_prepare_publication_image", "resize this image and set it to 300 DPI for journal submission"),
    ("workflow_list_domains", "which workflow domains exist in this workspace?"),
    ("workflow_validate_workflow", "validate the irb-submission workflow and report its issues"),
    ("automl_train_and_wait|automl_submit_automl_job", "train an AutoML model on my dataset and wait until it's done"),
    ("automl_compare_roc_curves", "compare the ROC curves of these two models with a DeLong test"),
    ("automl_kaplan_meier_survival|rde_run_clinical_study", "做 Kaplan-Meier 存活分析並做 log-rank 檢定"),
    ("automl_power_ttest", "power analysis for a two-group t-test with effect size 0.5"),
    ("automl_check_multicollinearity|rde_correlation_matrix", "check VIF multicollinearity among my predictors"),
    ("cgu_cgu_session|cgu_cgu_diverge|cgu_cgu_ideas", "幫我針對『降低術後譫妄』發想一些有創意的點子"),
    ("cgu_cgu_judge", "compare these two ideas pairwise and judge which one is better"),
    ("libre_convert_document", "convert report.docx to PDF"),
    ("libre_read_spreadsheet_data", "read the data in the budget.ods spreadsheet"),
    ("libre_batch_convert_documents", "把資料夾裡所有 .doc 檔批次轉成 .docx"),
    ("paper_draft_action", "start drafting the Methods section of my manuscript"),
    ("paper_save_reference_mcp|paper_reference_action", "save PMID 31452104 with verified metadata into my paper project's references"),
    ("paper_export_document", "export my manuscript as a Word file for submission"),
    ("medagent_get_patient_by_mrn|medagent_search_patient", "find the patient with MRN S6534835 in FHIR"),
    ("medagent_get_lab_observations", "show this patient's latest potassium lab results"),
    ("medagent_create_vital_sign", "幫這位病人記錄一筆血壓 120/80"),
    ("medagent_create_medication_order", "order 1 g IV cefazolin for this patient"),
    ("medcalc_calculate|medcalc_discover", "calculate the CHA2DS2-VASc score for a 75-year-old woman with hypertension"),
    ("medcalc_discover|medcalc_get_related_tools", "有哪些醫學計算器可以用來評估腎功能?"),
    ("nsforge_derivation_start", "start a new derivation of the one-compartment PK model"),
    ("nsforge_derivation_rollback", "roll my derivation back to step 3"),
    ("nsforge_formula_search", "look up the Michaelis-Menten formula"),
    ("nsforge_formula_constants", "取得普朗克常數的數值"),
    ("openevidence_oe_ask", "ask OpenEvidence what the first-line treatment for community-acquired pneumonia is"),
    ("openevidence_oe_history_list", "list my OpenEvidence question history"),
    ("pharmacy_check_drug_interaction|pharmacy_check_multi_drug_interactions", "check the interaction between warfarin and amiodarone"),
    ("pharmacy_calculate_creatinine_clearance|medcalc_calculate", "calculate creatinine clearance for a 70-year-old, 60 kg, creatinine 1.5"),
    ("pharmacy_get_nhi_coverage", "Rivaroxaban 健保有給付嗎?"),
    ("pharmacy_calculate_pediatric_dose|pharmacy_calculate_dose_by_weight", "pediatric dose of amoxicillin for a 15 kg child"),
    ("reaper_set_tempo", "set the project tempo to 120 BPM"),
    ("reaper_add_fx", "add a reverb effect to track 2"),
    ("reaper_render_project", "render the music project to a WAV file"),
    ("reaper_create_track", "在 REAPER 新增一條叫 Vocals 的軌道"),
    ("rde_scan_data_folder|automl_list_available_files", "掃描 data 資料夾,列出可以分析的檔案"),
    ("rde_assess_quality", "assess the data quality of my dataset and detect PII"),
    ("rde_generate_table_one|automl_generate_tableone_directly", "generate a Table 1 of baseline characteristics"),
    ("rde_draft_sample_size_plan", "draft a prospective sample size plan for my study"),
    ("rde_export_report|rde_export_final_report", "export the EDA report to Word"),
    ("rca_rc_start_session", "start a new root cause analysis for last night's medication error"),
    ("rca_rc_ask_why", "ask why again to drill down the 5-Why chain"),
    ("rca_rc_add_cause", "add a cause under 'Method' in the fishbone diagram"),
    ("rca_rc_suggest_hfacs", "建議這個原因的 HFACS 分類"),
    ("sympy_solve_algebraically|nsforge_derivation_solve_for", "solve x^2 - 5x + 6 = 0"),
    ("sympy_matrix_eigenvalues", "compute the eigenvalues of the matrix [[2,1],[1,2]]"),
    ("sympy_dsolve_ode", "solve the ODE y'' + y = 0"),
    ("sympy_calculate_curl", "calculate the curl of the vector field F = (y, -x, 0)"),
    ("sympy_convert_to_units", "把 5 英里換算成公里"),
    ("pharmacy_search_formulary", "search the hospital medication catalog for propofol"),
    ("pharmacy_submit_order", "commit the approved medication plan to the HIS"),
    ("pharmacy_get_drug_warnings", "is metformin contraindicated for a patient with eGFR 25?"),
    ("", "搜尋臺灣目前生效的抗生素政策"),
    ("medcalc_calculate", "calculate eGFR with CKD-EPI 2021: creatinine 1.2, 65-year-old male"),
]

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    tools, server_of, _ = load()
    names = {t["function"]["name"] for t in tools}
    old = [json.loads(l) for l in open(os.path.join(here, "..", "data", "queries_hard.jsonl"), encoding="utf-8")]
    out = []
    for q in old:
        if q["expect"] and not all(e in names for e in q["expect"]):
            print("dropped (tool not in pool):", q["query"])
            continue
        out.append({"query": q["query"], "expect": q["expect"], "lang": q["lang"], "skip": False})
    # Queries whose expectations involve the private servers live next to their dumps ($PRIVATE_TOOLS_DIR).
    priv = {}
    pdir = os.environ.get("PRIVATE_TOOLS_DIR")
    if pdir and os.path.exists(os.path.join(pdir, "queries_private.jsonl")):
        priv = {d["query"]: d["expect"] for d in map(json.loads, open(os.path.join(pdir, "queries_private.jsonl"), encoding="utf-8"))}
    for es, q in NEW:
        exp = [e for e in priv.get(q, es.split("|")) if e in names]
        out.append({"query": q, "expect": exp, "lang": "zh" if any("\u4e00" <= c <= "\u9fff" for c in q) else "en", "skip": not exp})
    missing = [e for es, _ in NEW for e in es.split("|") if e and e not in names]
    assert not missing, missing
    with open(os.path.join(here, "queries_real.jsonl"), "w", encoding="utf-8") as f:
        for i, q in enumerate(out):  # ids stay stable whether or not the private servers are present
            if not q.pop("skip", False):
                f.write(json.dumps(dict(id=i, **q), ensure_ascii=False) + "\n")
    out = [q for q in out if q["expect"] or q["query"] in {x["query"] for x in old if not x["expect"]}]
    pos = [q for q in out if q["expect"]]
    print(len(out), "queries:", len(pos), "tool requests,", len(out) - len(pos), "chit-chat; zh:", sum(q["lang"] == "zh" for q in out))
