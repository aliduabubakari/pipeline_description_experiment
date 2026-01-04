#!/usr/bin/env python3
"""
Phase 2 (deterministic):
Read PipelineSpec JSONs (Phase 1 outputs) and generate C0–C9 prompt variants.

Improvements (based on Phase 1 outputs):
- Do NOT include Spec Quality Notes in prompt text by default (prevents Airflow leakage).
- Stronger C5 redaction: also redacts identifiers in inputs/outputs/parameters (URLs, connection ids).
- Fixes minor prose grammar in C9.

Example:
  python scripts/render_variants_phase2.py \
    --input-dir ./pipeline_specs \
    --output-dir ./pipeline_variants \
    --write-variant-specs \
    --resume
"""

import re
import json
import argparse
import copy
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------
# Helpers
# ----------------------------

def _ensure_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []

def _ensure_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}

def _sorted_unique_str(items: List[str]) -> List[str]:
    return sorted({s for s in items if isinstance(s, str) and s.strip()})

def _safe_text(x: Any) -> str:
    return x if isinstance(x, str) else ""

def _apply_replacements_case_insensitive(text: str, replacements: List[Tuple[str, str]]) -> str:
    """
    Deterministic replacement. Applies longest-first, case-insensitive.
    """
    if not isinstance(text, str) or not text:
        return text

    reps = sorted(
        [(a, b) for (a, b) in replacements if isinstance(a, str) and a.strip()],
        key=lambda ab: len(ab[0]),
        reverse=True
    )

    out = text
    for needle, repl in reps:
        if len(needle.strip()) < 3:
            continue
        pattern = re.compile(re.escape(needle), flags=re.IGNORECASE)
        out = pattern.sub(repl, out)
    return out

def _fmt_kv_list(kvs: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for kv in kvs:
        if not isinstance(kv, dict):
            continue
        k = kv.get("key")
        v = kv.get("value")
        if not k:
            continue
        if v is None or v == "":
            lines.append(f"{k}")
        else:
            lines.append(f"{k}={v}")
    return lines

def _fmt_artifacts(items: List[Dict[str, Any]], label: str) -> List[str]:
    lines: List[str] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        ident = it.get("identifier")
        desc = it.get("description")
        if not ident:
            continue
        if desc:
            lines.append(f"- {label}: {ident} ({desc})")
        else:
            lines.append(f"- {label}: {ident}")
    return lines

def _render_section(title: str, lines: List[str]) -> str:
    lines = [ln for ln in lines if ln is not None and str(ln).strip() != ""]
    if not lines:
        return ""
    out = [f"{title}:"]
    out.extend(lines)
    out.append("")
    return "\n".join(out)


# ----------------------------
# Redaction token extraction
# ----------------------------

_URL_RE = re.compile(r"https?://[^\s)>,]+")

def _generic_name(stype: str) -> str:
    stype = (stype or "").lower()
    mapping = {
        "database": "<database>",
        "http_api": "<http_api>",
        "object_storage": "<object_storage>",
        "message_queue": "<message_queue>",
        "cloud_service": "<cloud_service>",
        "ftp": "<ftp>",
        "filesystem": "<filesystem>",
        "email": "<email>",
        "other": "<external_system>",
    }
    return mapping.get(stype, "<external_system>")

def build_external_system_replacements(spec: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Build replacements for C5 (External systems):
    - system names -> <type>
    - system identifiers (URLs, connection ids, hosts) -> <type_identifier>
    Also harvest URLs found anywhere in step inputs/parameters as backup.
    """
    reps: List[Tuple[str, str]] = []

    # From external_systems
    for sys in _ensure_list(spec.get("external_systems")):
        if not isinstance(sys, dict):
            continue
        stype = (sys.get("type") or "other").lower()
        name = sys.get("name")
        ident = sys.get("identifier")
        if isinstance(name, str) and name.strip():
            reps.append((name.strip(), _generic_name(stype)))
        if isinstance(ident, str) and ident.strip():
            reps.append((ident.strip(), f"<{stype}_identifier>"))

    # From step-level external_system_refs (if any)
    for st in _ensure_list(spec.get("steps")):
        if not isinstance(st, dict):
            continue
        for r in _ensure_list(st.get("external_system_refs")):
            if not isinstance(r, dict):
                continue
            stype = (r.get("type") or "other").lower()
            name = r.get("name")
            ident = r.get("identifier")
            if isinstance(name, str) and name.strip():
                reps.append((name.strip(), _generic_name(stype)))
            if isinstance(ident, str) and ident.strip():
                reps.append((ident.strip(), f"<{stype}_identifier>"))

    # Extra harvesting: URLs in inputs/outputs/parameters (often appear as api_endpoint)
    for st in _ensure_list(spec.get("steps")):
        if not isinstance(st, dict):
            continue

        for io_key in ["inputs", "outputs"]:
            for item in _ensure_list(st.get(io_key)):
                if not isinstance(item, dict):
                    continue
                ident = item.get("identifier")
                if isinstance(ident, str):
                    for m in _URL_RE.findall(ident):
                        reps.append((m, "<http_api_identifier>"))

        for kv in _ensure_list(st.get("parameters")):
            if not isinstance(kv, dict):
                continue
            v = kv.get("value")
            if isinstance(v, str):
                for m in _URL_RE.findall(v):
                    reps.append((m, "<http_api_identifier>"))

    return reps

def build_data_artifact_replacements(spec: Dict[str, Any]) -> List[Tuple[str, str]]:
    reps: List[Tuple[str, str]] = []
    da = _ensure_dict(spec.get("data_artifacts"))

    for f in _ensure_list(da.get("files")):
        if isinstance(f, dict) and isinstance(f.get("identifier"), str) and f["identifier"].strip():
            reps.append((f["identifier"].strip(), "<file_path>"))
    for t in _ensure_list(da.get("tables")):
        if isinstance(t, dict) and isinstance(t.get("identifier"), str) and t["identifier"].strip():
            reps.append((t["identifier"].strip(), "<table_name>"))
    for b in _ensure_list(da.get("buckets")):
        if isinstance(b, dict) and isinstance(b.get("identifier"), str) and b["identifier"].strip():
            reps.append((b["identifier"].strip(), "<bucket_name>"))
    for tp in _ensure_list(da.get("topics")):
        if isinstance(tp, dict) and isinstance(tp.get("identifier"), str) and tp["identifier"].strip():
            reps.append((tp["identifier"].strip(), "<topic_name>"))

    # Also redact identifiers inside step inputs/outputs (generic)
    for st in _ensure_list(spec.get("steps")):
        if not isinstance(st, dict):
            continue
        for io_key in ["inputs", "outputs"]:
            for item in _ensure_list(st.get(io_key)):
                if not isinstance(item, dict):
                    continue
                ident = item.get("identifier")
                if isinstance(ident, str) and ident.strip():
                    reps.append((ident.strip(), "<data_artifact>"))
    return reps

def build_env_infra_replacements(spec: Dict[str, Any]) -> List[Tuple[str, str]]:
    reps: List[Tuple[str, str]] = []
    nets: List[str] = []
    for st in _ensure_list(spec.get("steps")):
        if not isinstance(st, dict):
            continue
        env = _ensure_dict(st.get("env_infra"))
        for ev in _ensure_list(env.get("env_vars")):
            if isinstance(ev, str) and ev.strip():
                reps.append((ev.strip(), "<ENV_VAR>"))
        for m in _ensure_list(env.get("mounts")):
            if isinstance(m, str) and m.strip():
                reps.append((m.strip(), "<MOUNT>"))
        net = env.get("network")
        if isinstance(net, str) and net.strip():
            nets.append(net.strip())
    for n in _sorted_unique_str(nets):
        reps.append((n, "<NETWORK>"))
    return reps


# ----------------------------
# Redaction application
# ----------------------------

def _redact_text_fields(spec: Dict[str, Any], replacements: List[Tuple[str, str]]) -> Dict[str, Any]:
    """
    Redact common free-text fields (summary/objective/notes/warnings).
    """
    s = copy.deepcopy(spec)

    if "pipeline_name" in s:
        s["pipeline_name"] = _apply_replacements_case_insensitive(_safe_text(s["pipeline_name"]), replacements)

    summary = _ensure_dict(s.get("summary"))
    summary["purpose"] = _apply_replacements_case_insensitive(_safe_text(summary.get("purpose")), replacements)
    summary["notes"] = _apply_replacements_case_insensitive(_safe_text(summary.get("notes")), replacements)
    if summary.get("business_domain") is not None:
        summary["business_domain"] = _apply_replacements_case_insensitive(_safe_text(summary.get("business_domain")), replacements)
    s["summary"] = summary

    for st in _ensure_list(s.get("steps")):
        if not isinstance(st, dict):
            continue
        st["objective"] = _apply_replacements_case_insensitive(_safe_text(st.get("objective")), replacements)
        for io_key in ["inputs", "outputs"]:
            for item in _ensure_list(st.get(io_key)):
                if not isinstance(item, dict):
                    continue
                if item.get("description") is not None:
                    item["description"] = _apply_replacements_case_insensitive(_safe_text(item.get("description")), replacements)

    q = _ensure_dict(s.get("quality"))
    warns = _ensure_list(q.get("extraction_warnings"))
    q["extraction_warnings"] = [
        _apply_replacements_case_insensitive(w, replacements) if isinstance(w, str) else w
        for w in warns
    ]
    s["quality"] = q
    return s

def _redact_identifiers_for_c5(spec: Dict[str, Any], replacements: List[Tuple[str, str]]) -> Dict[str, Any]:
    """
    For C5 only: also redact identifiers in inputs/outputs/parameters that may contain URLs/hosts/conn ids.
    """
    s = copy.deepcopy(spec)
    for st in _ensure_list(s.get("steps")):
        if not isinstance(st, dict):
            continue

        for io_key in ["inputs", "outputs"]:
            for item in _ensure_list(st.get(io_key)):
                if not isinstance(item, dict):
                    continue
                ident = item.get("identifier")
                if isinstance(ident, str) and ident.strip():
                    item["identifier"] = _apply_replacements_case_insensitive(ident, replacements)

        for kv in _ensure_list(st.get("parameters")):
            if not isinstance(kv, dict):
                continue
            v = kv.get("value")
            if isinstance(v, str) and v.strip():
                kv["value"] = _apply_replacements_case_insensitive(v, replacements)

    return s


# ----------------------------
# Rendering
# ----------------------------

def render_structured(
    spec: Dict[str, Any],
    *,
    include_schedule: bool = True,
    include_control_flow: bool = True,
    include_mechanism: bool = True,
    include_parameters: bool = True,
    include_external_system_details: bool = True,
    include_data_artifacts: bool = True,
    include_env_infra: bool = True,
    include_failure_handling: bool = True,
    ordered_steps: bool = True,
    include_quality_notes: bool = False,
    canonical_pipeline_id: Optional[str] = None
) -> str:
    summary = _ensure_dict(spec.get("summary"))
    schedule = _ensure_dict(spec.get("schedule"))
    cf = _ensure_dict(spec.get("control_flow"))
    data_artifacts = _ensure_dict(spec.get("data_artifacts"))

    display_id = canonical_pipeline_id or spec.get("pipeline_id") or "unknown_pipeline_id"
    pipeline_name = spec.get("pipeline_name") or spec.get("pipeline_id") or "Unnamed Pipeline"

    lines: List[str] = []

    # Summary
    sum_lines = [
        f"- Pipeline ID: {display_id}",
        f"- Name: {pipeline_name}",
        f"- Purpose: {summary.get('purpose') or 'N/A'}",
        f"- Execution model: {summary.get('execution_model') or 'unknown'}",
        f"- Topology pattern: {summary.get('topology_pattern') or 'unknown'}",
    ]
    if summary.get("business_domain"):
        sum_lines.append(f"- Business domain: {summary.get('business_domain')}")
    if summary.get("notes"):
        sum_lines.append(f"- Notes: {summary.get('notes')}")
    lines.append(_render_section("Pipeline Summary", sum_lines))

    # Control flow
    if include_control_flow:
        cf_lines: List[str] = []
        edges = _ensure_list(cf.get("edges"))
        if edges:
            cf_lines.append("- Dependencies (directed edges):")
            for e in edges:
                if isinstance(e, dict) and e.get("from") and e.get("to"):
                    cf_lines.append(f"  - {e['from']} -> {e['to']}")
        lines.append(_render_section("Control Flow", cf_lines))

    # Data artifacts
    if include_data_artifacts:
        da_lines: List[str] = []
        da_lines.extend(_fmt_artifacts(_ensure_list(data_artifacts.get("files")), "file"))
        da_lines.extend(_fmt_artifacts(_ensure_list(data_artifacts.get("tables")), "table"))
        da_lines.extend(_fmt_artifacts(_ensure_list(data_artifacts.get("buckets")), "bucket"))
        da_lines.extend(_fmt_artifacts(_ensure_list(data_artifacts.get("topics")), "topic"))
        lines.append(_render_section("Data Artifacts / I-O Identifiers", da_lines))

    # External systems
    if include_external_system_details:
        ext_lines: List[str] = []
        for s in _ensure_list(spec.get("external_systems")):
            if not isinstance(s, dict):
                continue
            stype = s.get("type") or "other"
            name = s.get("name")
            ident = s.get("identifier")
            base = f"- {stype}"
            if name:
                base += f": {name}"
            if ident:
                base += f" ({ident})"
            ext_lines.append(base)
        lines.append(_render_section("External Systems", ext_lines))

    # Steps
    steps = [s for s in _ensure_list(spec.get("steps")) if isinstance(s, dict)]
    if not ordered_steps:
        steps = sorted(steps, key=lambda x: (x.get("step_id") or x.get("name") or ""))

    step_lines: List[str] = []
    for idx, st in enumerate(steps, 1):
        step_id = st.get("step_id") or f"step_{idx:02d}"
        name = st.get("name") or step_id

        if ordered_steps:
            step_lines.append(f"{idx}. {step_id} — {name}")
        else:
            step_lines.append(f"- {step_id} — {name}")

        step_lines.append(f"  - Objective: {st.get('objective') or 'N/A'}")

        if include_mechanism:
            step_lines.append(f"  - Mechanism: {st.get('mechanism') or 'unknown'}")

        if include_data_artifacts:
            ins = _ensure_list(st.get("inputs"))
            outs = _ensure_list(st.get("outputs"))

            def fmt_io(items: List[Dict[str, Any]]) -> List[str]:
                parts = []
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    t = it.get("type") or "data"
                    ident = it.get("identifier")
                    desc = it.get("description")
                    if ident:
                        parts.append(f"{t}:{ident}")
                    elif desc:
                        parts.append(f"{t}:{desc}")
                return parts

            in_parts = fmt_io(ins)
            out_parts = fmt_io(outs)
            if in_parts:
                step_lines.append(f"  - Inputs: {', '.join(in_parts)}")
            if out_parts:
                step_lines.append(f"  - Outputs: {', '.join(out_parts)}")

        if include_parameters:
            kvs = _fmt_kv_list(_ensure_list(st.get("parameters")))
            if kvs:
                step_lines.append("  - Step parameters:")
                for kv in kvs:
                    step_lines.append(f"    - {kv}")

        if include_failure_handling:
            fh = _ensure_dict(st.get("failure_handling"))
            retries = fh.get("retries")
            retry_delay = fh.get("retry_delay")
            if retries is not None or retry_delay:
                step_lines.append("  - Failure handling:")
                if retries is not None:
                    step_lines.append(f"    - retries: {retries}")
                if retry_delay:
                    step_lines.append(f"    - retry delay: {retry_delay}")

        step_lines.append("")

    lines.append(_render_section("Pipeline Steps", step_lines))

    # Scheduling
    if include_schedule:
        sched_lines: List[str] = []
        sched_lines.append(f"- Schedule type: {schedule.get('schedule_type') or 'unknown'}")
        if schedule.get("schedule_expression") is not None:
            sched_lines.append(f"- Schedule expression: {schedule.get('schedule_expression')}")
        if schedule.get("start_date"):
            sched_lines.append(f"- Start date: {schedule.get('start_date')}")
        if schedule.get("catchup_backfill") is not None:
            sched_lines.append(f"- Catchup/backfill: {schedule.get('catchup_backfill')}")
        lines.append(_render_section("Scheduling", sched_lines))

    # Quality notes (OFF by default)
    if include_quality_notes:
        q = _ensure_dict(spec.get("quality"))
        q_lines: List[str] = [f"- Extraction confidence: {q.get('confidence') or 'unknown'}"]
        for w in _ensure_list(q.get("extraction_warnings")):
            if isinstance(w, str) and w.strip():
                q_lines.append(f"- Warning: {w}")
        lines.append(_render_section("Spec Quality Notes", q_lines))

    out = "\n".join([sec for sec in lines if sec.strip()])
    return out.strip() + "\n"


def render_prose_unstructured(spec: Dict[str, Any], canonical_pipeline_id: Optional[str] = None) -> str:
    summary = _ensure_dict(spec.get("summary"))
    schedule = _ensure_dict(spec.get("schedule"))
    cf = _ensure_dict(spec.get("control_flow"))

    pipeline_name = spec.get("pipeline_name") or canonical_pipeline_id or spec.get("pipeline_id") or "Unnamed Pipeline"
    purpose = summary.get("purpose") or "N/A"
    exec_model = summary.get("execution_model") or "unknown"
    topo = summary.get("topology_pattern") or "unknown"

    # Fix grammar here:
    para1 = (
        f"{pipeline_name} is a workflow with execution model {exec_model} and topology pattern {topo}. "
        f"Its purpose is: {purpose}."
    )

    sched_type = schedule.get("schedule_type") or "unknown"
    sched_expr = schedule.get("schedule_expression")
    sched_bits = [f"schedule type is {sched_type}"]
    if sched_expr:
        sched_bits.append(f"expression is {sched_expr}")
    if schedule.get("start_date"):
        sched_bits.append(f"start date is {schedule.get('start_date')}")
    if schedule.get("catchup_backfill") is not None:
        sched_bits.append(f"catchup/backfill is {schedule.get('catchup_backfill')}")
    sched_sentence = "Scheduling: " + ", ".join(sched_bits) + "."

    edges = _ensure_list(cf.get("edges"))
    if edges:
        edge_pairs = []
        for e in edges:
            if isinstance(e, dict) and e.get("from") and e.get("to"):
                edge_pairs.append(f"{e['from']}→{e['to']}")
        cf_sentence = "Control flow includes dependencies such as: " + "; ".join(edge_pairs[:12]) + "."
    else:
        cf_sentence = "Control flow dependencies are not explicitly specified."

    steps = [s for s in _ensure_list(spec.get("steps")) if isinstance(s, dict)]
    step_sents: List[str] = []
    for st in steps:
        sid = st.get("step_id") or "step"
        obj = st.get("objective") or "N/A"
        mech = st.get("mechanism") or "unknown"
        step_sents.append(f"{sid} performs {obj} using mechanism {mech}.")

    para2 = " ".join([sched_sentence, cf_sentence])
    para3 = " ".join(step_sents) if step_sents else "The workflow steps could not be extracted."

    return "\n\n".join([para1.strip(), para2.strip(), para3.strip()]).strip() + "\n"


# ----------------------------
# Masking per class (C0–C9)
# ----------------------------

def mask_C0(spec: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(spec)

def mask_C1_no_scheduling(spec: Dict[str, Any]) -> Dict[str, Any]:
    s = copy.deepcopy(spec)
    s["schedule"] = {"schedule_type": None, "schedule_expression": None, "timezone": None, "start_date": None, "catchup_backfill": None}
    return s

def mask_C2_no_control_flow(spec: Dict[str, Any]) -> Dict[str, Any]:
    s = copy.deepcopy(spec)
    s["control_flow"] = {"edges": [], "parallel_groups": [], "branch_points": [], "gates": []}
    return s

def mask_C3_no_mechanism(spec: Dict[str, Any]) -> Dict[str, Any]:
    s = copy.deepcopy(spec)
    for st in _ensure_list(s.get("steps")):
        if isinstance(st, dict):
            st["mechanism"] = None
    return s

def mask_C4_no_step_parameters(spec: Dict[str, Any]) -> Dict[str, Any]:
    s = copy.deepcopy(spec)
    for st in _ensure_list(s.get("steps")):
        if isinstance(st, dict):
            st["parameters"] = []
    return s

def mask_C5_no_external_system_details(spec: Dict[str, Any]) -> Dict[str, Any]:
    s = copy.deepcopy(spec)
    s["external_systems"] = []
    for st in _ensure_list(s.get("steps")):
        if isinstance(st, dict):
            st["external_system_refs"] = []
    return s

def mask_C6_no_data_artifacts(spec: Dict[str, Any]) -> Dict[str, Any]:
    s = copy.deepcopy(spec)
    s["data_artifacts"] = {"files": [], "tables": [], "buckets": [], "topics": []}
    for st in _ensure_list(s.get("steps")):
        if not isinstance(st, dict):
            continue
        for io_key in ["inputs", "outputs"]:
            redacted = []
            for item in _ensure_list(st.get(io_key)):
                if not isinstance(item, dict):
                    continue
                redacted.append({
                    "type": item.get("type") or ("input" if io_key == "inputs" else "output"),
                    "identifier": None,
                    "description": item.get("description") or None
                })
            st[io_key] = redacted
    return s

def mask_C7_no_env_infra(spec: Dict[str, Any]) -> Dict[str, Any]:
    s = copy.deepcopy(spec)
    for st in _ensure_list(s.get("steps")):
        if isinstance(st, dict):
            st["env_infra"] = {"env_vars": [], "mounts": [], "network": None, "other": []}
    return s

def mask_C8_no_failure_handling(spec: Dict[str, Any]) -> Dict[str, Any]:
    s = copy.deepcopy(spec)
    for st in _ensure_list(s.get("steps")):
        if isinstance(st, dict):
            st["failure_handling"] = {"retries": None, "retry_delay": None, "timeout": None, "alerts": [], "idempotency": []}
    return s


CLASSES = [
    ("C0", "FULL_STRUCTURED", mask_C0),
    ("C1", "NO_SCHEDULING", mask_C1_no_scheduling),
    ("C2", "NO_CONTROL_FLOW", mask_C2_no_control_flow),
    ("C3", "NO_EXECUTION_MECHANISM", mask_C3_no_mechanism),
    ("C4", "NO_STEP_PARAMETERS", mask_C4_no_step_parameters),
    ("C5", "NO_EXTERNAL_SYSTEM_DETAILS", mask_C5_no_external_system_details),
    ("C6", "NO_DATA_ARTIFACT_DETAILS", mask_C6_no_data_artifacts),
    ("C7", "NO_ENV_INFRA_DETAILS", mask_C7_no_env_infra),
    ("C8", "NO_FAILURE_HANDLING_DETAILS", mask_C8_no_failure_handling),
    ("C9", "PROSE_UNSTRUCTURED", mask_C0),  # format-only
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2: deterministically render C0–C9 variants from PipelineSpec JSONs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input-dir", default="./pipeline_specs")
    parser.add_argument("--output-dir", default="./pipeline_variants")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--write-variant-specs", action="store_true")
    parser.add_argument("--include-quality-notes", action="store_true",
                        help="Include Spec Quality Notes in output text prompts (off by default).")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec_files = sorted(in_dir.glob("*.pipelinespec.json"))
    if not spec_files:
        raise SystemExit(f"No *.pipelinespec.json files found in: {in_dir}")

    manifest: Dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "classes": [{"id": cid, "name": name} for (cid, name, _) in CLASSES],
        "pipelines": []
    }

    for idx, sf in enumerate(spec_files, 1):
        if sf.name.upper() == "MANIFEST.JSON":
            continue

        canonical_pipeline_id = sf.name.replace(".pipelinespec.json", "")
        spec = json.loads(sf.read_text(encoding="utf-8"))
        spec_pipeline_id = spec.get("pipeline_id")

        pipe_dir = out_dir / canonical_pipeline_id
        pipe_dir.mkdir(parents=True, exist_ok=True)

        specs_dir = pipe_dir / "specs"
        if args.write_variant_specs:
            specs_dir.mkdir(parents=True, exist_ok=True)

        if args.resume:
            all_exist = True
            for cid, cname, _ in CLASSES:
                if not (pipe_dir / f"{cid}_{cname}.txt").exists():
                    all_exist = False
                    break
            if all_exist:
                print(f"[{idx}/{len(spec_files)}] resume-skip: {canonical_pipeline_id}")
                manifest["pipelines"].append({
                    "canonical_pipeline_id": canonical_pipeline_id,
                    "spec_pipeline_id": spec_pipeline_id,
                    "status": "skipped_resume"
                })
                continue

        print(f"[{idx}/{len(spec_files)}] rendering: {canonical_pipeline_id}")

        # Deterministic redaction tables from original spec
        ext_reps = build_external_system_replacements(spec)
        da_reps = build_data_artifact_replacements(spec)
        env_reps = build_env_infra_replacements(spec)

        variant_records: Dict[str, Any] = {
            "canonical_pipeline_id": canonical_pipeline_id,
            "spec_pipeline_id": spec_pipeline_id,
            "variants": {}
        }

        for cid, cname, mask_fn in CLASSES:
            variant_spec = mask_fn(spec)

            # Apply redaction depending on class
            if cid == "C5":
                variant_spec = _redact_text_fields(variant_spec, ext_reps)
                variant_spec = _redact_identifiers_for_c5(variant_spec, ext_reps)
            elif cid == "C6":
                variant_spec = _redact_text_fields(variant_spec, da_reps)
            elif cid == "C7":
                variant_spec = _redact_text_fields(variant_spec, env_reps)

            # Render
            if cid == "C9":
                txt = render_prose_unstructured(variant_spec, canonical_pipeline_id=canonical_pipeline_id)
            else:
                txt = render_structured(
                    variant_spec,
                    include_schedule=(cid != "C1"),
                    include_control_flow=(cid != "C2"),
                    ordered_steps=(cid != "C2"),
                    include_mechanism=(cid != "C3"),
                    include_parameters=(cid != "C4"),
                    include_external_system_details=(cid != "C5"),
                    include_data_artifacts=(cid != "C6"),
                    include_env_infra=(cid != "C7"),
                    include_failure_handling=(cid != "C8"),
                    include_quality_notes=args.include_quality_notes,
                    canonical_pipeline_id=canonical_pipeline_id
                )

            txt_path = pipe_dir / f"{cid}_{cname}.txt"
            txt_path.write_text(txt, encoding="utf-8")

            spec_path = None
            if args.write_variant_specs:
                spec_path = specs_dir / f"{cid}.json"
                vcopy = copy.deepcopy(variant_spec)
                vcopy["variant"] = {
                    "class_id": cid,
                    "class_name": cname,
                    "canonical_pipeline_id": canonical_pipeline_id,
                    "spec_pipeline_id": spec_pipeline_id
                }
                spec_path.write_text(json.dumps(vcopy, indent=2, ensure_ascii=False), encoding="utf-8")

            variant_records["variants"][cid] = {
                "class_name": cname,
                "prompt_file": str(txt_path),
                "spec_file": str(spec_path) if spec_path else None
            }

        manifest["pipelines"].append(variant_records)

    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 72)
    print("Phase 2 complete")
    print(f"  pipelines processed: {len(manifest['pipelines'])}")
    print(f"  output: {out_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()