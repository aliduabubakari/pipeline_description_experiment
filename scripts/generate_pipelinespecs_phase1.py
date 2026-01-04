#!/usr/bin/env python3
"""
Phase 1 (LLM once per DAG):
Generate an orchestrator-agnostic PipelineSpec JSON for each DAG JSON in ./selected_batch_dags.

UPDATED: Now supports --retry-failed flag to only process previously failed files.
ENHANCED: Added JSON correction using LLM for syntax errors.

Inputs:
JSON files like ./selected_batch_dags/<name>.json containing:
repo, file_path
analysis (topology/tasks/etc.)
sampled_versions[0].code (Airflow DAG code)

Outputs:
./pipeline_specs/<stem>.pipelinespec.json
"""

import os
import re
import json
import time
import copy
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, List, Set, Tuple

from openai import OpenAI

DEFAULT_INPUT_DIR = "./selected_batch_dags"
DEFAULT_OUTPUT_DIR = "./pipeline_specs"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_MAX_CODE_CHARS = 30000
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 8000

REQUEST_SLEEP_SECONDS = 0.35
RETRY_ATTEMPTS = 2
RETRY_BACKOFF = 2.0
CORRECTION_ATTEMPTS = 2

SYSTEM_PROMPT = (
"You are an expert workflow specification extractor.\n"
"Return ONLY a single valid JSON object (no markdown, no commentary).\n"
"The JSON must be an ORCHESTRATOR-AGNOSTIC pipeline specification.\n"
"\n"
"STRICT JSON FORMATTING RULES:\n"
"1. All strings must be enclosed in DOUBLE quotes (\")\n"
"2. NEVER use trailing commas after the last element in arrays or objects\n"
"3. Escape any double quotes inside strings with a backslash: \\\"\n"
"4. Use proper comma separation between array elements and object key-value pairs\n"
"5. Do NOT mention Apache Airflow or Airflow-specific constructs in descriptive fields\n"
"6. If something is unknown, use null or [] and add an item to quality.extraction_warnings\n"
"\n"
"BEFORE RETURNING, VERIFY YOUR JSON:\n"
"- No trailing commas\n"
"- All strings properly quoted and escaped\n"
"- Valid JSON structure (can be parsed by json.loads())\n"
)

JSON_CORRECTION_SYSTEM_PROMPT = (
"You are a JSON syntax correction specialist. Your task is to fix invalid JSON.\n"
"Return ONLY the corrected JSON string, no explanations, no markdown.\n"
"\n"
"COMMON ISSUES TO FIX:\n"
"1. Remove trailing commas in arrays and objects\n"
"2. Ensure all strings use double quotes, not single quotes\n"
"3. Properly escape double quotes within strings: \\\"\n"
"4. Add missing commas between array elements or object properties\n"
"5. Fix unclosed brackets or braces\n"
"6. Ensure all object keys are quoted with double quotes\n"
"\n"
"Preserve all content and structure - only fix syntax errors.\n"
)

PIPELINESPEC_SCHEMA = """
You MUST output JSON matching this schema exactly (keys must exist; use null/[] when unknown):

{
"spec_version": "1.0",
"pipeline_id": string,
"pipeline_name": string,
"source": {
"unique_key": string|null,
"repo": string|null,
"file_path": string|null,
"repo_url": string|null,
"detected_airflow_version": string|null,
"sampled_version": string|null,
"sampled_commit": string|null,
"sampled_time": string|null,
"source_fingerprint_sha256": string
},
"summary": {
"purpose": string,
"business_domain": string|null,
"execution_model": "batch"|"manual"|"event_driven"|"streaming"|"unknown",
"topology_pattern": "linear"|"fan_out_fan_in"|"fan_out_only"|"branch_merge"|"sensor_gated"|"staged_etl"|"mixed"|"unknown",
"notes": string|null
},
"schedule": {
"schedule_type": "manual"|"cron"|"interval"|"event_driven"|"unknown",
"schedule_expression": string|null,
"timezone": string|null,
"start_date": string|null,
"catchup_backfill": boolean|null
},
"control_flow": {
"edges": [
{"from": string, "to": string}
],
"parallel_groups": [
{"group_id": string, "steps": [string]}
],
"branch_points": [
{
"step": string,
"branches": [
{"condition": string, "next_steps": [string]}
],
"merge_step": string|null
}
],
"gates": [
{"type": string, "step": string, "waits_for": string}
]
},
"external_systems": [
{
"type": "database"|"http_api"|"object_storage"|"message_queue"|"cloud_service"|"ftp"|"filesystem"|"email"|"other",
"name": string|null,
"identifier": string|null,
"details": [string],
"auth": [string]
}
],
"data_artifacts": {
"files": [{"identifier": string, "description": string|null}],
"tables": [{"identifier": string, "description": string|null}],
"buckets": [{"identifier": string, "description": string|null}],
"topics": [{"identifier": string, "description": string|null}]
},
"steps": [
{
"step_id": string,
"name": string,
"objective": string,
"mechanism": "python_callable"|"sql_query"|"http_request"|"shell_command"|"container_run"|"spark_job"|"kubernetes_job"|"wait_poll"|"external_workflow_trigger"|"notification"|"unknown",
"inputs": [{"type": string, "identifier": string|null, "description": string|null}],
"outputs": [{"type": string, "identifier": string|null, "description": string|null}],
"external_system_refs": [{"type": string, "name": string|null, "identifier": string|null}],
"parameters": [{"key": string, "value": string|null}],
"env_infra": {
"env_vars": [string],
"mounts": [string],
"network": string|null,
"other": [string]
},
"failure_handling": {
"retries": integer|null,
"retry_delay": string|null,
"timeout": string|null,
"alerts": [string],
"idempotency": [string]
}
}
],
"quality": {
"confidence": "high"|"medium"|"low",
"extraction_warnings": [string]
}
}

Additional guidance:

Keep step.objective high-level (do not embed file paths/table names inside objective; put those in inputs/outputs/data_artifacts/parameters).

Use actual step identifiers from code when possible (e.g., task_id strings). If unavailable, use stable ids like "step_01", "step_02".

For schedule_expression, preserve what you see (e.g., "@daily", "0 1 * * *", None).

For manual trigger workflows, set schedule_type="manual" and schedule_expression=null.

IMPORTANT: Ensure all strings are properly escaped and quoted. No trailing commas.
"""

USER_PROMPT_TEMPLATE = """
Create an orchestrator-agnostic PipelineSpec JSON from the given metadata + code.

{schema}

METADATA CONTEXT (hints only; do not quote Airflow-specific terms in descriptions):

repo: {repo}

repo_url: {repo_url}

file_path: {file_path}

detected_airflow_version: {detected_airflow_version}

dataset_topology_pattern_hint: {pattern_hint}

dataset_task_count_hint: {task_count_hint}

dataset_operator_types_hint: {operator_types_hint}

dataset_high_level_description_hint: {desc_hint}

CODE TO ANALYZE:
{code}
""".strip()

def get_client() -> OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit(
            "ERROR: DEEPSEEK_API_KEY not set. "
            "Set it via: export DEEPSEEK_API_KEY='...'"
        )
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

def extract_latest_code(dag_data: Dict[str, Any]) -> Optional[str]:
    versions = dag_data.get("sampled_versions") or []
    if not versions:
        return None
    return versions[0].get("code")

def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()

def extract_json_object(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Try to extract JSON from text, return (json_text, error_message).
    Enhanced with better error detection.
    """
    t = (text or "").strip()
    
    # Remove common JSON code block markers
    patterns_to_remove = [
        r'^```json\s*',
        r'^```\s*',
        r'\s*```$',
        r'^JSON:\s*',
        r'^Output:\s*',
        r'^Here( is|\'s)? (the )?(JSON|output):\s*'
    ]
    
    for pattern in patterns_to_remove:
        t = re.sub(pattern, '', t, flags=re.IGNORECASE)
    
    t = t.strip()
    
    # Try direct parse first
    try:
        json.loads(t)
        return t, None
    except json.JSONDecodeError as e:
        error_msg = f"JSON error: {e.msg} at line {e.lineno} column {e.colno} (char {e.pos})"
        return t, error_msg

def correct_json_with_llm(
    client: OpenAI,
    model: str,
    broken_json: str,
    error_msg: str,
    temperature: float,
    max_tokens: int
) -> Optional[Dict[str, Any]]:
    """
    Use LLM to correct JSON syntax errors.
    """
    user_prompt = f"""Fix the following JSON string that has syntax errors.

Error message: {error_msg}

Broken JSON:
{broken_json}

IMPORTANT: Return ONLY the corrected JSON string, no other text."""
    
    for attempt in range(CORRECTION_ATTEMPTS):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JSON_CORRECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,  # Low temperature for correction
                max_tokens=max_tokens,
                stream=False,
            )
            content = resp.choices[0].message.content.strip()
            
            # Try to extract and parse the corrected JSON
            json_text, new_error = extract_json_object(content)
            if json_text and not new_error:
                return json.loads(json_text)
            
            # If still invalid, try one more time with the new error
            if new_error:
                print(f"    Correction attempt {attempt + 1} failed: {new_error}")
                if attempt < CORRECTION_ATTEMPTS - 1:
                    error_msg = new_error
                    broken_json = json_text or content
                    time.sleep(1.0)
                    continue
                    
        except Exception as e:
            print(f"    Correction attempt {attempt + 1} failed: {e}")
            if attempt < CORRECTION_ATTEMPTS - 1:
                time.sleep(1.0)
                continue
    
    return None

def call_deepseek_with_correction(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    max_corrections: int = 2
) -> Dict[str, Any]:
    """
    Call DeepSeek with JSON correction fallback.
    """
    last_error: Optional[Exception] = None
    
    for attempt in range(RETRY_ATTEMPTS + 1):
        try:
            print(f"    Generation attempt {attempt + 1}/{RETRY_ATTEMPTS + 1}")
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            content = resp.choices[0].message.content
            
            # Try to extract and parse JSON
            json_text, error_msg = extract_json_object(content)
            
            if json_text and not error_msg:
                # Success - parse and return
                return json.loads(json_text)
            elif json_text and error_msg:
                # JSON extracted but has errors - try to correct
                print(f"    JSON syntax error detected: {error_msg}")
                print(f"    Attempting LLM-based correction...")
                
                corrected = correct_json_with_llm(
                    client=client,
                    model=model,
                    broken_json=json_text,
                    error_msg=error_msg,
                    temperature=0.1,
                    max_tokens=4000  # Lower for correction
                )
                
                if corrected:
                    print(f"    ✓ JSON successfully corrected")
                    return corrected
                else:
                    raise ValueError(f"JSON correction failed: {error_msg}")
            else:
                raise ValueError("Could not extract any JSON from model output")
                
        except Exception as e:
            last_error = e
            if attempt < RETRY_ATTEMPTS:
                sleep_time = RETRY_BACKOFF ** attempt
                print(f"    Retry {attempt + 1}/{RETRY_ATTEMPTS} after {sleep_time}s: {str(e)[:100]}")
                time.sleep(sleep_time)
            else:
                raise RuntimeError(f"DeepSeek failed after retries and corrections: {last_error}")
    
    raise RuntimeError(f"DeepSeek failed after all attempts: {last_error}")

def normalize_spec(spec: Dict[str, Any], pipeline_id: str, source_fingerprint: str,
                  dag_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make sure all top-level keys exist (best-effort).
    """
    s = copy.deepcopy(spec) if isinstance(spec, dict) else {}
    
    s.setdefault("spec_version", "1.0")
    s.setdefault("pipeline_id", pipeline_id)
    s.setdefault("pipeline_name", pipeline_id)
    
    s.setdefault("source", {})
    s["source"].setdefault("unique_key", dag_data.get("unique_key"))
    s["source"].setdefault("repo", dag_data.get("repo"))
    s["source"].setdefault("file_path", dag_data.get("file_path"))
    s["source"].setdefault("repo_url", dag_data.get("repo_url"))
    s["source"].setdefault("detected_airflow_version", dag_data.get("detected_airflow_version"))
    versions = dag_data.get("sampled_versions") or []
    v0 = versions[0] if versions else {}
    s["source"].setdefault("sampled_version", v0.get("version"))
    s["source"].setdefault("sampled_commit", v0.get("commit"))
    s["source"].setdefault("sampled_time", dag_data.get("sampled_time"))
    s["source"]["source_fingerprint_sha256"] = source_fingerprint
    
    s.setdefault("summary", {})
    s["summary"].setdefault("purpose", "")
    s["summary"].setdefault("business_domain", None)
    s["summary"].setdefault("execution_model", "unknown")
    s["summary"].setdefault("topology_pattern", "unknown")
    s["summary"].setdefault("notes", None)
    
    s.setdefault("schedule", {})
    s["schedule"].setdefault("schedule_type", "unknown")
    s["schedule"].setdefault("schedule_expression", None)
    s["schedule"].setdefault("timezone", None)
    s["schedule"].setdefault("start_date", None)
    s["schedule"].setdefault("catchup_backfill", None)
    
    s.setdefault("control_flow", {})
    s["control_flow"].setdefault("edges", [])
    s["control_flow"].setdefault("parallel_groups", [])
    s["control_flow"].setdefault("branch_points", [])
    s["control_flow"].setdefault("gates", [])
    
    s.setdefault("external_systems", [])
    s.setdefault("data_artifacts", {})
    s["data_artifacts"].setdefault("files", [])
    s["data_artifacts"].setdefault("tables", [])
    s["data_artifacts"].setdefault("buckets", [])
    s["data_artifacts"].setdefault("topics", [])
    
    s.setdefault("steps", [])
    
    s.setdefault("quality", {})
    s["quality"].setdefault("confidence", "medium")
    s["quality"].setdefault("extraction_warnings", [])
    
    # Attach generation stamp
    s["generation"] = {
        "generated_at": datetime.now().isoformat() + "Z",
    }
    return s

def build_user_prompt(dag_data: Dict[str, Any], code: str, schema: str,
                     max_code_chars: int) -> str:
    a = dag_data.get("analysis", {}) or {}
    topo = a.get("topology", {}) or {}
    tasks = a.get("tasks", {}) or {}
    
    code_snippet = code[:max_code_chars]
    if len(code) > max_code_chars:
        code_snippet += "\n\n... [CODE TRUNCATED]"
    
    return USER_PROMPT_TEMPLATE.format(
        schema=schema,
        repo=dag_data.get("repo", "N/A"),
        repo_url=dag_data.get("repo_url", "N/A"),
        file_path=dag_data.get("file_path", "N/A"),
        detected_airflow_version=dag_data.get("detected_airflow_version", "N/A"),
        pattern_hint=topo.get("pattern", "unknown"),
        task_count_hint=tasks.get("total_count", "unknown"),
        operator_types_hint=", ".join(tasks.get("operator_types", []) or []),
        desc_hint=a.get("description", "N/A"),
        code=code_snippet
    )

def get_failed_files_from_manifest(manifest_path: Path) -> Set[str]:
    """Extract list of failed input files from MANIFEST.json."""
    if not manifest_path.exists():
        return set()
    
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        failed = set()
        for pipeline in manifest.get("pipelines", []):
            if pipeline.get("status") in ["failed", "failed_no_code"]:
                failed.add(pipeline.get("input_file"))
        return failed
    except Exception as e:
        print(f"Warning: Could not read manifest: {e}")
        return set()

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1: generate orchestrator-agnostic PipelineSpec JSONs using DeepSeek",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-code-chars", type=int, default=DEFAULT_MAX_CODE_CHARS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--resume", action="store_true", help="Skip files if output already exists")
    parser.add_argument("--retry-failed", action="store_true",
                       help="Only process files that failed in previous run (reads from MANIFEST.json)")
    parser.add_argument("--only-files", type=str,
                       help="Comma-separated list of specific input filenames to process")
    parser.add_argument("--no-correction", action="store_true",
                       help="Disable JSON correction (use original retry logic)")
    args = parser.parse_args()
    
    client = get_client()
    
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all JSON files
    all_json_files = sorted(in_dir.glob("*.json"))
    if not all_json_files:
        raise SystemExit(f"No JSON files found in: {in_dir}")
    
    # Filter files based on arguments
    files_to_process: List[Path] = []
    
    if args.retry_failed:
        manifest_path = out_dir / "MANIFEST.json"
        failed_files = get_failed_files_from_manifest(manifest_path)
        if not failed_files:
            print("No failed files found in MANIFEST.json")
            return
        print(f"Retrying {len(failed_files)} failed files from manifest")
        files_to_process = [f for f in all_json_files if f.name in failed_files]
    elif args.only_files:
        only_names = set(f.strip() for f in args.only_files.split(","))
        files_to_process = [f for f in all_json_files if f.name in only_names]
        if not files_to_process:
            print(f"ERROR: None of the specified files found: {only_names}")
            return
        print(f"Processing {len(files_to_process)} specified files")
    else:
        files_to_process = all_json_files
    
    json_files = [f for f in files_to_process if f.name != "SELECTION_SUMMARY.json"]
    
    manifest: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat() + "Z",
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "model": args.model,
        "count_total_files": len(json_files),
        "mode": "retry_failed" if args.retry_failed else ("only_files" if args.only_files else "normal"),
        "pipelines": []
    }
    
    success = 0
    skipped = 0
    failed = 0
    
    for idx, jf in enumerate(json_files, 1):
        stem = jf.stem
        out_path = out_dir / f"{stem}.pipelinespec.json"
        
        if args.resume and out_path.exists() and not args.retry_failed:
            skipped += 1
            manifest["pipelines"].append({
                "input_file": jf.name,
                "pipeline_id": stem,
                "output_file": out_path.name,
                "status": "skipped_resume"
            })
            print(f"[{idx}/{len(json_files)}] resume-skip: {jf.name}")
            continue
        
        try:
            dag_data = json.loads(jf.read_text(encoding="utf-8"))
            code = extract_latest_code(dag_data)
            if not code:
                failed += 1
                manifest["pipelines"].append({
                    "input_file": jf.name,
                    "pipeline_id": stem,
                    "status": "failed_no_code"
                })
                print(f"[{idx}/{len(json_files)}] FAIL (no code): {jf.name}")
                continue
            
            # fingerprint ties the spec to the concrete source snapshot
            source_fingerprint = sha256_text(
                (dag_data.get("unique_key", "") or "") + "\n" +
                (dag_data.get("repo", "") or "") + "\n" +
                (dag_data.get("file_path", "") or "") + "\n" +
                code
            )
            
            user_prompt = build_user_prompt(dag_data, code, PIPELINESPEC_SCHEMA, args.max_code_chars)
            
            print(f"[{idx}/{len(json_files)}] generating pipelinespec: {jf.name}")
            time.sleep(REQUEST_SLEEP_SECONDS)
            
            # Use the new correction-enabled function
            spec = call_deepseek_with_correction(
                client=client,
                model=args.model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )
            
            spec = normalize_spec(spec, pipeline_id=stem, source_fingerprint=source_fingerprint, dag_data=dag_data)
            
            out_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
            success += 1
            
            manifest["pipelines"].append({
                "input_file": jf.name,
                "pipeline_id": stem,
                "output_file": out_path.name,
                "status": "success"
            })
            print(f"  ✓ wrote {out_path}")
            
        except Exception as e:
            failed += 1
            manifest["pipelines"].append({
                "input_file": jf.name,
                "pipeline_id": stem,
                "status": "failed",
                "error": str(e)
            })
            print(f"  ✗ ERROR on {jf.name}: {e}")
    
    # Write or update manifest
    manifest_path = out_dir / "MANIFEST.json"
    if args.retry_failed or args.only_files:
        # Merge with existing manifest
        if manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                # Update pipelines that we just processed
                processed_files = {p["input_file"] for p in manifest["pipelines"]}
                updated_pipelines = []
                for p in existing.get("pipelines", []):
                    if p["input_file"] in processed_files:
                        # Replace with new result
                        updated_pipelines.extend([
                            np for np in manifest["pipelines"] 
                            if np["input_file"] == p["input_file"]
                        ])
                    else:
                        # Keep old result
                        updated_pipelines.append(p)
                existing["pipelines"] = updated_pipelines
                existing["last_updated"] = datetime.now().isoformat() + "Z"
                manifest = existing
            except Exception as e:
                print(f"Warning: Could not merge with existing manifest: {e}")
    
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    
    print("\n" + "=" * 72)
    print("Phase 1 complete")
    print(f"  success: {success}")
    print(f"  skipped: {skipped}")
    print(f"  failed:  {failed}")
    print(f"  output:  {out_dir}")
    print("=" * 72)

if __name__ == "__main__":
    main()