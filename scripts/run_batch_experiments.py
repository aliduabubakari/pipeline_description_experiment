#!/usr/bin/env python3
"""
Batch Experiment Runner
=======================

Key robustness:
- Sanitizes model/provider strings used in paths and filenames.
- Writes a failure stub unified JSON for every failed run (so CSV includes all attempted runs).
- SMART RESUME: skip only if unified JSON exists AND is valid (not 'null', parses to dict).
- Prevents writing 'null' unified JSONs by validating UnifiedEvaluator output.
"""

from __future__ import annotations

import sys
import re
import json
import time
import argparse
import traceback
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.config_loader import load_config, validate_api_keys
from utils.llm_provider import LLMProvider

from scripts.direct_prompting_non_reasoning import DirectDAGGenerator as NonReasoningGenerator
from scripts.direct_prompting_reasoning import ReasoningModelClient, ReasoningDAGGenerator
from scripts.direct_prompting_react_agent import ReActDAGAgent

from scripts.evaluators.semantic_analyzer import SemanticAnalyzer
from scripts.evaluators.unified_evaluator import UnifiedEvaluator

ORCH_CHOICES = ["airflow", "prefect", "dagster"]
MODE_CHOICES = ["non_reasoning", "reasoning", "react"]


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def split_csv_arg(s: str) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def write_json(path: Path, payload: Any) -> None:
    mkdir(path.parent)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def sanitize_for_path(s: str) -> str:
    if s is None:
        return "unknown"
    s = str(s).strip()
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s) or "unknown"


def is_valid_unified_json(path: Path) -> bool:
    """
    Smart resume predicate:
    - file exists
    - not literally "null"
    - parses as JSON object/dict
    - has required top-level keys used by exporter
    """
    try:
        txt = path.read_text(encoding="utf-8").strip()
        if not txt or txt == "null":
            return False
        obj = json.loads(txt)
        if not isinstance(obj, dict):
            return False
        # minimal keys exporter expects
        if "summary" not in obj or "file_path" not in obj or "orchestrator" not in obj:
            return False
        return True
    except Exception:
        return False


def list_prompt_files(prompts_dir: Path, classes: List[str], pipelines: Optional[List[str]] = None) -> List[Path]:
    out: List[Path] = []
    if pipelines:
        pipe_dirs = [prompts_dir / p for p in pipelines]
    else:
        pipe_dirs = [p for p in prompts_dir.iterdir() if p.is_dir()]

    wanted = set(classes) if classes else None

    for pd in sorted(pipe_dirs):
        if not pd.exists() or not pd.is_dir():
            continue
        for f in sorted(pd.glob("C*_*.txt")):
            class_id = f.stem.split("_", 1)[0]
            if wanted is None or class_id in wanted:
                out.append(f)

    return out


def parse_prompt_context(prompt_path: Path) -> Dict[str, str]:
    pipeline_id = prompt_path.parent.name
    variant_stem = prompt_path.stem
    parts = variant_stem.split("_", 1)
    class_id = parts[0]
    class_name = parts[1] if len(parts) > 1 else ""
    return {
        "pipeline_id": pipeline_id,
        "variant_stem": variant_stem,
        "class_id": class_id,
        "class_name": class_name,
    }


def deterministic_rep_dir(base_dir: Path, rep: int) -> Path:
    return base_dir / f"rep_{rep:02d}"


def deterministic_rep_filename(base_filename: str, rep: int) -> str:
    if base_filename.endswith(".py"):
        return base_filename[:-3] + f"__rep{rep:02d}.py"
    return base_filename + f"__rep{rep:02d}.py"


def unify_out_path(eval_root: Path, code_file: Path) -> Path:
    return eval_root / (code_file.as_posix() + ".unified.json")


def semantic_sidecar_path(code_file: Path) -> Path:
    return code_file.with_name(code_file.name + ".semantic.json")


def model_name_from_config(config: Dict[str, Any], model_key: str) -> str:
    provider = config.get("model_settings", {}).get("active_provider")
    if not provider:
        return model_key
    model_cfg = config.get("model_settings", {}).get(provider, {}).get("models", {}).get(model_key, {})
    return model_cfg.get("model_name") or model_key


def provider_from_config(config: Dict[str, Any]) -> str:
    return config.get("model_settings", {}).get("active_provider") or "unknown_provider"


def compute_output_paths(
    *,
    generation_root: Path,
    mode: str,
    orchestrator: str,
    provider: str,
    model_name_raw: str,
    prompt_ctx: Dict[str, str],
    rep: int
) -> Tuple[Path, Path]:
    provider_s = sanitize_for_path(provider)
    model_s = sanitize_for_path(model_name_raw)
    pipeline_id = sanitize_for_path(prompt_ctx["pipeline_id"])
    variant_stem = sanitize_for_path(prompt_ctx["variant_stem"])
    orch_s = sanitize_for_path(orchestrator)

    base_out_dir = generation_root / mode / orch_s / provider_s / model_s / pipeline_id / variant_stem
    out_dir = deterministic_rep_dir(base_out_dir, rep)

    base_filename = f"{pipeline_id}__{variant_stem}__{orch_s}__{model_s}.py"
    filename = deterministic_rep_filename(base_filename, rep)

    code_file = out_dir / filename
    return out_dir, code_file


def write_failure_stub_outputs(
    *,
    out_dir: Path,
    code_file: Path,
    prompt_path: Path,
    prompt_ctx: Dict[str, str],
    orchestrator: str,
    provider: str,
    model_key: str,
    model_name_raw: str,
    mode: str,
    rep: int,
    error: str,
    traceback_str: str,
) -> None:
    mkdir(out_dir)
    stub = (
        "# GENERATION FAILED\n"
        f"# mode={mode} orchestrator={orchestrator} model_key={model_key}\n"
        f"# prompt={prompt_path}\n"
        f"# error={error}\n"
        "pass\n"
    )
    code_file.write_text(stub, encoding="utf-8")

    meta = {
        "run_type": mode,
        "timestamp": datetime.now().isoformat(),
        "input_file": str(prompt_path),
        "pipeline_id": prompt_ctx["pipeline_id"],
        "class_id": prompt_ctx["class_id"],
        "class_name": prompt_ctx["class_name"],
        "variant_stem": prompt_ctx["variant_stem"],
        "orchestrator": orchestrator,
        "model_info": {
            "provider": provider,
            "model_key": model_key,
            "model_name": model_name_raw,
        },
        "repetition": rep,
        "token_usage": None,
        "generation_error": error,
        "generation_traceback": traceback_str,
        "code_file": str(code_file),
    }
    write_json(out_dir / "generation_metadata.json", meta)


def write_failure_stub_unified(
    *,
    eval_root: Path,
    code_file: Path,
    orchestrator: str,
    alpha: float,
    yaml_valid: Optional[bool],
    prompt_ctx: Dict[str, str],
    provider: str,
    model_key: str,
    model_name_raw: str,
    mode: str,
    rep: int,
    error: str,
) -> Path:
    unified_path = unify_out_path(eval_root, code_file)
    mkdir(unified_path.parent)

    payload = {
        "file_path": str(code_file),
        "orchestrator": orchestrator,
        "evaluation_timestamp": datetime.now().isoformat(),
        "alpha": alpha,
        "yaml_valid": True if yaml_valid is None else bool(yaml_valid),
        "static_analysis": None,
        "platform_compliance": None,
        "semantic_analysis": None,
        "summary": {
            "static_score": None,
            "compliance_score": None,
            "combined_score": None,
            "platform_gate_passed": False,
            "passed": False,
            "issues": {"total": 0, "critical": 0, "major": 0, "minor": 0, "info": 0},
            "semantic_fidelity_oracle": None,
            "semantic_fidelity_variant": None,
            "semantic_issues": {"total": 0, "critical": 0, "major": 0, "minor": 0, "info": 0},
        },
        "run_context": {
            "pipeline_id": prompt_ctx["pipeline_id"],
            "variant_stem": prompt_ctx["variant_stem"],
            "class_id": prompt_ctx["class_id"],
            "class_name": prompt_ctx["class_name"],
            "provider": provider,
            "model_key": model_key,
            "model_name": model_name_raw,
            "mode": mode,
            "repetition": rep,
        },
        "error": {"stage": "generation_or_evaluation", "message": error},
    }

    write_json(unified_path, payload)
    return unified_path


def run_non_reasoning(
    *,
    config: Dict[str, Any],
    model_key: str,
    orchestrator: str,
    prompt_path: Path,
    prompt_text: str,
    out_dir: Path,
    code_file: Path,
) -> Dict[str, Any]:
    llm = LLMProvider(config, model_key)
    model_info = llm.get_model_info() or {}
    gen = NonReasoningGenerator(config=config, llm_provider=llm, orchestrator=orchestrator)
    mkdir(out_dir)
    return gen.generate(
        pipeline_description=prompt_text,
        input_path=prompt_path,
        output_root=Path("."),
        output_dir_override=out_dir,
        output_filename_override=code_file.name,
        model_info=model_info,
    )


def run_reasoning(
    *,
    config: Dict[str, Any],
    model_key: str,
    orchestrator: str,
    prompt_path: Path,
    prompt_text: str,
    out_dir: Path,
    code_file: Path,
) -> Dict[str, Any]:
    client = ReasoningModelClient(config, model_key)
    gen = ReasoningDAGGenerator(model_client=client, orchestrator=orchestrator)
    mkdir(out_dir)
    return gen.generate(
        pipeline_description=prompt_text,
        input_path=prompt_path,
        output_root=Path("."),
        output_dir_override=out_dir,
        output_filename_override=code_file.name,
    )


def run_react(
    *,
    config: Dict[str, Any],
    model_key: str,
    orchestrator: str,
    prompt_path: Path,
    prompt_text: str,
    out_dir: Path,
    code_file: Path,
    max_iterations: int,
    prompt_ctx: Dict[str, str],
    rep: int,
) -> Dict[str, Any]:
    llm = LLMProvider(config, model_key)
    model_info = llm.get_model_info() or {}
    mkdir(out_dir)

    agent = ReActDAGAgent(llm_provider=llm, orchestrator=orchestrator, max_iterations=max_iterations)
    run = agent.run(pipeline_description=prompt_text, output_dir=out_dir)

    code_file.write_text(run["final_code"], encoding="utf-8")

    meta = {
        "run_type": "react_agent_non_reasoning",
        "timestamp": datetime.now().isoformat(),
        "input_file": str(prompt_path),
        "pipeline_id": prompt_ctx["pipeline_id"],
        "class_id": prompt_ctx["class_id"],
        "class_name": prompt_ctx["class_name"],
        "variant_stem": prompt_ctx["variant_stem"],
        "orchestrator": orchestrator,
        "model_info": model_info,
        "token_usage": run["token_accounting"],
        "prompt_chars": len(prompt_text),
        "max_iterations": max_iterations,
        "iterations_ran": len(run["trace"]),
        "final_validation": run["final_observation"],
        "final_code_file": str(code_file),
        "repetition": rep,
    }
    write_json(out_dir / "generation_metadata.json", meta)

    return {"run_type": "react", "code_file": str(code_file), "metadata_file": str(out_dir / "generation_metadata.json")}


def run_evaluation(
    *,
    code_file: Path,
    eval_root: Path,
    pipeline_specs_dir: Path,
    pipeline_variants_dir: Path,
    alpha: float,
    yaml_valid: Optional[bool],
    semantic_reference_mode: str,
    bert_model: str,
    semantic_device: Optional[str],
    semantic_max_items: int,
) -> Dict[str, Any]:
    unified_path = unify_out_path(eval_root, code_file)
    mkdir(unified_path.parent)

    ue = UnifiedEvaluator(
        alpha=alpha,
        yaml_valid=yaml_valid,
        semantic_mode="load",
        pipeline_specs_dir=pipeline_specs_dir,
        pipeline_variants_dir=pipeline_variants_dir,
        semantic_reference_mode=semantic_reference_mode,
        bert_model=bert_model,
        bert_device=semantic_device,
        semantic_max_items=semantic_max_items,
    )

    payload = ue.evaluate(code_file)

    # ✅ Prevent writing "null"
    if payload is None:
        raise RuntimeError(f"UnifiedEvaluator returned None for {code_file}")
    if hasattr(payload, "to_dict"):
        payload = payload.to_dict()
    if not isinstance(payload, dict):
        raise TypeError(f"UnifiedEvaluator returned {type(payload)} for {code_file}")

    write_json(unified_path, payload)
    return {"unified_json": str(unified_path), "semantic_sidecar": str(semantic_sidecar_path(code_file))}


def run_exporter(
    *,
    eval_root: Path,
    summary_csv: Optional[str],
    issues_csv: Optional[str],
    gates_csv: Optional[str],
    scores_csv: Optional[str],
) -> None:
    if not summary_csv:
        return

    cmd = [
        sys.executable,
        "scripts/evaluators/export_unified_to_csv.py",
        "--input", str(eval_root),
        "--out", summary_csv,
    ]
    if issues_csv:
        cmd += ["--issues-out", issues_csv]
    if gates_csv:
        cmd += ["--gates-out", gates_csv]
    if scores_csv:
        cmd += ["--scores-out", scores_csv]

    print("Running exporter:", " ".join(cmd))
    subprocess.run(cmd, check=False)


def main():
    ap = argparse.ArgumentParser(description="Run batch experiments (generate + evaluate + export).")
    ap.add_argument("--prompts-dir", default="pipeline_variants")
    ap.add_argument("--pipeline-specs-dir", default="pipeline_specs")
    ap.add_argument("--pipeline-variants-dir", default="pipeline_variants")

    ap.add_argument("--config-nonreasoning", default="config_llm.json")
    ap.add_argument("--config-reasoning", default="config_reasoning_llm.json")

    ap.add_argument("--modes", default="non_reasoning")
    ap.add_argument("--orchestrators", default="airflow,prefect,dagster")
    ap.add_argument("--classes", default="C0,C1,C2,C3,C4,C5,C6,C7,C8,C9")
    ap.add_argument("--pipelines", default="")

    ap.add_argument("--repetitions", type=int, default=1)

    ap.add_argument("--nonreasoning-models", default="")
    ap.add_argument("--reasoning-models", default="")
    ap.add_argument("--react-models", default="")
    ap.add_argument("--react-max-iterations", type=int, default=2)

    ap.add_argument("--generation-root", default="generated_runs")
    ap.add_argument("--eval-root", default="eval_results/unified")

    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--yaml-valid", default="none", choices=["true", "false", "none"])

    ap.add_argument("--semantic", action="store_true")
    ap.add_argument("--semantic-reference-mode", default="both", choices=["oracle", "variant", "both"])
    ap.add_argument("--bert-model", default="microsoft/codebert-base")
    ap.add_argument("--semantic-device", default=None)
    ap.add_argument("--semantic-max-items", type=int, default=30)

    ap.add_argument("--export-csv", default=None)
    ap.add_argument("--export-issues-csv", default=None)
    ap.add_argument("--export-gates-csv", default=None)
    ap.add_argument("--export-scores-csv", default=None)

    ap.add_argument("--sleep-seconds", type=float, default=0.0)
    ap.add_argument("--resume", action="store_true")

    ap.add_argument("--max-prompts", type=int, default=0)
    ap.add_argument("--max-runs", type=int, default=0)

    args = ap.parse_args()

    prompts_dir = Path(args.prompts_dir)
    pipeline_specs_dir = Path(args.pipeline_specs_dir)
    pipeline_variants_dir = Path(args.pipeline_variants_dir)
    generation_root = Path(args.generation_root)
    eval_root = Path(args.eval_root)

    mkdir(generation_root)
    mkdir(eval_root)

    modes = split_csv_arg(args.modes)
    orchestrators = split_csv_arg(args.orchestrators)
    classes = split_csv_arg(args.classes)
    pipelines = split_csv_arg(args.pipelines) if args.pipelines else None

    nonreason_models = split_csv_arg(args.nonreasoning_models)
    reasoning_models = split_csv_arg(args.reasoning_models)
    react_models = split_csv_arg(args.react_models) if args.react_models else nonreason_models

    if args.yaml_valid == "none":
        yaml_valid = None
    elif args.yaml_valid == "true":
        yaml_valid = True
    else:
        yaml_valid = False

    cfg_non = load_config(args.config_nonreasoning)
    cfg_reason = load_config(args.config_reasoning)

    if not validate_api_keys(cfg_non):
        raise SystemExit("Non-reasoning API key validation failed.")
    if not validate_api_keys(cfg_reason):
        raise SystemExit("Reasoning API key validation failed.")

    prompt_files = list_prompt_files(prompts_dir, classes=classes, pipelines=pipelines)
    if args.max_prompts and args.max_prompts > 0:
        prompt_files = prompt_files[: args.max_prompts]
    if not prompt_files:
        raise SystemExit("No prompt files found.")

    semantic_analyzer: Optional[SemanticAnalyzer] = None
    if args.semantic:
        semantic_analyzer = SemanticAnalyzer(
            pipeline_specs_dir=pipeline_specs_dir,
            pipeline_variants_dir=pipeline_variants_dir,
            reference_mode=args.semantic_reference_mode,
            bert_model=args.bert_model,
            device=args.semantic_device,
            max_items=args.semantic_max_items,
        )

    run_id = f"run_{now_ts()}"
    manifest_path = eval_root / f"{run_id}_MANIFEST.json"
    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "started_at": datetime.now().isoformat(),
        "args": vars(args),
        "records": [],
        "errors": [],
    }

    total_attempted = 0
    total_success = 0

    for mode in modes:
        if mode not in MODE_CHOICES:
            raise SystemExit(f"Unknown mode: {mode}")

        if mode == "non_reasoning" and not nonreason_models:
            raise SystemExit("Selected non_reasoning but --nonreasoning-models is empty.")
        if mode == "reasoning" and not reasoning_models:
            raise SystemExit("Selected reasoning but --reasoning-models is empty.")
        if mode == "react" and not react_models:
            raise SystemExit("Selected react but no --react-models and no nonreasoning models.")

        models = nonreason_models if mode == "non_reasoning" else (reasoning_models if mode == "reasoning" else react_models)
        cfg_for_mode = cfg_non if mode in ("non_reasoning", "react") else cfg_reason
        provider = provider_from_config(cfg_for_mode)

        for model_key in models:
            model_name_raw = model_name_from_config(cfg_for_mode, model_key)

            for orch in orchestrators:
                if orch not in ORCH_CHOICES:
                    raise SystemExit(f"Unknown orchestrator: {orch}")

                for prompt_path in prompt_files:
                    prompt_ctx = parse_prompt_context(prompt_path)
                    prompt_text = read_text(prompt_path)

                    for rep in range(1, args.repetitions + 1):
                        if args.max_runs and args.max_runs > 0 and total_attempted >= args.max_runs:
                            break

                        total_attempted += 1

                        out_dir, code_file = compute_output_paths(
                            generation_root=generation_root,
                            mode=mode,
                            orchestrator=orch,
                            provider=provider,
                            model_name_raw=model_name_raw,
                            prompt_ctx=prompt_ctx,
                            rep=rep,
                        )
                        unified_path = unify_out_path(eval_root, code_file)

                        record_base = {
                            "mode": mode,
                            "model_key": model_key,
                            "model_name": model_name_raw,
                            "provider": provider,
                            "orchestrator": orch,
                            "prompt_path": str(prompt_path),
                            **prompt_ctx,
                            "repetition": rep,
                            "out_dir": str(out_dir),
                            "code_file": str(code_file),
                            "unified_json": str(unified_path),
                        }

                        # ✅ SMART RESUME
                        if args.resume and unified_path.exists():
                            if is_valid_unified_json(unified_path):
                                manifest["records"].append({**record_base, "status": "skipped_resume_valid"})
                                write_json(manifest_path, manifest)
                                continue
                            else:
                                # invalid (e.g., "null") -> delete and recompute
                                try:
                                    unified_path.unlink()
                                except Exception:
                                    pass

                        try:
                            # --- GENERATION ---
                            if mode == "non_reasoning":
                                gen_result = run_non_reasoning(
                                    config=cfg_non,
                                    model_key=model_key,
                                    orchestrator=orch,
                                    prompt_path=prompt_path,
                                    prompt_text=prompt_text,
                                    out_dir=out_dir,
                                    code_file=code_file,
                                )
                            elif mode == "reasoning":
                                gen_result = run_reasoning(
                                    config=cfg_reason,
                                    model_key=model_key,
                                    orchestrator=orch,
                                    prompt_path=prompt_path,
                                    prompt_text=prompt_text,
                                    out_dir=out_dir,
                                    code_file=code_file,
                                )
                            else:
                                gen_result = run_react(
                                    config=cfg_non,
                                    model_key=model_key,
                                    orchestrator=orch,
                                    prompt_path=prompt_path,
                                    prompt_text=prompt_text,
                                    out_dir=out_dir,
                                    code_file=code_file,
                                    max_iterations=args.react_max_iterations,
                                    prompt_ctx=prompt_ctx,
                                    rep=rep,
                                )

                            if args.sleep_seconds > 0:
                                time.sleep(args.sleep_seconds)

                            # --- SEMANTIC SIDECAR ---
                            if semantic_analyzer is not None:
                                sem_result = semantic_analyzer.evaluate(code_file)
                                sem_payload = sem_result.to_dict()
                                sem_payload["issues"] = [i.to_dict() for i in sem_result.all_issues]
                                sem_payload["issue_summary"] = {
                                    "total": len(sem_payload["issues"]),
                                    "critical": sum(1 for i in sem_payload["issues"] if i.get("severity") == "critical"),
                                    "major": sum(1 for i in sem_payload["issues"] if i.get("severity") == "major"),
                                    "minor": sum(1 for i in sem_payload["issues"] if i.get("severity") == "minor"),
                                    "info": sum(1 for i in sem_payload["issues"] if i.get("severity") == "info"),
                                }
                                write_json(semantic_sidecar_path(code_file), sem_payload)

                            # --- UNIFIED ---
                            eval_info = run_evaluation(
                                code_file=code_file,
                                eval_root=eval_root,
                                pipeline_specs_dir=pipeline_specs_dir,
                                pipeline_variants_dir=pipeline_variants_dir,
                                alpha=args.alpha,
                                yaml_valid=yaml_valid,
                                semantic_reference_mode=args.semantic_reference_mode,
                                bert_model=args.bert_model,
                                semantic_device=args.semantic_device,
                                semantic_max_items=args.semantic_max_items,
                            )

                            manifest["records"].append({
                                **record_base,
                                "status": "success",
                                "generation": gen_result,
                                "semantic_sidecar": eval_info["semantic_sidecar"],
                                "unified_json": eval_info["unified_json"],
                            })
                            total_success += 1

                        except Exception as e:
                            tb = traceback.format_exc()

                            try:
                                write_failure_stub_outputs(
                                    out_dir=out_dir,
                                    code_file=code_file,
                                    prompt_path=prompt_path,
                                    prompt_ctx=prompt_ctx,
                                    orchestrator=orch,
                                    provider=provider,
                                    model_key=model_key,
                                    model_name_raw=model_name_raw,
                                    mode=mode,
                                    rep=rep,
                                    error=str(e),
                                    traceback_str=tb,
                                )
                                stub_unified = write_failure_stub_unified(
                                    eval_root=eval_root,
                                    code_file=code_file,
                                    orchestrator=orch,
                                    alpha=args.alpha,
                                    yaml_valid=yaml_valid,
                                    prompt_ctx=prompt_ctx,
                                    provider=provider,
                                    model_key=model_key,
                                    model_name_raw=model_name_raw,
                                    mode=mode,
                                    rep=rep,
                                    error=str(e),
                                )
                            except Exception:
                                stub_unified = None

                            manifest["errors"].append({
                                **record_base,
                                "error": str(e),
                                "traceback": tb,
                                "stub_unified_written": str(stub_unified) if stub_unified else None,
                            })

                        finally:
                            write_json(manifest_path, manifest)

    manifest["finished_at"] = datetime.now().isoformat()
    manifest["stats"] = {
        "attempted": total_attempted,
        "success": total_success,
        "errors": len(manifest["errors"]),
        "skipped": len([r for r in manifest["records"] if str(r.get("status", "")).startswith("skipped")]),
    }
    write_json(manifest_path, manifest)

    print(f"\nBatch finished. Manifest: {manifest_path}")
    print(f"Attempted: {total_attempted}")
    print(f"Success:   {total_success}")
    print(f"Errors:    {len(manifest['errors'])}")

    run_exporter(
        eval_root=eval_root,
        summary_csv=args.export_csv,
        issues_csv=args.export_issues_csv,
        gates_csv=args.export_gates_csv,
        scores_csv=args.export_scores_csv,
    )


if __name__ == "__main__":
    main()