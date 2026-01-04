#!/usr/bin/env python3
"""
ReAct Agent (single-agent, non-reasoning LLM) for Workflow Code Generation
========================================================================

This script is a third generation mode for your experiments:
- Baseline non-reasoning: one-shot code generation (direct_prompting_non_reasoning.py)
- Reasoning models: one-shot w/ extraction (direct_prompting_reasoning.py)
- THIS SCRIPT: non-reasoning model + ReAct loop:
    THINK (plan) -> ACT (code) -> OBSERVE (tools) -> repeat (optional)

Key features:
- Update A: Detect docker-based PER INPUT DESCRIPTION (Mechanism: container_run etc.)
- Update B: Deterministic output dir + filename derived from input path
- Metadata: pipeline_id/class_id/class_name/prompt_sha256/tokens split (input/reasoning/output)
- Prompt constraint: honor "Mechanism" mapping in the description

Tools (local, deterministic):
- python_compile: compile() to catch syntax errors
- orchestrator_smoke_check: verify code contains orchestrator indicators
- placeholder_check: detect "<TODO>" and similar placeholders

Usage:
  python scripts/direct_prompting_react_agent.py \
    --config config_llm.json \
    --input pipeline_variants/<pipeline_id>/C0_FULL_STRUCTURED.txt \
    --orchestrator airflow \
    --output-root generated_react \
    --max-iterations 2 \
    --skip-dependency-check
"""

import os
import sys
import re
import json
import hashlib
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, Any, List

# Add parent directory to Python path for importing utils package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Optional dependency checks
def check_dependencies() -> bool:
    missing = []
    try:
        import openai  # noqa: F401
    except ImportError:
        missing.append("openai")
    try:
        import anthropic  # noqa: F401
    except ImportError:
        missing.append("anthropic")
    if missing:
        print(f"Error: missing dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    return True

# Try to import your project utils
try:
    from utils.config_loader import load_config, validate_api_keys
    from utils.llm_provider import LLMProvider
except ImportError:
    print("Unable to import utils modules. Using fallback stubs (for ad-hoc testing only).")

    def load_config(config_path=None):
        if not config_path or not os.path.exists(config_path):
            return {}
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def validate_api_keys(config):
        return True

    class LLMProvider:
        def __init__(self, config, model_key=None):
            self._config = config
            self._model = model_key or "unknown"

        def generate_completion(self, system_prompt, user_prompt):
            content = "# Dummy output; real LLMProvider required.\nprint('Hello')\n"
            tokens = {"input_tokens": 0, "output_tokens": 0}
            return content, tokens

        def get_model_info(self):
            return {"provider": "stub", "model_name": self._model, "model_key": self._model}


ORCHESTRATOR_CHOICES = ["airflow", "prefect", "dagster"]


# ----------------------------
# Determinism + parsing helpers
# ----------------------------

def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()

def sanitize_for_path(s: str) -> str:
    if s is None:
        return "unknown"
    s = str(s).strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s or "unknown"

def parse_input_context(input_path: Path) -> Dict[str, Optional[str]]:
    """
    Expects: pipeline_variants/<pipeline_id>/C5_NO_....txt
    """
    pipeline_id = input_path.parent.name if input_path.parent else None
    stem = input_path.stem  # C5_NO_EXTERNAL_SYSTEM_DETAILS
    class_id, class_name = None, None
    if stem:
        parts = stem.split("_", 1)
        class_id = parts[0] if parts else None
        class_name = parts[1] if len(parts) > 1 else None
    return {
        "pipeline_id": pipeline_id,
        "class_id": class_id,
        "class_name": class_name,
        "variant_stem": stem,
    }

def detect_docker_based(description: str) -> bool:
    text = (description or "").lower()
    return (
        "mechanism: container_run" in text
        or "container_run" in text
        or "docker" in text
        or "container image" in text
    )

def build_deterministic_output_dir(
    output_root: Path,
    orchestrator: str,
    model_info: Dict[str, Any],
    pipeline_id: str,
    variant_stem: str
) -> Path:
    provider = sanitize_for_path(model_info.get("provider") or "unknown_provider")
    model_name = sanitize_for_path(model_info.get("model_name") or model_info.get("model_key") or "unknown_model")
    orchestrator = sanitize_for_path(orchestrator)
    pipeline_id = sanitize_for_path(pipeline_id)
    variant_stem = sanitize_for_path(variant_stem)
    return output_root / orchestrator / provider / model_name / pipeline_id / variant_stem

def build_deterministic_filename(
    pipeline_id: str,
    variant_stem: str,
    orchestrator: str,
    model_info: Dict[str, Any]
) -> str:
    model_name = sanitize_for_path(model_info.get("model_name") or model_info.get("model_key") or "unknown_model")
    pipeline_id = sanitize_for_path(pipeline_id)
    variant_stem = sanitize_for_path(variant_stem)
    orchestrator = sanitize_for_path(orchestrator)
    return f"{pipeline_id}__{variant_stem}__{orchestrator}__{model_name}.py"

def truncate_middle(text: str, max_chars: int = 12000) -> str:
    if not isinstance(text, str):
        return ""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n\n... [TRUNCATED] ...\n\n" + text[-half:]


def extract_code(content: str) -> str:
    """
    Extract clean Python code from LLM response that may include markdown fences.
    """
    if not content:
        return ""
    text = content.strip()

    if "```python" in text:
        return text.split("```python", 1)[1].split("```", 1)[0].strip()
    if "```py" in text:
        return text.split("```py", 1)[1].split("```", 1)[0].strip()
    if "```" in text:
        parts = text.split("```", 1)
        if len(parts) > 1:
            return parts[1].split("```", 1)[0].strip()

    return text


# ----------------------------
# Local deterministic tools (OBSERVE)
# ----------------------------

def tool_python_compile(code: str) -> Dict[str, Any]:
    try:
        compile(code, "<generated_workflow>", "exec")
        return {"ok": True, "error": None}
    except SyntaxError as e:
        return {
            "ok": False,
            "error": f"SyntaxError: {e.msg} (line={e.lineno}, offset={e.offset})",
            "lineno": e.lineno,
            "offset": e.offset
        }
    except Exception as e:
        return {"ok": False, "error": f"CompileError: {type(e).__name__}: {e}"}

def tool_placeholder_check(code: str) -> Dict[str, Any]:
    issues = []
    if "<TODO" in code or "<todo" in code:
        issues.append("Found '<TODO' placeholder(s)")
    if re.search(r"<[^>]+>", code):
        # This can false-positive on typing hints in docstrings, but generally catches template artifacts.
        issues.append("Found angle-bracket placeholder(s) like <...>")
    return {"ok": len(issues) == 0, "issues": issues}

def tool_orchestrator_smoke_check(code: str, orchestrator: str) -> Dict[str, Any]:
    o = orchestrator.lower().strip()
    errors: List[str] = []

    code_lower = code.lower()

    if o == "airflow":
        # Accept classic DAG(...) or @dag TaskFlow
        has_airflow_import = ("airflow" in code_lower)
        has_dag_def = ("dag(" in code_lower) or ("with dag" in code_lower) or ("@dag" in code_lower)
        if not has_airflow_import:
            errors.append("Missing Airflow imports/usage (expected 'airflow' reference).")
        if not has_dag_def:
            errors.append("Missing Airflow DAG definition (expected DAG(...) or @dag).")

    elif o == "prefect":
        has_prefect_import = ("prefect" in code_lower)
        has_flow = ("@flow" in code) or ("@flow(" in code) or ("flow(" in code_lower)
        if not has_prefect_import:
            errors.append("Missing Prefect imports/usage (expected 'prefect' reference).")
        if not has_flow:
            errors.append("Missing Prefect flow definition (expected @flow).")

    elif o == "dagster":
        has_dagster_import = ("dagster" in code_lower)
        has_job = ("@job" in code) or ("definitions(" in code_lower) or ("define_asset_job" in code_lower)
        has_op_or_asset = ("@op" in code) or ("@asset" in code) or ("op(" in code_lower)
        if not has_dagster_import:
            errors.append("Missing Dagster imports/usage (expected 'dagster' reference).")
        if not has_job:
            errors.append("Missing Dagster job/Definitions (expected @job or Definitions).")
        if not has_op_or_asset:
            errors.append("Missing Dagster ops/assets (expected @op or @asset).")

    else:
        errors.append(f"Unknown orchestrator: {orchestrator}")

    return {"ok": len(errors) == 0, "errors": errors}


def run_observation_tools(code: str, orchestrator: str) -> Dict[str, Any]:
    compile_res = tool_python_compile(code)
    smoke_res = tool_orchestrator_smoke_check(code, orchestrator)
    placeholder_res = tool_placeholder_check(code)

    ok = compile_res["ok"] and smoke_res["ok"] and placeholder_res["ok"]
    return {
        "ok": ok,
        "python_compile": compile_res,
        "orchestrator_smoke_check": smoke_res,
        "placeholder_check": placeholder_res
    }


# ----------------------------
# Prompt building (THINK/ACT)
# ----------------------------

MECHANISM_MAPPING_GUIDE = """MECHANISM MAPPING (MUST FOLLOW):
- Mechanism: container_run => run a container image (orchestrator-appropriate container runner)
- Mechanism: http_request  => perform an HTTP request (requests library or orchestrator native)
- Mechanism: sql_query      => execute SQL against a database connection/resource
- Mechanism: shell_command  => run a shell command (subprocess or orchestrator op)
- Mechanism: wait_poll      => implement a wait/sensor/poll loop until condition is met
- Mechanism: notification   => send notification (email/log) as described
- Mechanism: external_workflow_trigger => trigger another workflow/job if described
"""

def orchestrator_guidance(orchestrator: str, docker_based: bool) -> Tuple[str, str]:
    orchestrator = orchestrator.lower().strip()
    docker_hint_common = ""

    if docker_based:
        if orchestrator == "airflow":
            docker_hint_common = "- If Mechanism=container_run, use DockerOperator with image/command/env/mounts/network if available.\n"
        elif orchestrator == "prefect":
            docker_hint_common = "- If Mechanism=container_run, use prefect-docker DockerContainer or safe subprocess runner.\n"
        elif orchestrator == "dagster":
            docker_hint_common = "- If Mechanism=container_run, implement container execution via resource or subprocess.\n"

    if orchestrator == "airflow":
        sys_addon = (
            "You are an expert Apache Airflow developer.\n"
            "Use Airflow 2.x style imports and idioms.\n"
        )
        user_addon = (
            "AIRFLOW CONSTRAINTS:\n"
            "- Use a DAG (DAG(...) or @dag) and explicit dependencies.\n"
            "- Use Sensors for wait_poll gates; BranchPythonOperator for branching.\n"
            f"{docker_hint_common}"
        )
    elif orchestrator == "prefect":
        sys_addon = (
            "You are an expert Prefect 2.x developer.\n"
            "Use @task and @flow.\n"
        )
        user_addon = (
            "PREFECT CONSTRAINTS:\n"
            "- Use from prefect import flow, task.\n"
            "- Use .submit() for parallelism.\n"
            "- Include if __name__ == '__main__': flow() for local execution.\n"
            "- If scheduling is mentioned, include a short comment about deployment schedule.\n"
            f"{docker_hint_common}"
        )
    elif orchestrator == "dagster":
        sys_addon = (
            "You are an expert Dagster developer.\n"
            "Use @op and @job (or Definitions/Assets) idiomatically.\n"
        )
        user_addon = (
            "DAGSTER CONSTRAINTS:\n"
            "- Use @op and @job (or Definitions).\n"
            "- Include if __name__ == '__main__': job.execute_in_process().\n"
            f"{docker_hint_common}"
        )
    else:
        sys_addon = "You are an expert workflow developer.\n"
        user_addon = "Use idiomatic constructs for the requested orchestrator.\n"

    return sys_addon, user_addon


def build_think_prompts(
    orchestrator: str,
    pipeline_description: str,
    iteration: int,
    history: Dict[str, Any],
) -> Tuple[str, str]:
    docker_based = detect_docker_based(pipeline_description)
    sys_addon, user_addon = orchestrator_guidance(orchestrator, docker_based)

    system_prompt = (
        f"{sys_addon}"
        "You are running in a ReAct agent loop. This is the THINK step.\n"
        "Return ONLY plain text. Do NOT output Python code.\n"
    )

    prev_obs = history.get("last_observation")
    prev_code_summary = history.get("last_code_summary")  # short text

    if iteration == 1:
        user_prompt = (
            "THINK STEP (Iteration 1):\n"
            "Create a detailed implementation plan for converting the pipeline description into code.\n"
            "Your plan should include:\n"
            "- Proposed step/task/op function names\n"
            "- How to wire dependencies (including branching/parallelism/gates)\n"
            "- How to implement each Mechanism type\n"
            "- Safe defaults for missing values\n"
            "- Any pitfalls to avoid\n\n"
            f"{user_addon}\n"
            f"{MECHANISM_MAPPING_GUIDE}\n"
            "PIPELINE DESCRIPTION:\n"
            f"{pipeline_description}\n"
        )
    else:
        user_prompt = (
            f"THINK STEP (Iteration {iteration}):\n"
            "You previously generated code that failed validation.\n"
            "Given the observation results, produce a concrete fix plan.\n"
            "Do NOT output code.\n\n"
            f"{user_addon}\n"
            f"{MECHANISM_MAPPING_GUIDE}\n"
            "PREVIOUS CODE SUMMARY:\n"
            f"{prev_code_summary}\n\n"
            "OBSERVATIONS (validation results):\n"
            f"{json.dumps(prev_obs, indent=2)}\n\n"
            "PIPELINE DESCRIPTION:\n"
            f"{pipeline_description}\n"
        )

    return system_prompt, user_prompt


def build_act_prompts(
    orchestrator: str,
    pipeline_description: str,
    iteration: int,
    plan_text: str,
    history: Dict[str, Any],
) -> Tuple[str, str]:
    docker_based = detect_docker_based(pipeline_description)
    sys_addon, user_addon = orchestrator_guidance(orchestrator, docker_based)

    system_prompt = (
        f"{sys_addon}"
        "You are running in a ReAct agent loop. This is the ACT step.\n"
        "Return ONLY valid Python code for a single module. No markdown, no explanations.\n"
    )

    prev_code = history.get("last_code")
    prev_obs = history.get("last_observation")

    if iteration == 1:
        user_prompt = (
            "ACT STEP (Iteration 1):\n"
            "Generate the complete orchestrator code now.\n\n"
            f"{user_addon}\n"
            f"{MECHANISM_MAPPING_GUIDE}\n"
            "IMPLEMENTATION PLAN (from THINK step):\n"
            f"{plan_text}\n\n"
            "PIPELINE DESCRIPTION:\n"
            f"{pipeline_description}\n"
        )
    else:
        user_prompt = (
            f"ACT STEP (Iteration {iteration}):\n"
            "Regenerate a corrected FULL Python module that fixes the issues described in OBSERVATIONS.\n"
            "Return ONLY the complete corrected Python code.\n\n"
            f"{user_addon}\n"
            f"{MECHANISM_MAPPING_GUIDE}\n"
            "FIX PLAN (from THINK step):\n"
            f"{plan_text}\n\n"
            "OBSERVATIONS (validation results):\n"
            f"{json.dumps(prev_obs, indent=2)}\n\n"
            "PREVIOUS CODE (for reference, may be truncated):\n"
            f"{truncate_middle(prev_code or '', 12000)}\n\n"
            "PIPELINE DESCRIPTION:\n"
            f"{pipeline_description}\n"
        )

    return system_prompt, user_prompt


def summarize_code_for_prompt(code: str, max_lines: int = 60) -> str:
    """
    Short summary to avoid huge token usage in THINK step for later iterations.
    """
    if not code:
        return "N/A"
    lines = code.splitlines()
    head = lines[:max_lines]
    return "\n".join(head)


# ----------------------------
# ReAct Agent runner
# ----------------------------

class ReActDAGAgent:
    def __init__(self, llm_provider: LLMProvider, orchestrator: str, max_iterations: int = 2):
        self.llm_provider = llm_provider
        self.orchestrator = orchestrator.lower().strip()
        self.max_iterations = max_iterations

        # Token accounting
        self.tokens = {
            "input_tokens_total": 0,
            "reasoning_tokens_total": 0,  # THINK completions
            "output_tokens_total": 0,     # ACT completions (code)
            "calls": []  # per-call breakdown
        }

    def _accumulate_tokens(self, usage: Dict[str, Any], *, phase: str) -> None:
        usage = usage or {}
        inp = int(usage.get("input_tokens", 0) or 0)
        out = int(usage.get("output_tokens", 0) or 0)

        self.tokens["input_tokens_total"] += inp
        if phase == "think":
            self.tokens["reasoning_tokens_total"] += out
        elif phase == "act":
            self.tokens["output_tokens_total"] += out

        self.tokens["calls"].append({
            "phase": phase,
            "input_tokens": inp,
            "output_tokens": out
        })

    def run(self, pipeline_description: str, output_dir: Path) -> Dict[str, Any]:
        history: Dict[str, Any] = {
            "last_code": None,
            "last_code_summary": None,
            "last_observation": None,
        }

        trace: List[Dict[str, Any]] = []
        final_code: str = ""
        final_obs: Dict[str, Any] = {"ok": False}

        for iteration in range(1, self.max_iterations + 1):
            # THINK
            think_sys, think_user = build_think_prompts(
                orchestrator=self.orchestrator,
                pipeline_description=pipeline_description,
                iteration=iteration,
                history=history
            )
            think_resp, think_usage = self.llm_provider.generate_completion(think_sys, think_user)
            self._accumulate_tokens(think_usage, phase="think")
            plan_text = (think_resp or "").strip()

            # Persist thought
            (output_dir / f"iteration_{iteration:02d}_think.txt").write_text(plan_text + "\n", encoding="utf-8")

            # ACT
            act_sys, act_user = build_act_prompts(
                orchestrator=self.orchestrator,
                pipeline_description=pipeline_description,
                iteration=iteration,
                plan_text=plan_text,
                history=history
            )
            act_resp, act_usage = self.llm_provider.generate_completion(act_sys, act_user)
            self._accumulate_tokens(act_usage, phase="act")

            code = extract_code(act_resp)
            if not code.endswith("\n"):
                code += "\n"

            # Persist code draft
            (output_dir / f"iteration_{iteration:02d}_code.py").write_text(code, encoding="utf-8")

            # OBSERVE (tools)
            obs = run_observation_tools(code, self.orchestrator)
            (output_dir / f"iteration_{iteration:02d}_observation.json").write_text(
                json.dumps(obs, indent=2),
                encoding="utf-8"
            )

            trace.append({
                "iteration": iteration,
                "think_chars": len(plan_text),
                "code_chars": len(code),
                "observation": obs
            })

            # Update history for next iteration
            history["last_code"] = code
            history["last_code_summary"] = summarize_code_for_prompt(code)
            history["last_observation"] = obs

            final_code = code
            final_obs = obs

            if obs.get("ok"):
                break

        # Save trace
        (output_dir / "react_trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")

        return {
            "final_code": final_code,
            "final_observation": final_obs,
            "trace": trace,
            "token_accounting": self.tokens
        }


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ReAct agent loop (non-reasoning model): generate orchestrator code from pipeline description"
    )
    parser.add_argument("--config", default="config_llm.json", help="Path to LLM config JSON")
    parser.add_argument("--input", required=True, help="Path to pipeline description file (.txt)")
    parser.add_argument("--orchestrator", choices=ORCHESTRATOR_CHOICES, default="airflow")
    parser.add_argument("--provider", choices=["deepinfra", "openai", "claude", "azureopenai", "ollama"],
                        help="Override provider specified in config")
    parser.add_argument("--model", help="Override model key for your LLMProvider")

    parser.add_argument("--output-root", default="generated_react",
                        help="Root directory for deterministic outputs")
    parser.add_argument("--output-dir", default=None,
                        help="Optional override output directory (disables deterministic layout)")
    parser.add_argument("--output-filename", default=None,
                        help="Optional override output filename (disables deterministic naming)")

    parser.add_argument("--max-iterations", type=int, default=2,
                        help="Number of THINK+ACT iterations (includes initial generation)")
    parser.add_argument("--skip-dependency-check", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    if not args.skip_dependency_check and not check_dependencies():
        sys.exit(1)

    config = load_config(args.config)
    if not config:
        logging.error(f"Failed to load config from: {args.config}")
        sys.exit(1)

    if args.provider:
        config.setdefault("model_settings", {})
        config["model_settings"]["active_provider"] = args.provider

    if not args.skip_dependency_check and not validate_api_keys(config):
        logging.error("API key validation failed.")
        sys.exit(1)

    input_path = Path(args.input)
    pipeline_description = input_path.read_text(encoding="utf-8").strip()
    if not pipeline_description:
        logging.error("Pipeline description is empty.")
        sys.exit(1)

    # Model init
    llm_provider = LLMProvider(config, args.model)
    model_info = llm_provider.get_model_info() or {}
    logging.info(f"Using provider={model_info.get('provider')} model={model_info.get('model_name')}")

    # Deterministic context
    ctx = parse_input_context(input_path)
    pipeline_id = ctx.get("pipeline_id") or "unknown_pipeline"
    variant_stem = ctx.get("variant_stem") or "unknown_variant"

    output_root = Path(args.output_root)
    output_dir_override = Path(args.output_dir) if args.output_dir else None

    if output_dir_override is not None:
        output_dir = output_dir_override
    else:
        output_dir = build_deterministic_output_dir(
            output_root=output_root,
            orchestrator=args.orchestrator,
            model_info=model_info,
            pipeline_id=pipeline_id,
            variant_stem=variant_stem
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = (
        args.output_filename
        if args.output_filename is not None
        else build_deterministic_filename(
            pipeline_id=pipeline_id,
            variant_stem=variant_stem,
            orchestrator=args.orchestrator,
            model_info=model_info
        )
    )

    agent = ReActDAGAgent(
        llm_provider=llm_provider,
        orchestrator=args.orchestrator,
        max_iterations=max(1, args.max_iterations)
    )
    result = agent.run(pipeline_description=pipeline_description, output_dir=output_dir)

    # Write final code file (deterministic)
    final_code_path = output_dir / output_filename
    final_code_path.write_text(result["final_code"], encoding="utf-8")

    # Build metadata
    meta = {
        "run_type": "react_agent_non_reasoning",
        "timestamp": datetime.now().isoformat(),
        "input_file": str(input_path),
        "pipeline_id": pipeline_id,
        "class_id": ctx.get("class_id"),
        "class_name": ctx.get("class_name"),
        "variant_stem": variant_stem,
        "orchestrator": args.orchestrator,
        "model_info": model_info,
        "docker_based_detected": detect_docker_based(pipeline_description),
        "prompt_sha256": sha256_text(pipeline_description),
        "prompt_chars": len(pipeline_description),
        "max_iterations": args.max_iterations,
        "iterations_ran": len(result["trace"]),
        "final_validation": result["final_observation"],
        "token_usage": result["token_accounting"],
        "output_dir": str(output_dir),
        "final_code_file": str(final_code_path),
        "react_trace_file": str(output_dir / "react_trace.json")
    }

    meta_path = output_dir / "generation_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Print summary
    tokens = meta["token_usage"]
    total_in = tokens.get("input_tokens_total", 0)
    total_reason = tokens.get("reasoning_tokens_total", 0)
    total_out = tokens.get("output_tokens_total", 0)

    print("\nGeneration completed (ReAct agent).")
    print(f"Pipeline:        {pipeline_id}")
    print(f"Variant:         {variant_stem}")
    print(f"Orchestrator:    {args.orchestrator}")
    print(f"Iterations ran:  {meta['iterations_ran']}/{args.max_iterations}")
    print(f"Tokens (input):  {total_in}")
    print(f"Tokens (reason): {total_reason}")
    print(f"Tokens (output): {total_out}")
    print(f"Final code:      {final_code_path}")
    print(f"Metadata:        {meta_path}")


if __name__ == "__main__":
    main()