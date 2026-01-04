#!/usr/bin/env python3
"""
Direct Prompting (Non-Reasoning Models) for Workflow Code Generation
===================================================================
Generates Airflow / Prefect / Dagster code from a pipeline description text file.

Key Improvements:
- Detect docker-based pipeline PER INPUT DESCRIPTION (not global config)
- Deterministic output directory + filename derived from input file path:
    pipeline_variants/<pipeline_id>/C5_NO_....txt
  => output_root/<orchestrator>/<provider>/<model>/<pipeline_id>/C5_NO_.../<pipeline_id>__C5_NO_...__<orchestrator>__<model>.py
- Metadata enriched: pipeline_id, class_id, class_name, prompt_sha256, prompt length, input path, model info
- Adds explicit instruction: honor "Mechanism" field in the description

Usage:
  python scripts/direct_prompting_non_reasoning.py \
    --config config_llm.json \
    --input pipeline_variants/<pipeline_id>/C0_FULL_STRUCTURED.txt \
    --orchestrator airflow \
    --output-root generated_non_reasoning \
    --model <optional_model_key>

Notes:
- If you pass --output-dir explicitly, we will use it (override deterministic layout).
- Otherwise, we compute output_dir deterministically under --output-root.
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
from typing import Dict, Tuple, Optional, Any

# Add parent directory to Python path for importing utils package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Optional dependency checks (for convenience when running standalone)
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

# Try to import your project's utilities
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
# Determinism helpers
# ----------------------------

def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def sanitize_for_path(s: str) -> str:
    if s is None:
        return "unknown"
    s = str(s).strip()
    # Replace path-hostile chars with underscore
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s or "unknown"


def parse_input_context(input_path: Path) -> Dict[str, Optional[str]]:
    """
    Expecting input like: pipeline_variants/<pipeline_id>/C5_NO_....txt
    """
    pipeline_id = input_path.parent.name if input_path.parent else None
    stem = input_path.stem  # e.g., C5_NO_EXTERNAL_SYSTEM_DETAILS
    class_id = None
    class_name = None
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


def detect_docker_based(pipeline_description: str) -> bool:
    """
    Update A: per-description detection of docker/container usage.
    """
    text = (pipeline_description or "").lower()
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
    orchestrator = sanitize_for_path(orchestrator)
    pipeline_id = sanitize_for_path(pipeline_id)
    variant_stem = sanitize_for_path(variant_stem)
    return f"{pipeline_id}__{variant_stem}__{orchestrator}__{model_name}.py"


# ----------------------------
# Prompt building
# ----------------------------

def orchestrator_guidance(orchestrator: str, docker_based: bool) -> Tuple[str, str]:
    """
    Returns: (system_addon, user_addon)
    """
    orchestrator = orchestrator.lower().strip()

    # Docker hints (orchestrator-specific)
    docker_hint_common = ""
    if docker_based:
        if orchestrator == "airflow":
            docker_hint_common = (
                "- If a step has Mechanism=container_run, use DockerOperator with image/command/env/mounts/network when available.\n"
            )
        elif orchestrator == "prefect":
            docker_hint_common = (
                "- If a step has Mechanism=container_run, use prefect-docker (DockerContainer) or a safe subprocess-based container run.\n"
            )
        elif orchestrator == "dagster":
            docker_hint_common = (
                "- If a step has Mechanism=container_run, implement container execution via a resource or subprocess invocation pattern.\n"
            )

    if orchestrator == "airflow":
        system_addon = (
            "You are an expert Apache Airflow developer. Generate complete, executable Apache Airflow DAG Python code.\n"
            "Use Airflow 2.x imports, DAG context manager, explicit dependencies with >>, and clean task naming.\n"
        )
        user_addon = (
            "CONSTRAINTS FOR AIRFLOW 2.x:\n"
            "- Use Airflow 2.x style imports.\n"
            "- Define a DAG with schedule_interval (or None), catchup, default_args.\n"
            "- Implement branching with BranchPythonOperator if needed.\n"
            "- Implement waiting gates with Sensors (FileSensor/SqlSensor/ExternalTaskSensor) if applicable.\n"
            f"{docker_hint_common}"
        )
    elif orchestrator == "prefect":
        system_addon = (
            "You are an expert Prefect 2.x developer. Generate complete, executable Prefect 2.x Python code.\n"
            "Use @task for steps and @flow for orchestration.\n"
        )
        user_addon = (
            "CONSTRAINTS FOR PREFECT 2.x:\n"
            "- Use: from prefect import flow, task.\n"
            "- For parallelism, use .submit() and then gather results.\n"
            "- Include if __name__ == '__main__': flow() for local execution.\n"
            "- If scheduling is mentioned, include a brief comment about deployment scheduling.\n"
            f"{docker_hint_common}"
        )
    elif orchestrator == "dagster":
        system_addon = (
            "You are an expert Dagster developer. Generate complete, executable Dagster Python code.\n"
            "Use @op for steps and @job (or graph) to define dependencies.\n"
        )
        user_addon = (
            "CONSTRAINTS FOR DAGSTER:\n"
            "- Use @op for each step and @job to connect them.\n"
            "- Provide minimal resource/config stubs as needed.\n"
            "- Include if __name__ == '__main__': job.execute_in_process() for local execution.\n"
            f"{docker_hint_common}"
        )
    else:
        system_addon = "You are an expert workflow developer. Generate clean Python code.\n"
        user_addon = "Use idiomatic constructs for the requested orchestrator.\n"

    return system_addon, user_addon


MECHANISM_MAPPING_GUIDE = """MECHANISM MAPPING (MUST FOLLOW):
- Mechanism: container_run => run a container image (orchestrator-appropriate container runner)
- Mechanism: http_request  => perform an HTTP request (requests library or orchestrator native)
- Mechanism: sql_query      => execute SQL against a database connection/resource
- Mechanism: shell_command  => run a shell command (subprocess or orchestrator op)
- Mechanism: wait_poll      => implement a wait/sensor/poll loop until condition is met
- Mechanism: notification   => send notification (email/log) as described
- Mechanism: external_workflow_trigger => trigger another workflow/job if described
"""


class DirectDAGGenerator:
    """
    Generate orchestrator code (Airflow | Prefect | Dagster) from a pipeline description.
    """

    def __init__(self, config: Dict, llm_provider: LLMProvider, orchestrator: str = "airflow"):
        self.config = config or {}
        self.llm_provider = llm_provider
        self.orchestrator = (orchestrator or "airflow").lower().strip()
        self.token_usage_history = {"initial": {"input_tokens": 0, "output_tokens": 0}}

    def _build_prompts(self, pipeline_description: str) -> Tuple[str, str]:
        docker_based = detect_docker_based(pipeline_description)

        sys_addon, user_addon = orchestrator_guidance(self.orchestrator, docker_based)

        system_prompt = (
            f"{sys_addon}"
            "Return ONLY valid Python code for a single module. Do not include markdown fences or explanations.\n"
        )

        orchestrator_title = {
            "airflow": "Apache Airflow DAG",
            "prefect": "Prefect 2 Flow",
            "dagster": "Dagster Job",
        }.get(self.orchestrator, "Workflow")

        user_prompt = (
            f"Convert the following pipeline description into a {orchestrator_title} Python module.\n\n"
            f"{user_addon}\n"
            f"{MECHANISM_MAPPING_GUIDE}\n"
            "GENERAL CONSTRAINTS:\n"
            "- The code must be executable (assuming required packages are installed) and PEP 8 compliant.\n"
            "- Use clear function/task/op names derived from the pipeline steps.\n"
            "- Steps must have explicit dependencies reflecting order, branching, parallelism, and gates.\n"
            "- Avoid placeholders like <TODO> in code; if something is unknown, use minimal safe defaults or add a comment.\n"
            "- Output only the complete Python code.\n\n"
            "PIPELINE DESCRIPTION:\n"
            f"{pipeline_description}\n"
        )

        return system_prompt, user_prompt

    def generate_initial_code(self, pipeline_description: str) -> Tuple[str, Dict]:
        logging.info(f"Generating {self.orchestrator} code from pipeline description")
        system_prompt, user_prompt = self._build_prompts(pipeline_description)

        raw_code, token_usage = self.llm_provider.generate_completion(system_prompt, user_prompt)
        clean_code = self.extract_code(raw_code)

        self.token_usage_history["initial"] = token_usage or {"input_tokens": 0, "output_tokens": 0}
        return clean_code, (token_usage or {})

    @staticmethod
    def extract_code(content: str) -> str:
        """
        Extract clean Python code from LLM response that may contain fences.
        """
        if not content:
            return ""
        text = content.strip()

        # Common code fences
        if "```python" in text:
            return text.split("```python", 1)[1].split("```", 1)[0].strip()
        if "```py" in text:
            return text.split("```py", 1)[1].split("```", 1)[0].strip()
        if "```" in text:
            parts = text.split("```", 1)
            if len(parts) > 1:
                return parts[1].split("```", 1)[0].strip()

        return text

    def save_outputs(
        self,
        code: str,
        output_dir: Path,
        output_filename: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)

        code_path = output_dir / output_filename
        code_path.write_text(code, encoding="utf-8")

        meta_path = output_dir / "generation_metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return {"code_file": str(code_path), "metadata_file": str(meta_path)}

    def generate(
        self,
        pipeline_description: str,
        *,
        input_path: Path,
        output_root: Path,
        output_dir_override: Optional[Path] = None,
        output_filename_override: Optional[str] = None,
        model_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ctx = parse_input_context(input_path)
        pipeline_id = ctx.get("pipeline_id") or "unknown_pipeline"
        class_id = ctx.get("class_id")
        class_name = ctx.get("class_name")
        variant_stem = ctx.get("variant_stem") or "unknown_variant"

        model_info = model_info or {}
        resolved_output_dir = (
            output_dir_override
            if output_dir_override is not None
            else build_deterministic_output_dir(
                output_root=output_root,
                orchestrator=self.orchestrator,
                model_info=model_info,
                pipeline_id=pipeline_id,
                variant_stem=variant_stem
            )
        )

        resolved_filename = (
            output_filename_override
            if output_filename_override is not None
            else build_deterministic_filename(
                pipeline_id=pipeline_id,
                variant_stem=variant_stem,
                orchestrator=self.orchestrator,
                model_info=model_info
            )
        )

        code, token_usage = self.generate_initial_code(pipeline_description)

        metadata: Dict[str, Any] = {
            "run_type": "non_reasoning",
            "timestamp": datetime.now().isoformat(),
            "input_file": str(input_path),
            "pipeline_id": pipeline_id,
            "class_id": class_id,
            "class_name": class_name,
            "variant_stem": variant_stem,
            "orchestrator": self.orchestrator,
            "model_info": model_info,
            "token_usage": token_usage,
            "prompt_sha256": sha256_text(pipeline_description),
            "prompt_chars": len(pipeline_description or ""),
            "docker_based_detected": detect_docker_based(pipeline_description),
            "resolved_output_dir": str(resolved_output_dir),
            "resolved_output_filename": resolved_filename,
        }

        files = self.save_outputs(
            code=code,
            output_dir=resolved_output_dir,
            output_filename=resolved_filename,
            metadata=metadata
        )

        metadata.update(files)
        return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Non-reasoning direct prompting: generate orchestrator code from a pipeline description"
    )
    parser.add_argument("--config", default="config_llm.json", help="Path to LLM configuration JSON")
    parser.add_argument("--input", required=True, help="Path to pipeline description file (.txt)")
    parser.add_argument("--orchestrator", choices=ORCHESTRATOR_CHOICES, default="airflow")
    parser.add_argument("--provider", choices=["deepinfra", "openai", "claude", "azureopenai", "ollama"],
                        help="Override provider specified in config")
    parser.add_argument("--model", help="Override model key (used by your LLMProvider)")

    # Deterministic outputs
    parser.add_argument("--output-root", default="generated_non_reasoning",
                        help="Root directory for deterministic outputs")
    parser.add_argument("--output-dir", default=None,
                        help="Optional override output directory (disables deterministic layout for this run)")
    parser.add_argument("--output-filename", default=None,
                        help="Optional override output filename (disables deterministic naming for this run)")

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
    try:
        pipeline_description = input_path.read_text(encoding="utf-8").strip()
    except Exception as e:
        logging.error(f"Failed to read input file: {input_path} error={e}")
        sys.exit(1)

    if not pipeline_description:
        logging.error("Pipeline description is empty.")
        sys.exit(1)

    output_root = Path(args.output_root)
    output_dir_override = Path(args.output_dir) if args.output_dir else None

    # Init provider
    llm_provider = LLMProvider(config, args.model)
    model_info = llm_provider.get_model_info() or {}
    logging.info(f"Using provider={model_info.get('provider')} model={model_info.get('model_name')}")

    generator = DirectDAGGenerator(config=config, llm_provider=llm_provider, orchestrator=args.orchestrator)
    results = generator.generate(
        pipeline_description=pipeline_description,
        input_path=input_path,
        output_root=output_root,
        output_dir_override=output_dir_override,
        output_filename_override=args.output_filename,
        model_info=model_info
    )

    total_tokens = (results.get("token_usage", {}).get("input_tokens", 0) +
                    results.get("token_usage", {}).get("output_tokens", 0))

    print("\nGeneration completed (Non-reasoning).")
    print(f"Pipeline:      {results.get('pipeline_id')}")
    print(f"Variant:       {results.get('variant_stem')}")
    print(f"Orchestrator:  {results.get('orchestrator')}")
    print(f"Tokens used:   {total_tokens}")
    print(f"Code file:     {results.get('code_file')}")
    print(f"Metadata file: {results.get('metadata_file')}")


if __name__ == "__main__":
    main()