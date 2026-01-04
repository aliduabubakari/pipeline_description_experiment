#!/usr/bin/env python3
"""
Direct Prompting with Reasoning Models for Workflow Code Generation
==================================================================

Goal:
- Generate Airflow / Prefect / Dagster code using a reasoning-capable model.
- Enforce strict separation between reasoning and executable code:
    1) Reasoning section (saved to __reasoning.txt)
    2) Python code section (saved to .py, used for evaluation)

Output Format (MANDATORY):
<think>
... reasoning ...
</think>
```python
... complete python module ...
```

If the model deviates, we use robust heuristic extraction:
- Strip <think>/<reasoning>/<analysis> blocks
- Extract markdown code blocks if present
- Extract code after "Here is the code:"-style transitions
- Extract code starting from first Python-looking line
- Fallback: return remaining content (lowest confidence)

Also includes retry/backoff for transient API failures (429/model busy).

Usage:
  python scripts/direct_prompting_reasoning.py \
    --config config_reasoning_llm.json \
    --input pipeline_variants/<pipeline_id>/C0_FULL_STRUCTURED.txt \
    --orchestrator airflow \
    --output-root generated_reasoning \
    --model Qwen3-235B-A22B-Thinking-2507
"""

import os
import sys
import re
import json
import time
import random
import hashlib
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass

from openai import OpenAI
from openai import RateLimitError, APIError, APITimeoutError


# ----------------------------
# Determinism helpers
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
    pipeline_id = input_path.parent.name if input_path.parent else None
    stem = input_path.stem
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

def detect_docker_based(pipeline_description: str) -> bool:
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
# Extraction
# ----------------------------

@dataclass
class ExtractionResult:
    code: str
    reasoning: str
    confidence: str  # high|medium|low
    method: str
    warnings: List[str]


class CodeExtractor:
    """
    Robust heuristic-based extractor that separates reasoning and code.
    """

    # Reasoning block patterns
    REASONING_PATTERNS = [
        (r"<think>(.*?)</think>", "think_tags"),
        (r"<thinking>(.*?)</thinking>", "thinking_tags"),
        (r"<reasoning>(.*?)</reasoning>", "reasoning_tags"),
        (r"<analysis>(.*?)</analysis>", "analysis_tags"),
        (r"\[REASONING\](.*?)\[/REASONING\]", "reasoning_brackets"),
    ]

    # Code fences
    CODE_BLOCK_PATTERNS = [
        (r"```python\s*\n(.*?)\n```", "python_markdown"),
        (r"```py\s*\n(.*?)\n```", "py_markdown"),
        (r"```\s*\n(.*?)\n```", "generic_markdown"),
    ]

    # Transition phrases that often precede code (from your old implementation)
    CODE_TRANSITIONS = [
        r"here'?s?\s+the\s+(?:complete|final|full)?\s*(?:code|implementation|solution|dag|workflow)",
        r"final\s+(?:code|implementation|solution|dag|workflow)",
        r"complete\s+(?:code|implementation|solution|dag|workflow)",
        r"below\s+is\s+the\s+(?:complete|final|full)?\s*(?:code|implementation|solution|dag|workflow)",
        r"(?:python|airflow|prefect|dagster)\s+(?:code|implementation)",
        r"code\s*[:\-]\s*$",
    ]

    PYTHON_START_PATTERNS = [
        r'^\s*"""',
        r"^\s*'''",
        r"^\s*#.*coding",
        r"^\s*from\s+\w+\s+import",
        r"^\s*import\s+\w+",
        r"^\s*@\w+",
        r"^\s*def\s+\w+",
        r"^\s*class\s+\w+",
        r"^\s*with\s+DAG",
        r"^\s*@flow",
        r"^\s*@job",
        r"^\s*@op",
    ]

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def extract(self, raw_response: str) -> ExtractionResult:
        warnings: List[str] = []

        reasoning_text, primary_content = self._extract_reasoning_blocks(raw_response)

        # 1) Try fenced code blocks
        code, method = self._extract_from_code_blocks(primary_content)
        if code:
            return ExtractionResult(code=code, reasoning=reasoning_text, confidence="high", method=method, warnings=warnings)

        # 2) Try transition phrase extraction ("here is the code:")
        code, method = self._extract_after_transition(primary_content)
        if code:
            conf = "high" if self._validate_python_code(code) else "medium"
            return ExtractionResult(code=code, reasoning=reasoning_text, confidence=conf, method=method, warnings=warnings)

        # 3) Try python pattern match
        code, method = self._extract_by_python_patterns(primary_content)
        if code:
            conf = "high" if self._validate_python_code(code) else "medium"
            return ExtractionResult(code=code, reasoning=reasoning_text, confidence=conf, method=method, warnings=warnings)

        # 4) Fallback
        warnings.append("Fallback extraction used; response may include non-code content.")
        return ExtractionResult(
            code=primary_content.strip(),
            reasoning=reasoning_text,
            confidence="low",
            method="fallback_full_content",
            warnings=warnings
        )

    def _extract_reasoning_blocks(self, content: str) -> Tuple[str, str]:
        for pattern, _name in self.REASONING_PATTERNS:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            if matches:
                reasoning = "\n\n".join(matches).strip()
                remaining = re.sub(pattern, "", content, flags=re.DOTALL | re.IGNORECASE).strip()
                return reasoning, remaining
        return "", content

    def _extract_from_code_blocks(self, content: str) -> Tuple[Optional[str], Optional[str]]:
        for pattern, method in self.CODE_BLOCK_PATTERNS:
            matches = re.findall(pattern, content, re.DOTALL)
            if matches:
                code = max(matches, key=len).strip()
                return code, method
        return None, None

    def _extract_after_transition(self, content: str) -> Tuple[Optional[str], Optional[str]]:
        for transition_pattern in self.CODE_TRANSITIONS:
            pattern = rf"{transition_pattern}[:\s]*\n+(.*)"
            m = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if m:
                candidate = m.group(1).strip()
                # remove any accidental fences
                candidate = re.sub(r"^```(?:python|py)?\s*\n?", "", candidate.strip())
                candidate = re.sub(r"\n?```\s*$", "", candidate.strip())
                return candidate.strip(), "transition_phrase"
        return None, None

    def _extract_by_python_patterns(self, content: str) -> Tuple[Optional[str], Optional[str]]:
        lines = content.split("\n")
        start_idx = -1
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            if any(re.match(pat, line) for pat in self.PYTHON_START_PATTERNS):
                start_idx = i
                break
        if start_idx >= 0:
            return "\n".join(lines[start_idx:]).strip(), "python_pattern_match"
        return None, None

    def _validate_python_code(self, code: str) -> bool:
        try:
            compile(code, "<string>", "exec")
            return True
        except SyntaxError:
            indicators = ["import ", "from ", "def ", "class ", "@flow", "@job", "@op", "DAG("]
            return any(ind in code for ind in indicators)

    def clean_code(self, code: str) -> str:
        """
        Clean extracted code so the .py file contains code only.
        Also strips any leaked reasoning tags if they slipped through.
        """
        if not code:
            return ""

        txt = code.strip()

        # Strip fences if present
        txt = re.sub(r"^```(?:python|py)?\s*\n?", "", txt)
        txt = re.sub(r"\n?```\s*$", "", txt)

        # Strip any leaked reasoning blocks
        txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL | re.IGNORECASE)
        txt = re.sub(r"<reasoning>.*?</reasoning>", "", txt, flags=re.DOTALL | re.IGNORECASE)
        txt = re.sub(r"<analysis>.*?</analysis>", "", txt, flags=re.DOTALL | re.IGNORECASE)

        # Normalize excessive blank lines
        txt = re.sub(r"\n{3,}", "\n\n", txt).strip()

        if not txt.endswith("\n"):
            txt += "\n"
        return txt


# ----------------------------
# Model client (OpenAI-compatible) with retry/backoff
# ----------------------------

class ReasoningModelClient:
    def __init__(self, config: Dict, model_key: Optional[str]):
        self.logger = logging.getLogger(self.__class__.__name__)

        provider_config = config["model_settings"]["deepinfra"]
        self.model_key = model_key or provider_config.get("active_model")

        if self.model_key not in provider_config["models"]:
            raise ValueError(f"Model key '{self.model_key}' not found in config")

        model_cfg = provider_config["models"][self.model_key]
        self.model_name = model_cfg["model_name"]
        self.api_key = model_cfg.get("api_key")
        self.base_url = model_cfg.get("base_url")
        self.max_tokens = model_cfg.get("max_tokens", 8000)
        self.temperature = model_cfg.get("temperature", 0)

        if not self.api_key:
            raise ValueError(f"API key missing for model: {self.model_key}")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate(self, system_prompt: str, user_prompt: str) -> Tuple[str, Dict]:
        """
        Retries transient failures like 429 Model busy.
        """
        max_attempts = 8
        base_sleep = 2.0
        last_exc: Optional[Exception] = None

        for attempt in range(1, max_attempts + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )

                content = resp.choices[0].message.content or ""
                usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                if getattr(resp, "usage", None):
                    usage["input_tokens"] = getattr(resp.usage, "prompt_tokens", 0) or 0
                    usage["output_tokens"] = getattr(resp.usage, "completion_tokens", 0) or 0
                    usage["total_tokens"] = getattr(resp.usage, "total_tokens", 0) or 0

                return content, usage

            except (RateLimitError, APITimeoutError, APIError) as e:
                last_exc = e
                sleep = base_sleep * (2 ** (attempt - 1))
                sleep = min(sleep, 90.0)
                sleep = sleep * (0.7 + 0.6 * random.random())
                self.logger.warning(f"Retryable error attempt {attempt}/{max_attempts}: {e}. Sleeping {sleep:.1f}s")
                time.sleep(sleep)

        raise last_exc  # type: ignore

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "deepinfra",
            "model_key": self.model_key,
            "model_name": self.model_name,
            "base_url": self.base_url,
        }


# ----------------------------
# Prompt building (strict separation)
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


class ReasoningDAGGenerator:
    def __init__(self, model_client: ReasoningModelClient, orchestrator: str = "airflow"):
        self.model_client = model_client
        self.extractor = CodeExtractor()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.orchestrator = orchestrator.lower().strip()

    def _build_prompts(self, pipeline_description: str) -> Tuple[str, str]:
        orch = self.orchestrator
        docker_based = detect_docker_based(pipeline_description)

        docker_hint = ""
        if docker_based:
            if orch == "airflow":
                docker_hint = "- If Mechanism=container_run, use DockerOperator.\n"
            elif orch == "prefect":
                docker_hint = "- If Mechanism=container_run, use prefect-docker DockerContainer or a safe subprocess runner.\n"
            elif orch == "dagster":
                docker_hint = "- If Mechanism=container_run, implement container execution via a resource or subprocess.\n"

        if orch == "airflow":
            orch_sys = "You are an expert Apache Airflow 2.x developer.\n"
            orch_user = (
                "AIRFLOW CONSTRAINTS:\n"
                "- Use Airflow 2.x imports and DAG context manager.\n"
                "- Use explicit dependencies with >>.\n"
                "- Use Sensors for wait_poll gates; BranchPythonOperator for branching if needed.\n"
            )
        elif orch == "prefect":
            orch_sys = "You are an expert Prefect 2.x developer.\n"
            orch_user = (
                "PREFECT CONSTRAINTS:\n"
                "- Use @task and @flow.\n"
                "- Use .submit() for parallelism.\n"
                "- Include if __name__ == '__main__': flow() for local execution.\n"
            )
        else:
            orch_sys = "You are an expert Dagster developer.\n"
            orch_user = (
                "DAGSTER CONSTRAINTS:\n"
                "- Use @op and @job to wire dependencies.\n"
                "- Include if __name__ == '__main__': job.execute_in_process().\n"
            )

        # Strict output format: reasoning in <think>, code in fenced python block, nothing else.
        system_prompt = (
            f"{orch_sys}"
            "You MUST follow this exact output format and include NOTHING else:\n"
            "<think>\n"
            "...your reasoning and mapping decisions...\n"
            "</think>\n"
            "```python\n"
            "# complete executable python module\n"
            "```\n"
        )

        user_prompt = (
            f"Convert the pipeline description below into a complete {orch.upper()} Python module.\n\n"
            f"{orch_user}"
            f"{docker_hint}"
            f"{MECHANISM_MAPPING_GUIDE}\n"
            "GENERAL CONSTRAINTS:\n"
            "- Code must be executable (assuming dependencies are installed) and PEP 8 compliant.\n"
            "- Implement explicit dependencies reflecting the Control Flow.\n"
            "- Avoid <TODO> placeholders; if unknown, use minimal safe defaults or comments.\n\n"
            "OUTPUT FORMAT (MANDATORY):\n"
            "<think>...</think>\n"
            "```python\n"
            "...code only...\n"
            "```\n\n"
            "PIPELINE DESCRIPTION:\n"
            f"{pipeline_description}\n"
        )

        return system_prompt, user_prompt

    def generate(
        self,
        pipeline_description: str,
        *,
        input_path: Path,
        output_root: Path,
        output_dir_override: Optional[Path] = None,
        output_filename_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        ctx = parse_input_context(input_path)
        pipeline_id = ctx.get("pipeline_id") or "unknown_pipeline"
        class_id = ctx.get("class_id")
        class_name = ctx.get("class_name")
        variant_stem = ctx.get("variant_stem") or "unknown_variant"

        model_info = self.model_client.get_model_info()

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
        resolved_output_dir.mkdir(parents=True, exist_ok=True)

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

        system_prompt, user_prompt = self._build_prompts(pipeline_description)

        raw_response, token_usage = self.model_client.generate(system_prompt, user_prompt)

        extraction = self.extractor.extract(raw_response)
        clean_code = self.extractor.clean_code(extraction.code)

        # Save code (used for evaluation)
        code_path = resolved_output_dir / resolved_filename
        code_path.write_text(clean_code, encoding="utf-8")

        # Save reasoning (for future reference)
        reasoning_path = None
        if extraction.reasoning.strip():
            reasoning_path = resolved_output_dir / resolved_filename.replace(".py", "__reasoning.txt")
            reasoning_path.write_text(extraction.reasoning.strip() + "\n", encoding="utf-8")

        metadata: Dict[str, Any] = {
            "run_type": "reasoning",
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
            "extraction_method": extraction.method,
            "extraction_confidence": extraction.confidence,
            "extraction_warnings": extraction.warnings,
            "resolved_output_dir": str(resolved_output_dir),
            "resolved_output_filename": resolved_filename,
            "code_file": str(code_path),
            "reasoning_file": str(reasoning_path) if reasoning_path else None,
        }

        meta_path = resolved_output_dir / "generation_metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        metadata["metadata_file"] = str(meta_path)
        return metadata


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Reasoning-model direct prompting: generate orchestrator code + separate reasoning"
    )
    parser.add_argument("--config", default="config_reasoning_llm.json")
    parser.add_argument("--input", required=True, help="Path to pipeline description .txt")
    parser.add_argument("--orchestrator", choices=["airflow", "prefect", "dagster"], default="airflow")
    parser.add_argument("--model", help="Model key to use (e.g., DeepSeek_R1, QwQ_32B)")

    parser.add_argument("--output-root", default="generated_reasoning",
                        help="Root directory for deterministic outputs")
    parser.add_argument("--output-dir", default=None,
                        help="Optional override output directory")
    parser.add_argument("--output-filename", default=None,
                        help="Optional override output filename")

    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    config = load_config(args.config)

    input_path = Path(args.input)
    pipeline_description = input_path.read_text(encoding="utf-8").strip()
    if not pipeline_description:
        raise SystemExit("Pipeline description is empty.")

    model_client = ReasoningModelClient(config, args.model)
    model_info = model_client.get_model_info()
    logging.info(f"Using reasoning model {model_info.get('model_name')} ({model_info.get('model_key')})")

    generator = ReasoningDAGGenerator(model_client=model_client, orchestrator=args.orchestrator)

    output_root = Path(args.output_root)
    output_dir_override = Path(args.output_dir) if args.output_dir else None

    results = generator.generate(
        pipeline_description=pipeline_description,
        input_path=input_path,
        output_root=output_root,
        output_dir_override=output_dir_override,
        output_filename_override=args.output_filename,
    )

    print("\nGeneration completed (Reasoning).")
    print(f"Pipeline:      {results.get('pipeline_id')}")
    print(f"Variant:       {results.get('variant_stem')}")
    print(f"Orchestrator:  {results.get('orchestrator')}")
    print(f"Model:         {results.get('model_info', {}).get('model_name')}")
    print(f"Code file:     {results.get('code_file')}")
    print(f"Reasoning file:{results.get('reasoning_file')}")
    print(f"Metadata file: {results.get('metadata_file')}")


if __name__ == "__main__":
    main()