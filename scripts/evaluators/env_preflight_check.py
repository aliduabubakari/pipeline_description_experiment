#!/usr/bin/env python3
"""
Environment Preflight Check + Auto-Setup (Prompt2Workflow)
==========================================================

Purpose
-------
1) Verify that an experiment environment is ready (Airflow 2.8.4 pinned, providers,
   evaluation deps, Prefect/Dagster, quality tools, CodeBERT BERTScore).
2) Optionally create + install everything into a venv built with Python 3.10.13.

Key requirement
---------------
Airflow 2.8.4 is compatible with Python 3.10. This script enforces Python 3.10.13
for the venv used for the experiments.

Usage
-----
# Just check an existing venv:
python scripts/evaluators/env_preflight_check.py --venv-path .venv-airflow284 --strict

# Create venv (if missing) + install deps + verify:
python scripts/evaluators/env_preflight_check.py --setup --venv-path .venv-airflow284 --strict

Options
-------
--with-mysql-provider:
  Attempts to install apache-airflow-providers-mysql (often fails on macOS unless
  system libs are installed). Default: skip.
"""

from __future__ import annotations

import sys
import os
import json
import argparse
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    installed_version: Optional[str] = None
    required_version: Optional[str] = None
    category: str = "general"
    install_hint: Optional[str] = None


# ----------------------------
# Shell helpers
# ----------------------------

def sh(cmd: List[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, capture_output=True, text=True)

def exists_exe(p: str) -> bool:
    try:
        return Path(p).exists()
    except Exception:
        return False


# ----------------------------
# Venv helpers
# ----------------------------

def venv_python(venv_path: Path) -> Path:
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"

def venv_pip_cmd(venv_path: Path) -> List[str]:
    return [str(venv_python(venv_path)), "-m", "pip"]

def pip_install(venv_path: Path, pkgs: List[str], *, constraint: Optional[str] = None) -> None:
    cmd = venv_pip_cmd(venv_path) + ["install"] + pkgs
    if constraint:
        cmd += ["--constraint", constraint]
    sh(cmd, check=True)

def pip_install_best_effort(venv_path: Path, pkgs: List[str], *, constraint: Optional[str] = None) -> bool:
    try:
        pip_install(venv_path, pkgs, constraint=constraint)
        return True
    except subprocess.CalledProcessError as e:
        print(f"WARNING: pip install failed: {' '.join(pkgs)}")
        if e.stderr:
            print(e.stderr[:600])
        return False

def pip_upgrade_base(venv_path: Path) -> None:
    pip_install_best_effort(venv_path, ["-U", "pip", "setuptools", "wheel"])


# ----------------------------
# Python 3.10.13 discovery (pyenv-aware)
# ----------------------------

def python_version_string(python_cmd: str) -> Optional[str]:
    try:
        r = sh([python_cmd, "--version"], check=False)
        out = (r.stdout or r.stderr or "").strip()
        if r.returncode == 0 and out.startswith("Python "):
            return out.replace("Python ", "")
        return None
    except Exception:
        return None

def is_python_31013(python_cmd: str) -> bool:
    v = python_version_string(python_cmd)
    return v == "3.10.13"

def try_pyenv_prefix(version: str = "3.10.13") -> Optional[str]:
    """
    Try to locate Python binary via:
      pyenv prefix 3.10.13 -> <prefix>
      then use <prefix>/bin/python
    This avoids needing `pyenv which python3.10` to work.
    """
    try:
        r = sh(["pyenv", "prefix", version], check=False)
        if r.returncode != 0:
            return None
        prefix = (r.stdout or "").strip()
        if not prefix:
            return None
        cand = str(Path(prefix) / "bin" / "python")
        if exists_exe(cand) and is_python_31013(cand):
            return cand
        return None
    except Exception:
        return None

def find_python_31013(preferred: Optional[str] = None) -> Optional[str]:
    """
    Find an executable that runs exactly Python 3.10.13.

    Strategy:
      1) If `preferred` is provided, try it first.
      2) Try pyenv prefix 3.10.13 (best for your machine).
      3) Try common command names (python3.10, python3, python) and common shim paths.
      4) As a last resort, try pyenv versions list and derive prefix (slow).
    """
    candidates: List[str] = []
    if preferred:
        candidates.append(preferred)

    # best option for your setup:
    pyenv_bin = try_pyenv_prefix("3.10.13")
    if pyenv_bin:
        return pyenv_bin

    # common commands (may fail if pyenv not initialized)
    candidates += ["python3.10", "python3", "python"]

    # pyenv shims
    home = Path.home()
    candidates += [
        str(home / ".pyenv" / "shims" / "python3.10"),
        str(home / ".pyenv" / "shims" / "python"),
    ]

    # de-dup
    seen = set()
    ordered: List[str] = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            ordered.append(c)

    for cmd in ordered:
        if is_python_31013(cmd):
            return cmd

    # last resort: enumerate pyenv prefixes if possible
    try:
        r = sh(["pyenv", "versions", "--bare"], check=False)
        if r.returncode == 0:
            versions = [ln.strip() for ln in (r.stdout or "").splitlines() if ln.strip()]
            # prioritize entries containing 3.10.13
            for v in versions:
                if v == "3.10.13":
                    py = try_pyenv_prefix("3.10.13")
                    if py:
                        return py
    except Exception:
        pass

    return None


def ensure_python_31013_or_fail(preferred: Optional[str]) -> str:
    resolved = find_python_31013(preferred)
    if not resolved:
        raise SystemExit(
            "ERROR: Could not find Python 3.10.13.\n"
            "Fix options:\n"
            "  - Ensure pyenv is installed and 3.10.13 is installed.\n"
            "  - Run: pyenv install 3.10.13\n"
            "  - Or pass an explicit python path via --python-cmd.\n"
            "Current hints:\n"
            f"  - pyenv prefix 3.10.13 must succeed, or python3.10 must be on PATH.\n"
        )
    return resolved


# ----------------------------
# Create venv
# ----------------------------

def create_venv(venv_path: Path, python_cmd: str) -> None:
    if venv_path.exists():
        return
    print(f"Creating virtual environment at: {venv_path}")
    sh([python_cmd, "-m", "venv", str(venv_path)], check=True)

    # verify venv python version
    vp = str(venv_python(venv_path))
    ver = python_version_string(vp)
    if ver != "3.10.13":
        raise SystemExit(
            f"ERROR: venv python version is {ver}, expected 3.10.13.\n"
            f"venv python: {vp}\n"
            "Recreate the venv with Python 3.10.13."
        )


# ----------------------------
# Setup environment (install deps)
# ----------------------------

def setup_env(venv_path: Path, python_cmd: str, airflow_version: str, with_mysql_provider: bool) -> None:
    """
    Creates venv if missing and installs all required packages.
    """
    create_venv(venv_path, python_cmd)

    constraint = f"https://raw.githubusercontent.com/apache/airflow/constraints-{airflow_version}/constraints-3.10.txt"
    print(f"Using Airflow constraints: {constraint}")

    pip_upgrade_base(venv_path)

    # Airflow core
    pip_install(venv_path, [f"apache-airflow=={airflow_version}"], constraint=constraint)

    # Providers needed for your dataset + PCT robustness
    pip_install(venv_path, [
        "apache-airflow-providers-docker",
        "apache-airflow-providers-http",
        "apache-airflow-providers-postgres",
        "apache-airflow-providers-sqlite",
        "apache-airflow-providers-amazon",
        "apache-airflow-providers-cncf-kubernetes",
    ], constraint=constraint)

    # MySQL provider optional (macOS build issues)
    if with_mysql_provider:
        print("Installing apache-airflow-providers-mysql (may require system libs on macOS).")
        print("If it fails, install: brew install pkg-config mariadb-connector-c")
        pip_install_best_effort(venv_path, ["apache-airflow-providers-mysql"], constraint=constraint)

    # Experiment + evaluation deps
    pip_install(venv_path, ["torch"])
    pip_install(venv_path, ["pyyaml", "rouge-score", "bert-score", "transformers"])
    pip_install(venv_path, ["prefect", "dagster"])
    pip_install(venv_path, ["pylint", "radon", "bandit", "flake8"])

    # Ensure bert-score updated (optional)
    pip_install_best_effort(venv_path, ["-U", "bert-score"])

    # Verify CodeBERT works
    cb = check_codebert_load(venv_path)
    if not cb.ok:
        raise SystemExit(f"ERROR: CodeBERT check failed: {cb.detail}")

    print("\nSetup complete.")
    print(f"Activate with:\n  source {venv_path}/bin/activate\n")


# ----------------------------
# Checks (inside venv)
# ----------------------------

def check_codebert_load(venv_path: Path) -> CheckResult:
    py = str(venv_python(venv_path))
    code = (
        "from bert_score import BERTScorer;"
        "BERTScorer(model_type='microsoft/codebert-base', num_layers=12, lang='en', rescale_with_baseline=False, device='cpu');"
        "print('OK')"
    )
    try:
        r = sh([py, "-c", code], check=True)
        ok = "OK" in (r.stdout or "")
        return CheckResult(
            name="codebert_bertscore",
            ok=ok,
            detail="CodeBERT works with BERTScore (num_layers=12)" if ok else "Unexpected output",
            category="models"
        )
    except Exception as e:
        return CheckResult(
            name="codebert_bertscore",
            ok=False,
            detail=f"Failed to load CodeBERT with BERTScore: {e}",
            category="models",
            install_hint="pip install -U bert-score transformers torch"
        )

def venv_pkg_version(venv_path: Path, dist: str) -> Optional[str]:
    py = str(venv_python(venv_path))
    code = (
        "from importlib.metadata import version, PackageNotFoundError\n"
        f"dist={dist!r}\n"
        "try:\n"
        "  print(version(dist))\n"
        "except PackageNotFoundError:\n"
        "  print('')\n"
    )
    r = sh([py, "-c", code], check=True)
    vv = (r.stdout or "").strip()
    return vv or None

def check_dist(venv_path: Path, dist: str, category: str, required: bool, exact: Optional[str] = None) -> CheckResult:
    v = venv_pkg_version(venv_path, dist)
    if v is None:
        return CheckResult(
            name=dist,
            ok=not required,
            detail="NOT INSTALLED" if required else "NOT INSTALLED (optional)",
            category=category
        )
    if exact:
        ok = (v == exact)
        return CheckResult(
            name=dist,
            ok=ok,
            detail=f"installed={v} required={exact}",
            installed_version=v,
            required_version=exact,
            category=category
        )
    return CheckResult(name=dist, ok=True, detail=f"installed={v}", installed_version=v, category=category)

def run_checks_in_venv(venv_path: Path, airflow_version: str, strict: bool, skip_codebert_check: bool) -> List[CheckResult]:
    py = venv_python(venv_path)
    if not py.exists():
        return [CheckResult(
            name="venv_exists",
            ok=False,
            detail=f"venv python not found at {py}",
            category="runtime",
            install_hint="Run with --setup to create and install dependencies."
        )]

    results: List[CheckResult] = []
    results.append(CheckResult(name="venv_path", ok=True, detail=str(venv_path), category="runtime"))

    # ensure venv python version
    v = python_version_string(str(py)) or "unknown"
    results.append(CheckResult(
        name="venv_python_version",
        ok=(v == "3.10.13"),
        detail=f"venv python version={v}",
        installed_version=v,
        required_version="3.10.13",
        category="runtime",
    ))

    # orchestrators
    results.append(check_dist(venv_path, "apache-airflow", "orchestrators", True, exact=airflow_version))
    results.append(check_dist(venv_path, "prefect", "orchestrators", True))
    results.append(check_dist(venv_path, "dagster", "orchestrators", True))

    # core evaluator deps
    results.append(check_dist(venv_path, "PyYAML", "core", True))
    results.append(check_dist(venv_path, "rouge-score", "core", True))
    results.append(check_dist(venv_path, "bert-score", "core", True))
    results.append(check_dist(venv_path, "torch", "core", True))
    results.append(check_dist(venv_path, "transformers", "core", True))

    # tools
    results.append(check_dist(venv_path, "pylint", "quality_tools", required=strict))
    results.append(check_dist(venv_path, "radon", "quality_tools", required=strict))
    results.append(check_dist(venv_path, "bandit", "quality_tools", required=strict))
    results.append(check_dist(venv_path, "flake8", "quality_tools", required=strict))

    # providers (required set)
    results.append(check_dist(venv_path, "apache-airflow-providers-docker", "airflow_providers", True))
    results.append(check_dist(venv_path, "apache-airflow-providers-http", "airflow_providers", True))
    results.append(check_dist(venv_path, "apache-airflow-providers-postgres", "airflow_providers", True))
    results.append(check_dist(venv_path, "apache-airflow-providers-sqlite", "airflow_providers", True))
    results.append(check_dist(venv_path, "apache-airflow-providers-amazon", "airflow_providers", True))
    results.append(check_dist(venv_path, "apache-airflow-providers-cncf-kubernetes", "airflow_providers", True))

    # optional providers
    results.append(check_dist(venv_path, "apache-airflow-providers-mysql", "airflow_providers", False))
    results.append(check_dist(venv_path, "apache-airflow-providers-google", "airflow_providers", False))

    if not skip_codebert_check:
        results.append(check_codebert_load(venv_path))

    return results


# ----------------------------
# Reporting
# ----------------------------

def print_report(results: List[CheckResult], as_json: bool) -> int:
    failures = [r for r in results if not r.ok]

    if as_json:
        payload = {
            "system": platform.platform(),
            "host_python": sys.version,
            "ok": len(failures) == 0,
            "results": [r.__dict__ for r in results],
            "failures": [r.__dict__ for r in failures],
        }
        print(json.dumps(payload, indent=2))
        return 0 if not failures else 2

    print("\n" + "=" * 88)
    print("ENV PREFLIGHT CHECK (venv-based)")
    print("=" * 88)
    print(f"System:      {platform.platform()}")
    print(f"Host Python: {sys.version.split()[0]}")
    print("")

    cats: Dict[str, List[CheckResult]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r)

    for cat in sorted(cats.keys()):
        print(f"[{cat}]")
        for r in cats[cat]:
            status = "✓ OK" if r.ok else "✗ FAIL"
            iv = f" ({r.installed_version})" if r.installed_version else ""
            req = f" [required={r.required_version}]" if r.required_version else ""
            print(f"  {status:7} {r.name}{iv}{req} — {r.detail}")
            if not r.ok and r.install_hint:
                print(f"          fix: {r.install_hint}")
        print("")

    if failures:
        print("RESULT: FAIL")
        return 2
    print("RESULT: PASS")
    return 0


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight check + auto setup for Prompt2Workflow experiments (Airflow 2.8.4 on Python 3.10.13).")
    parser.add_argument("--venv-path", default=".venv-airflow284")
    parser.add_argument("--airflow-version", default="2.8.4")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--skip-codebert-check", action="store_true")

    parser.add_argument("--setup", action="store_true", help="Create venv (if missing) and install all dependencies.")
    parser.add_argument("--python-cmd", default=None,
                        help="Preferred python executable for creating venv. If not provided, script finds pyenv 3.10.13 automatically.")
    parser.add_argument("--with-mysql-provider", action="store_true", help="Attempt to install MySQL provider (may fail without system libs).")

    args = parser.parse_args()
    venv_path = Path(args.venv_path)

    if args.setup:
        py310 = ensure_python_31013_or_fail(args.python_cmd)
        print(f"Using Python for venv creation: {py310} (version={python_version_string(py310)})")
        setup_env(
            venv_path=venv_path,
            python_cmd=py310,
            airflow_version=args.airflow_version,
            with_mysql_provider=args.with_mysql_provider,
        )

    results = run_checks_in_venv(
        venv_path=venv_path,
        airflow_version=args.airflow_version,
        strict=args.strict,
        skip_codebert_check=args.skip_codebert_check,
    )
    exit_code = print_report(results, as_json=args.json)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()