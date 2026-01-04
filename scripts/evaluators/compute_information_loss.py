#!/usr/bin/env python3
"""
Compute Information Loss (Spec-based) + Format Divergence (Chunked BERTScore)
============================================================================

Outputs two complementary metrics per pipeline p and class Ck:

(1) Content information loss L_k(p)  (PRIMARY; deterministic)
----------------------------------------------------------------
Compare variant spec (pipeline_variants/<pipeline_id>/specs/Ck.json)
to oracle spec (pipeline_specs/<pipeline_id>.pipelinespec.json) via fact sets.

- Extract deterministic atomic facts F0(p) from oracle and Fk(p) from variant
- Compute corpus-wide IDF/self-information weights from oracle facts:
      w(f) = log((N + 1) / (df(f) + 1))

- Weighted retention:
      R_k(p) = sum_{f in F0 ∩ Fk} w(f) / sum_{f in F0} w(f)
  Weighted loss:
      L_k(p) = 1 - R_k(p)

Also per-category:
      L_k^(c)(p), R_k^(c)(p)

(2) Format divergence D_format  (SECONDARY; BERT-family; robust for long prompts)
---------------------------------------------------------------------------------
Compare prompt texts:
  pipeline_variants/<pipeline_id>/Ck_*.txt vs pipeline_variants/<pipeline_id>/C0_*.txt

We compute:
      D_format = 1 - BERTScore_F1(Tk, T0)

Prompts often exceed max length (e.g., 881 tokens > 512), so we do CHUNKED BERTScore:
- tokenize using the same tokenizer as the model
- split into overlapping windows (chunk_tokens, overlap)
- score aligned chunk pairs
- aggregate P/R with token-length weighting, then compute F1

Robustness improvements in this version:
- Clamp R and L into [0,1] (removes tiny float artifacts)
- Optional prompt normalization to reduce whitespace-only chunks
- Handle blank chunks deterministically without calling BERTScore (prevents warnings)
- Optional suppression of tokenizer "sequence length > max" warnings (safe because we chunk)

Usage:
  python scripts/evaluators/compute_information_loss.py \
    --pipeline-specs-dir pipeline_specs \
    --pipeline-variants-dir pipeline_variants \
    --out results/information_loss.csv \
    --device mps \
    --bert-model microsoft/codebert-base
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# -----------------------------
# Optional BERTScore dependencies
# -----------------------------
_HAS_BERTSCORE = False
try:
    import torch  # type: ignore
    from bert_score import BERTScorer  # type: ignore
    from transformers import AutoConfig, AutoTokenizer  # type: ignore
    _HAS_BERTSCORE = True
except Exception:
    torch = None
    BERTScorer = None
    AutoConfig = None
    AutoTokenizer = None
    _HAS_BERTSCORE = False


DEFAULT_CLASSES = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

CATEGORIES = [
    "schedule",
    "control_flow",
    "mechanism",
    "parameters",
    "external_systems",
    "data_artifacts",
    "env_infra",
    "failure_handling",
]


# -----------------------------
# I/O + math helpers
# -----------------------------
def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (int, float, bool)):
        return str(x)
    try:
        return json.dumps(x, sort_keys=True, ensure_ascii=False)
    except Exception:
        return str(x)


def normalize_list_str(xs: Any) -> List[str]:
    if not isinstance(xs, list):
        return []
    return [safe_str(x).strip() for x in xs if safe_str(x).strip()]


def mean_or_none(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def clamp01(x: Optional[float], eps: float = 1e-12) -> Optional[float]:
    """
    Clamp into [0,1] and remove tiny floating noise like -2e-16.
    """
    if x is None:
        return None
    xf = float(x)
    if abs(xf) < eps:
        return 0.0
    if abs(xf - 1.0) < eps:
        return 1.0
    return max(0.0, min(1.0, xf))


def sum_weights(facts: Set[str], weights: Dict[str, float]) -> float:
    return float(sum(weights.get(f, 0.0) for f in facts))


def normalize_prompt_text(text: str) -> str:
    """
    Deterministic normalization to reduce "blank chunk" artifacts without destroying structure.
    - normalize newlines
    - strip trailing whitespace per line
    - collapse huge blank blocks
    - collapse runs of spaces/tabs
    - strip overall
    """
    if text is None:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = "\n".join(line.rstrip() for line in t.split("\n"))
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# -----------------------------
# File discovery
# -----------------------------
def discover_oracle_specs(pipeline_specs_dir: Path) -> List[Path]:
    return sorted(pipeline_specs_dir.glob("*.pipelinespec.json"))


def canonical_pipeline_id_from_oracle_path(p: Path) -> str:
    return p.name.replace(".pipelinespec.json", "")


def find_variant_spec(pipeline_variants_dir: Path, pipeline_id: str, class_id: str) -> Optional[Path]:
    p = pipeline_variants_dir / pipeline_id / "specs" / f"{class_id}.json"
    if p.exists():
        return p
    if class_id == "C9":
        p2 = pipeline_variants_dir / pipeline_id / "specs" / "C0.json"
        if p2.exists():
            return p2
    return None


def find_prompt_text(pipeline_variants_dir: Path, pipeline_id: str, class_id: str) -> Optional[Path]:
    d = pipeline_variants_dir / pipeline_id
    if not d.exists():
        return None
    matches = sorted(d.glob(f"{class_id}_*.txt"))
    return matches[0] if matches else None


# -----------------------------
# Fact extraction (PipelineSpec -> category->facts)
# -----------------------------
def extract_facts_from_spec(spec: Dict[str, Any]) -> Dict[str, Set[str]]:
    facts: Dict[str, Set[str]] = {c: set() for c in CATEGORIES}

    def add(cat: str, fact: str) -> None:
        if fact and str(fact).strip():
            facts[cat].add(str(fact).strip())

    # schedule
    sched = spec.get("schedule") or {}
    if isinstance(sched, dict):
        for k in ["schedule_type", "schedule_expression", "timezone", "start_date", "catchup_backfill"]:
            if k in sched and sched.get(k) is not None:
                add("schedule", f"{k}={safe_str(sched.get(k))}")

    # control_flow
    cf = spec.get("control_flow") or {}
    if isinstance(cf, dict):
        for e in cf.get("edges") or []:
            if not isinstance(e, dict):
                continue
            fr = safe_str(e.get("from")).strip()
            to = safe_str(e.get("to")).strip()
            if fr and to:
                add("control_flow", f"edge:{fr}->{to}")

        for g in cf.get("parallel_groups") or []:
            if not isinstance(g, dict):
                continue
            gid = safe_str(g.get("group_id") or "group").strip()
            steps = sorted(normalize_list_str(g.get("steps")))
            if steps:
                add("control_flow", f"parallel_group:{gid}:" + ",".join(steps))

        for g in cf.get("gates") or []:
            if not isinstance(g, dict):
                continue
            gtype = safe_str(g.get("type")).strip()
            step = safe_str(g.get("step")).strip()
            waits = safe_str(g.get("waits_for")).strip()
            if gtype or step or waits:
                add("control_flow", f"gate:type={gtype}|step={step}|waits_for={waits}")

        for b in cf.get("branch_points") or []:
            if not isinstance(b, dict):
                continue
            step = safe_str(b.get("step")).strip()
            merge = safe_str(b.get("merge_step")).strip()
            if step or merge:
                add("control_flow", f"branch_point:step={step}|merge={merge}")
            for br in b.get("branches") or []:
                if not isinstance(br, dict):
                    continue
                cond = safe_str(br.get("condition")).strip()
                nxt = sorted(normalize_list_str(br.get("next_steps")))
                add("control_flow", f"branch:at={step}|cond={cond}|next=" + ",".join(nxt))

    # external_systems (global)
    for s in spec.get("external_systems") or []:
        if not isinstance(s, dict):
            continue
        stype = safe_str(s.get("type")).strip()
        name = safe_str(s.get("name")).strip()
        ident = safe_str(s.get("identifier")).strip()

        if stype:
            add("external_systems", f"system_type:{stype}")
        if stype and name:
            add("external_systems", f"system_name:{stype}:{name}")
        if stype and ident:
            add("external_systems", f"system_identifier:{stype}:{ident}")

    # data_artifacts (global)
    da = spec.get("data_artifacts") or {}
    if isinstance(da, dict):
        for key, prefix in [("files", "file"), ("tables", "table"), ("buckets", "bucket"), ("topics", "topic")]:
            for it in da.get(key) or []:
                if not isinstance(it, dict):
                    continue
                ident = safe_str(it.get("identifier")).strip()
                if ident:
                    add("data_artifacts", f"{prefix}:{ident}")

    # steps: mechanism/params/env/failure + step IO + step system refs
    steps = spec.get("steps") or []
    if isinstance(steps, list):
        for st in steps:
            if not isinstance(st, dict):
                continue
            sid = safe_str(st.get("step_id") or st.get("name") or "").strip()
            if not sid:
                continue

            mech = st.get("mechanism")
            if mech is not None:
                add("mechanism", f"mechanism:{sid}={safe_str(mech)}")

            for kv in st.get("parameters") or []:
                if not isinstance(kv, dict):
                    continue
                k = safe_str(kv.get("key")).strip()
                v = kv.get("value")
                if k:
                    if v is None or safe_str(v).strip() == "":
                        add("parameters", f"param:{sid}:{k}")
                    else:
                        add("parameters", f"param:{sid}:{k}={safe_str(v)}")

            for io_key, io_prefix in [("inputs", "in"), ("outputs", "out")]:
                for item in st.get(io_key) or []:
                    if not isinstance(item, dict):
                        continue
                    t = safe_str(item.get("type")).strip()
                    ident = item.get("identifier")
                    if t:
                        add("data_artifacts", f"step_{io_prefix}_type:{sid}:{t}")
                    if ident is not None and safe_str(ident).strip():
                        add("data_artifacts", f"step_{io_prefix}_id:{sid}:{t}:{safe_str(ident).strip()}")

            for r in st.get("external_system_refs") or []:
                if not isinstance(r, dict):
                    continue
                rtype = safe_str(r.get("type")).strip()
                rname = safe_str(r.get("name")).strip()
                rid = safe_str(r.get("identifier")).strip()
                if rtype:
                    add("external_systems", f"step_system_type:{sid}:{rtype}")
                if rtype and rname:
                    add("external_systems", f"step_system_name:{sid}:{rtype}:{rname}")
                if rtype and rid:
                    add("external_systems", f"step_system_identifier:{sid}:{rtype}:{rid}")

            env = st.get("env_infra") or {}
            if isinstance(env, dict):
                for ev in env.get("env_vars") or []:
                    evs = safe_str(ev).strip()
                    if evs:
                        add("env_infra", f"env_var:{sid}:{evs}")
                for m in env.get("mounts") or []:
                    ms = safe_str(m).strip()
                    if ms:
                        add("env_infra", f"mount:{sid}:{ms}")
                net = env.get("network")
                if net is not None and safe_str(net).strip():
                    add("env_infra", f"network:{sid}:{safe_str(net).strip()}")
                for o in env.get("other") or []:
                    os_ = safe_str(o).strip()
                    if os_:
                        add("env_infra", f"env_other:{sid}:{os_}")

            fh = st.get("failure_handling") or {}
            if isinstance(fh, dict):
                for k in ["retries", "retry_delay", "timeout"]:
                    if fh.get(k) is not None:
                        add("failure_handling", f"{k}:{sid}={safe_str(fh.get(k))}")
                for a in fh.get("alerts") or []:
                    av = safe_str(a).strip()
                    if av:
                        add("failure_handling", f"alert:{sid}:{av}")
                for idem in fh.get("idempotency") or []:
                    iv = safe_str(idem).strip()
                    if iv:
                        add("failure_handling", f"idempotency:{sid}:{iv}")

    return facts


# -----------------------------
# IDF weights from oracle facts
# -----------------------------
@dataclass
class IDFModel:
    N: int
    df: Dict[str, int]
    weights: Dict[str, float]


def compute_idf_model(oracle_facts_by_pipeline: Dict[str, Set[str]]) -> IDFModel:
    pipeline_ids = list(oracle_facts_by_pipeline.keys())
    N = len(pipeline_ids)
    df: Dict[str, int] = {}

    for pid in pipeline_ids:
        for f in oracle_facts_by_pipeline[pid]:
            df[f] = df.get(f, 0) + 1

    weights: Dict[str, float] = {}
    for f, dfi in df.items():
        weights[f] = math.log((N + 1.0) / (dfi + 1.0))

    return IDFModel(N=N, df=df, weights=weights)


# -----------------------------
# Chunked BERTScore (format divergence)
# -----------------------------
@dataclass
class ChunkInfo:
    text: str
    eff_len: int  # non-overlap token contribution


class PromptChunker:
    def __init__(self, tokenizer, chunk_tokens: int, overlap: int):
        self.tokenizer = tokenizer
        self.chunk_tokens = int(chunk_tokens)
        self.overlap = int(overlap)
        if self.chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be > 0")
        if self.overlap < 0:
            raise ValueError("overlap must be >= 0")
        if self.overlap >= self.chunk_tokens:
            raise ValueError("overlap must be < chunk_tokens")
        self.stride = self.chunk_tokens - self.overlap
        self._cache: Dict[Tuple[str, str], Tuple[int, List[ChunkInfo]]] = {}

    def tokenize_len(self, text: str) -> int:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return len(ids)

    def chunks(self, key: Tuple[str, str], text: str) -> Tuple[int, List[ChunkInfo]]:
        if key in self._cache:
            return self._cache[key]

        ids = self.tokenizer.encode(text, add_special_tokens=False)
        n = len(ids)
        chunks: List[ChunkInfo] = []

        if n == 0:
            self._cache[key] = (0, chunks)
            return 0, chunks

        start = 0
        first = True
        while start < n:
            end = min(start + self.chunk_tokens, n)
            window = ids[start:end]
            chunk_text = self.tokenizer.decode(
                window,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            window_len = len(window)
            eff = window_len if first else max(0, window_len - self.overlap)
            first = False

            chunks.append(ChunkInfo(text=chunk_text, eff_len=eff))
            start += self.stride

        self._cache[key] = (n, chunks)
        return n, chunks


def init_bertscorer(model: str, device: Optional[str]) -> Tuple[Any, Any, str, int]:
    """
    Returns (scorer, tokenizer, device, model_max_len_tokens).
    model_max_len_tokens is the tokenizer's original max length (before we suppress warnings).
    """
    if not _HAS_BERTSCORE:
        raise RuntimeError("Missing dependencies. Install: pip install bert-score torch transformers")

    # device selection
    dev = device
    if dev is None or str(dev).strip().lower() == "auto":
        try:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                dev = "mps"
            elif torch.cuda.is_available():
                dev = "cuda"
            else:
                dev = "cpu"
        except Exception:
            dev = "cpu"
    dev = str(dev)

    # infer layers (CodeBERT often needs this)
    num_layers = None
    try:
        cfg = AutoConfig.from_pretrained(model)
        num_layers = getattr(cfg, "num_hidden_layers", None)
    except Exception:
        num_layers = None
    if num_layers is None and model == "microsoft/codebert-base":
        num_layers = 12

    scorer = BERTScorer(
        model_type=model,
        num_layers=num_layers,
        lang="en",
        rescale_with_baseline=False,
        device=dev,
    )

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

    # Capture original max length for sanity checks
    orig_max = getattr(tokenizer, "model_max_length", 512)
    try:
        orig_max = int(orig_max)
    except Exception:
        orig_max = 512
    if orig_max > 100000:  # some tokenizers use huge sentinel values
        orig_max = 512

    return scorer, tokenizer, dev, orig_max


def compute_format_divergence_chunked(
    rows: List[Dict[str, Any]],
    prompt_texts: Dict[Tuple[str, str], str],
    *,
    classes: List[str],
    bert_model: str,
    device: Optional[str],
    batch_size: int,
    chunk_tokens: int,
    chunk_overlap: int,
    normalize_prompts: bool,
    quiet_transformers_warnings: bool,
    blank_chunk_policy: str = "neutral",  # neutral: both blank=>1; one blank=>0
) -> None:
    """
    Adds to each row:
      prompt_tokens_C0, prompt_tokens_Ck,
      bert_p, bert_r, bert_f1 (chunked agg),
      D_format = 1 - bert_f1,
      bert_model/device/chunk metadata.

    Handles blank chunks without calling BERTScore to avoid warnings.
    """
    log = logging.getLogger("compute_information_loss")

    if quiet_transformers_warnings:
        # Hide tokenizer warnings like "sequence length > max length"
        for name in ["transformers.tokenization_utils_base", "transformers.tokenization_utils_fast"]:
            logging.getLogger(name).setLevel(logging.ERROR)

    scorer, tokenizer, dev, model_max_len = init_bertscorer(bert_model, device)

    # Ensure chunk_tokens safely below model max (BERTScore adds special tokens)
    safe_chunk_max = max(8, model_max_len - 2)
    if chunk_tokens > safe_chunk_max:
        log.warning(f"--bert-chunk-tokens {chunk_tokens} > safe max {safe_chunk_max}; clamping.")
        chunk_tokens = safe_chunk_max

    # Suppress warning when encoding *full* long prompts for chunking (we will chunk anyway)
    # (This does not affect BERTScore's internal tokenizer.)
    tokenizer.model_max_length = 10**9

    chunker = PromptChunker(tokenizer, chunk_tokens=chunk_tokens, overlap=chunk_overlap)

    # Accumulators by row index
    acc: Dict[int, Dict[str, float]] = {}

    # Flattened list of chunk pairs that need BERTScore:
    # (row_idx, wc, wr, cand_chunk_text, ref_chunk_text)
    pairs: List[Tuple[int, int, int, str, str]] = []

    blank_both = 0
    blank_one = 0
    total_pairs = 0

    def _get_text(pid: str, cid: str) -> Optional[str]:
        t = prompt_texts.get((pid, cid))
        if not isinstance(t, str):
            return None
        return normalize_prompt_text(t) if normalize_prompts else t

    for idx, r in enumerate(rows):
        pid = r["pipeline_id"]
        cid = r["class_id"]

        # Default metadata (for all rows)
        r["bert_model"] = bert_model
        r["bert_device"] = dev
        r["bert_chunk_tokens"] = int(chunk_tokens)
        r["bert_chunk_overlap"] = int(chunk_overlap)
        r["prompt_tokens_C0"] = None
        r["prompt_tokens_Ck"] = None

        if cid == "C0":
            t0 = _get_text(pid, "C0") or ""
            len0 = chunker.tokenize_len(t0) if t0 else 0
            r["prompt_tokens_C0"] = len0
            r["prompt_tokens_Ck"] = len0
            r["bert_p"] = 1.0
            r["bert_r"] = 1.0
            r["bert_f1"] = 1.0
            r["D_format"] = 0.0
            continue

        t0 = _get_text(pid, "C0")
        tk = _get_text(pid, cid)
        if not t0 or not tk:
            r["bert_p"] = None
            r["bert_r"] = None
            r["bert_f1"] = None
            r["D_format"] = None
            continue

        len_k, chunks_k = chunker.chunks((pid, cid), tk)
        len_0, chunks_0 = chunker.chunks((pid, "C0"), t0)

        r["prompt_tokens_C0"] = len_0
        r["prompt_tokens_Ck"] = len_k

        acc[idx] = {
            "P_num": 0.0,
            "R_num": 0.0,
            "P_den": float(len_k),
            "R_den": float(len_0),
        }

        m = min(len(chunks_k), len(chunks_0))
        for i in range(m):
            ck = chunks_k[i]
            c0 = chunks_0[i]

            cand_blank = (ck.text.strip() == "")
            ref_blank = (c0.text.strip() == "")

            total_pairs += 1
            wc = ck.eff_len
            wr = c0.eff_len

            if cand_blank and ref_blank:
                blank_both += 1
                if blank_chunk_policy == "neutral":
                    acc[idx]["P_num"] += float(wc) * 1.0
                    acc[idx]["R_num"] += float(wr) * 1.0
                else:
                    # conservative fallback: treat as 0 similarity
                    acc[idx]["P_num"] += 0.0
                    acc[idx]["R_num"] += 0.0
                continue

            if cand_blank or ref_blank:
                blank_one += 1
                # If one side is blank and the other is not, similarity should be 0.
                acc[idx]["P_num"] += 0.0
                acc[idx]["R_num"] += 0.0
                continue

            pairs.append((idx, wc, wr, ck.text, c0.text))

    # Batch score chunk pairs
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        idxs = [b[0] for b in batch]
        wcs = [b[1] for b in batch]
        wrs = [b[2] for b in batch]
        cands = [b[3] for b in batch]
        refs = [b[4] for b in batch]

        P, R, _F = scorer.score(cands, refs)
        P_list = [float(x) for x in P.tolist()]
        R_list = [float(x) for x in R.tolist()]

        for j, row_idx in enumerate(idxs):
            a = acc.get(row_idx)
            if a is None:
                continue
            a["P_num"] += float(wcs[j]) * P_list[j]
            a["R_num"] += float(wrs[j]) * R_list[j]

    # Finalize per row
    for idx, a in acc.items():
        P_den = a["P_den"]
        R_den = a["R_den"]

        P_val = (a["P_num"] / P_den) if P_den > 0 else None
        R_val = (a["R_num"] / R_den) if R_den > 0 else None

        if P_val is None or R_val is None:
            F1 = None
        else:
            F1 = (2.0 * P_val * R_val / (P_val + R_val)) if (P_val + R_val) > 0 else 0.0

        P_val = clamp01(P_val)
        R_val = clamp01(R_val)
        F1 = clamp01(F1)

        rows[idx]["bert_p"] = P_val
        rows[idx]["bert_r"] = R_val
        rows[idx]["bert_f1"] = F1
        rows[idx]["D_format"] = (1.0 - F1) if F1 is not None else None

    log.info(
        f"BERTScore chunking summary: total_aligned_pairs={total_pairs}, "
        f"scored_pairs={len(pairs)}, blank_both={blank_both}, blank_one={blank_one}, "
        f"chunk_tokens={chunk_tokens}, overlap={chunk_overlap}, model_max_len={model_max_len}"
    )


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute spec-based information loss (IDF weighted) and prompt format divergence (chunked BERTScore)."
    )
    ap.add_argument("--pipeline-specs-dir", default="pipeline_specs")
    ap.add_argument("--pipeline-variants-dir", default="pipeline_variants")

    ap.add_argument("--classes", default=",".join(DEFAULT_CLASSES))
    ap.add_argument("--pipelines", default="", help="Optional comma-separated pipeline_ids to output")
    ap.add_argument(
        "--idf-scope",
        default="all",
        choices=["all", "filtered"],
        help="Compute IDF weights from all oracle pipelines (recommended) or only filtered pipelines.",
    )

    ap.add_argument("--out", default="results/information_loss.csv")

    ap.add_argument("--skip-format-divergence", action="store_true")
    ap.add_argument("--normalize-prompts", dest="normalize_prompts", action="store_true", default=True)
    ap.add_argument("--no-normalize-prompts", dest="normalize_prompts", action="store_false")

    ap.add_argument("--quiet-transformers-warnings", dest="quiet_tf_warnings", action="store_true", default=True)
    ap.add_argument("--show-transformers-warnings", dest="quiet_tf_warnings", action="store_false")

    ap.add_argument("--bert-model", default="microsoft/codebert-base")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    ap.add_argument("--bert-batch-size", type=int, default=8)
    ap.add_argument("--bert-chunk-tokens", type=int, default=256)
    ap.add_argument("--bert-chunk-overlap", type=int, default=32)

    ap.add_argument("--write-weights-json", default=None)
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s - %(message)s")
    log = logging.getLogger("compute_information_loss")

    pipeline_specs_dir = Path(args.pipeline_specs_dir)
    pipeline_variants_dir = Path(args.pipeline_variants_dir)
    out_path = Path(args.out)

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    pipelines_filter = [p.strip() for p in args.pipelines.split(",") if p.strip()] if args.pipelines else None
    pipelines_filter_set = set(pipelines_filter) if pipelines_filter else None

    oracle_files = discover_oracle_specs(pipeline_specs_dir)
    if not oracle_files:
        raise SystemExit(f"No oracle specs found in {pipeline_specs_dir}")

    # Determine which pipelines contribute to IDF and which are output
    idf_pipeline_ids: Set[str] = set()
    output_pipeline_ids: Set[str] = set()

    for f in oracle_files:
        pid = canonical_pipeline_id_from_oracle_path(f)
        if pipelines_filter_set is None or pid in pipelines_filter_set:
            output_pipeline_ids.add(pid)
        if args.idf_scope == "all":
            idf_pipeline_ids.add(pid)
        else:
            if pipelines_filter_set is None or pid in pipelines_filter_set:
                idf_pipeline_ids.add(pid)

    if not output_pipeline_ids:
        raise SystemExit("No output pipelines selected (check --pipelines filter).")

    # Load oracle facts
    oracle_facts_all_by_pipeline: Dict[str, Set[str]] = {}
    oracle_facts_bycat_by_pipeline: Dict[str, Dict[str, Set[str]]] = {}

    for f in oracle_files:
        pid = canonical_pipeline_id_from_oracle_path(f)
        if pid not in idf_pipeline_ids and pid not in output_pipeline_ids:
            continue
        spec = read_json(f)
        facts_by_cat = extract_facts_from_spec(spec)
        all_facts = set().union(*facts_by_cat.values())
        oracle_facts_all_by_pipeline[pid] = all_facts
        oracle_facts_bycat_by_pipeline[pid] = facts_by_cat

    oracle_for_idf = {pid: oracle_facts_all_by_pipeline[pid] for pid in idf_pipeline_ids if pid in oracle_facts_all_by_pipeline}
    if not oracle_for_idf:
        raise SystemExit("No oracle facts collected for IDF (check --idf-scope and filters).")

    idf = compute_idf_model(oracle_for_idf)
    weights = idf.weights
    log.info(f"IDF scope pipelines: N={idf.N}")

    if args.write_weights_json:
        wpath = Path(args.write_weights_json)
        wpath.parent.mkdir(parents=True, exist_ok=True)
        wpath.write_text(
            json.dumps(
                {
                    "generated_at": datetime.utcnow().isoformat() + "Z",
                    "idf_scope": args.idf_scope,
                    "N_pipelines": idf.N,
                    "weights_formula": "w(f)=log((N+1)/(df+1))",
                    "df": idf.df,
                    "weights": idf.weights,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        log.info(f"Wrote weights: {wpath}")

    # Load prompt texts for output pipelines
    prompt_texts: Dict[Tuple[str, str], str] = {}
    for pid in sorted(output_pipeline_ids):
        for cid in set(classes + ["C0"]):
            ptxt = find_prompt_text(pipeline_variants_dir, pid, cid)
            if ptxt and ptxt.exists():
                txt = ptxt.read_text(encoding="utf-8")
                prompt_texts[(pid, cid)] = normalize_prompt_text(txt) if args.normalize_prompts else txt

    # Compute spec-based loss rows
    rows: List[Dict[str, Any]] = []

    for pid in sorted(output_pipeline_ids):
        if pid not in oracle_facts_all_by_pipeline:
            log.warning(f"Missing oracle facts for pipeline_id={pid}; skipping.")
            continue

        oracle_all = oracle_facts_all_by_pipeline[pid]
        oracle_by_cat = oracle_facts_bycat_by_pipeline[pid]

        oracle_weight_total = sum_weights(oracle_all, weights)
        oracle_cat_weight: Dict[str, float] = {cat: sum_weights(oracle_by_cat.get(cat, set()), weights) for cat in CATEGORIES}

        for cid in classes:
            vpath = find_variant_spec(pipeline_variants_dir, pid, cid)
            variant_spec = read_json(vpath) if (vpath and vpath.exists()) else None

            # prompt info
            p0 = find_prompt_text(pipeline_variants_dir, pid, "C0")
            pk = find_prompt_text(pipeline_variants_dir, pid, cid)
            t0 = prompt_texts.get((pid, "C0"))
            tk = prompt_texts.get((pid, cid))

            base_row: Dict[str, Any] = {
                "pipeline_id": pid,
                "class_id": cid,
                "variant_spec_path": str(vpath) if vpath else None,
                "variant_spec_missing": variant_spec is None,
                "prompt_path_C0": str(p0) if p0 else None,
                "prompt_path_Ck": str(pk) if pk else None,
                "prompt_chars_C0": len(t0) if isinstance(t0, str) else None,
                "prompt_chars_Ck": len(tk) if isinstance(tk, str) else None,
                "oracle_fact_count_total": len(oracle_all),
                "oracle_fact_weight_total": oracle_weight_total,
            }

            for cat in CATEGORIES:
                base_row[f"oracle_fact_count_{cat}"] = len(oracle_by_cat.get(cat, set()))
                base_row[f"oracle_fact_weight_{cat}"] = oracle_cat_weight[cat]

            if variant_spec is None:
                base_row.update({
                    "L_total": None, "R_total": None, "L_category_mean": None,
                    "variant_fact_count_total": None, "retained_fact_count_total": None, "retained_fact_weight_total": None
                })
                for cat in CATEGORIES:
                    base_row[f"L_{cat}"] = None
                    base_row[f"R_{cat}"] = None
                    base_row[f"variant_fact_count_{cat}"] = None
                    base_row[f"retained_fact_count_{cat}"] = None
                    base_row[f"retained_fact_weight_{cat}"] = None
                rows.append(base_row)
                continue

            variant_by_cat = extract_facts_from_spec(variant_spec)
            variant_all = set().union(*variant_by_cat.values())
            retained_all = oracle_all.intersection(variant_all)

            retained_weight_total = sum_weights(retained_all, weights)
            R_total = (retained_weight_total / oracle_weight_total) if oracle_weight_total > 0 else None
            L_total = (1.0 - R_total) if R_total is not None else None
            R_total = clamp01(R_total)
            L_total = clamp01(L_total)

            r = dict(base_row)
            r.update({
                "variant_fact_count_total": len(variant_all),
                "retained_fact_count_total": len(retained_all),
                "retained_fact_weight_total": retained_weight_total,
                "R_total": R_total,
                "L_total": L_total,
            })

            cat_losses: List[Optional[float]] = []
            for cat in CATEGORIES:
                oracle_cat = oracle_by_cat.get(cat, set())
                denom = oracle_cat_weight[cat]
                retained_cat = oracle_cat.intersection(variant_all)
                retained_cat_w = sum_weights(retained_cat, weights)

                R_cat = (retained_cat_w / denom) if denom > 0 else None
                L_cat = (1.0 - R_cat) if R_cat is not None else None
                R_cat = clamp01(R_cat)
                L_cat = clamp01(L_cat)

                r[f"variant_fact_count_{cat}"] = len(variant_by_cat.get(cat, set()))
                r[f"retained_fact_count_{cat}"] = len(retained_cat)
                r[f"retained_fact_weight_{cat}"] = retained_cat_w
                r[f"R_{cat}"] = R_cat
                r[f"L_{cat}"] = L_cat

                if L_cat is not None:
                    cat_losses.append(L_cat)

            r["L_category_mean"] = clamp01(mean_or_none(cat_losses))
            rows.append(r)

    # Compute chunked format divergence
    if args.skip_format_divergence:
        for r in rows:
            r["bert_model"] = None
            r["bert_device"] = None
            r["bert_chunk_tokens"] = None
            r["bert_chunk_overlap"] = None
            r["prompt_tokens_C0"] = None
            r["prompt_tokens_Ck"] = None
            r["bert_p"] = None
            r["bert_r"] = None
            r["bert_f1"] = None
            r["D_format"] = None
    else:
        if not _HAS_BERTSCORE:
            raise SystemExit("Missing deps: pip install bert-score torch transformers (or use --skip-format-divergence)")
        compute_format_divergence_chunked(
            rows=rows,
            prompt_texts=prompt_texts,
            classes=classes,
            bert_model=args.bert_model,
            device=args.device,
            batch_size=int(args.bert_batch_size),
            chunk_tokens=int(args.bert_chunk_tokens),
            chunk_overlap=int(args.bert_chunk_overlap),
            normalize_prompts=args.normalize_prompts,
            quiet_transformers_warnings=args.quiet_tf_warnings,
        )

    # Write CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_cols = [
        "pipeline_id", "class_id",
        "variant_spec_missing", "variant_spec_path",
        "L_total", "R_total", "L_category_mean",
        "oracle_fact_count_total", "variant_fact_count_total", "retained_fact_count_total",
        "oracle_fact_weight_total", "retained_fact_weight_total",
        "D_format", "bert_f1", "bert_p", "bert_r",
        "bert_model", "bert_device", "bert_chunk_tokens", "bert_chunk_overlap",
        "prompt_chars_C0", "prompt_chars_Ck",
        "prompt_tokens_C0", "prompt_tokens_Ck",
        "prompt_path_C0", "prompt_path_Ck",
    ]
    cat_cols: List[str] = []
    for cat in CATEGORIES:
        cat_cols.extend([
            f"L_{cat}", f"R_{cat}",
            f"oracle_fact_count_{cat}", f"variant_fact_count_{cat}", f"retained_fact_count_{cat}",
            f"oracle_fact_weight_{cat}", f"retained_fact_weight_{cat}",
        ])

    all_keys = sorted({k for r in rows for k in r.keys()})
    cols = [c for c in base_cols if c in all_keys] + [c for c in cat_cols if c in all_keys]
    for k in all_keys:
        if k not in cols:
            cols.append(k)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    log.info(f"Wrote: {out_path} (rows={len(rows)})")
    log.info("Interpretation:")
    log.info(" - L_total: spec-based content loss (IDF/self-information weighted).")
    log.info(" - D_format: 1 - chunked BERTScore F1(Tk, T0).")
    log.info(" - Expect C9: L_total≈0 but D_format>0.")


if __name__ == "__main__":
    main()