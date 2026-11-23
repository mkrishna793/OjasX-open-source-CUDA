"""Portable CPU/MPS deep neural safety encoders and Anthropic CC++ cascade."""

from __future__ import annotations

import ast
import math
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from .dedup import resolve_portable_device

DEFAULT_MODERNBERT_MODEL = "AllanK24/modernbert-Aegis-Content-Safety-2.0"
DEFAULT_NEMO_DOMAIN_MODEL = "nvidia/domain-classifier"
DEFAULT_PROTECTAI_PI_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"
DEFAULT_CODEBERT_MODEL = "mrm8488/codebert-base-finetuned-detect-insecure-code"
DEFAULT_CCPP_PROBES_PATH = Path(__file__).resolve().parent / "weights" / "ccpp_probes.pt"

_BUCKETS: Tuple[int, ...] = (64, 128, 192, 256, 384)

_PYTHON_DEF_RE = re.compile(r"(?m)^\s*(?:def|async\s+def|class|from\s+\S+\s+import|import\s+\S+)\b")
_C_CODE_RE = re.compile(
    r"(?m)(?:^\s*#\s*include\s*<|;\s*$|\b(?:int|void|char|size_t|uint\d+_t|struct|static\s+inline)\s+\w+\s*[\(;]|"
    r"\b(?:malloc|calloc|realloc|printf|snprintf|fprintf)\s*\()"
)
_C_SINK_PATTERNS: Tuple[Tuple[str, re.Pattern[str], float], ...] = (
    (
        "cwe120:unsafe_string_sink",
        re.compile(r"\b(?:strcpy|strcat|sprintf|vsprintf|gets|gets_s|scanf|sscanf|fscanf|strtok)\s*\("),
        0.78,
    ),
    (
        "cwe121:stack_buffer_copy",
        re.compile(
            r"\bchar\s+\w+\s*\[[^\]]+\][\s\S]{0,400}\b(?:strcpy|memcpy|memmove|sprintf|strcat)\s*\("
        ),
        0.76,
    ),
    (
        "cwe680:alloc_integer_overflow",
        re.compile(
            r"\b(?:malloc|calloc|realloc|reallocNew|alloca|heap_alloc|s_mallocf|mxMalloc|plat_alloc\w*)"
            r"\s*\([\s\S]{0,140}\*"
        ),
        0.74,
    ),
    (
        "cwe338:weak_prng",
        re.compile(r"\b(?:rand|srand)\s*\("),
        0.65,
    ),
    (
        "cwe328:weak_crypto",
        re.compile(r"\b(?:MD5|EVP_md5|SHA1|EVP_sha1|EVP_des\w*|DES_\w+|RC4)\s*\("),
        0.72,
    ),
    (
        "cwe78:os_command_exec",
        re.compile(
            r"\b(?:system|popen|execl|execv|execve|WinExec|ShellExecute\w*|CreateProcess\w*|"
            r"VirtualAlloc\w*|CreateRemoteThread)\s*\("
        ),
        0.82,
    ),
    (
        "cwe590:non_heap_free",
        re.compile(r"(?:char|int|struct)\s+\w+\s*\[[^\]]+\][\s\S]{0,300}\bfree\s*\("),
        0.72,
    ),
    (
        "cwe798:hardcoded_secret",
        re.compile(r'\b(?:password|passwd|secret|api_key|private_key)\s*=\s*"[^"\n]{3,}"'),
        0.70,
    ),
)

_MITRE_TTP_RE = re.compile(
    r"(?i)\b(?:fast[- ]?flux|c2\s+server|command\s+and\s+control|evade\s+detection|without\s+being\s+detected|"
    r"capture\s+screenshots?\s+of|keylogger|log\s+keystrokes?|ransomware|reverse\s+shell|bind\s+shell|"
    r"privilege\s+escalation|mimikatz|lsass|credential\s+(?:dump|harvest)|process\s+hollowing|dll\s+injection|"
    r"bypass\s+(?:amsi|edr|defender|antivirus|uac)|exfiltrat(?:e|ing|ion)|covert\s+channel|"
    r"persist(?:ence)?\s+mechanism|rootkit|botnet|shellcode|rop\s+chain|steal\s+(?:cookies|credentials|tokens))\b"
)

_SAFE_C_OR_PURE_RE = re.compile(
    r"(?i)\b(?:\w*alloc\w*|free|memcpy|memmove|strcpy|strcat|sprintf|vsprintf|gets|gets_s|scanf|sscanf|fscanf|"
    r"system|popen|execl|execv|execve|evp_\w*|md5|sha1|des_\w*|rc4)\b|"
    r"\b(?:char|uint\d+_t|int\d+_t)\s+\w+\s*\["
)
_DANGEROUS_PY_CALLS = frozenset({"exec", "compile", "__import__"})
_MEDIUM_PY_CALLS = frozenset({"eval"})
_DANGEROUS_PY_ATTR_CALLS = frozenset(
    {
        "os.system",
        "os.popen",
        "os.execl",
        "os.execv",
        "os.execve",
        "pickle.loads",
        "pickle.load",
        "marshal.loads",
        "marshal.load",
        "tempfile.mktemp",
    }
)
_MEDIUM_PY_ATTR_CALLS = frozenset({"hashlib.md5", "hashlib.sha1"})


def _load_hf_with_fallback(
    loader: Callable[..., Any],
    repo_id: str,
    *,
    local_files_only: bool = True,
    **kwargs: Any,
) -> Any:
    try:
        return loader(repo_id, local_files_only=local_files_only, **kwargs)
    except OSError:
        if not local_files_only:
            raise
        return loader(repo_id, local_files_only=False, **kwargs)


def _download_hf_with_fallback(
    repo_id: str,
    filename: str,
    *,
    local_files_only: bool = True,
) -> str:
    try:
        return hf_hub_download(repo_id, filename, local_files_only=local_files_only)
    except OSError:
        if not local_files_only:
            raise
        return hf_hub_download(repo_id, filename, local_files_only=False)


def _masked_layer_pool(
    hidden_states: Sequence[torch.Tensor],
    attention_mask: torch.Tensor,
    *,
    include_max: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Attention-masked mean (and optional max) pooling across all hidden layers [B, L, D]."""
    stacked = torch.stack(tuple(hidden_states), dim=1).to(dtype=torch.float32)  # [B, L, T, D]
    mask = attention_mask.to(dtype=torch.float32).unsqueeze(1).unsqueeze(-1)  # [B, 1, T, 1]
    denom = mask.sum(dim=2).clamp(min=1.0)  # [B, 1, 1]
    mean_pooled = (stacked * mask).sum(dim=2) / denom  # [B, L, D]
    if not include_max:
        return mean_pooled, None
    masked_vals = stacked.masked_fill(mask == 0.0, -1e4)
    max_pooled = masked_vals.max(dim=2).values
    return mean_pooled, max_pooled


def _pick_bucket(max_len: int, max_cap: int) -> int:
    for b in _BUCKETS:
        if max_len <= b and b <= max_cap:
            return b
    return max_cap


def _iter_length_buckets(
    tokenizer: Any,
    texts: Sequence[str],
    *,
    max_length: int,
    batch_size: int,
) -> List[Tuple[List[int], torch.Tensor, torch.Tensor]]:
    """Tokenize and sort into fixed length buckets to avoid MPSGraph recompilation."""
    if not texts:
        return []
    encoded = tokenizer(
        list(texts),
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    input_ids_list: List[List[int]] = encoded["input_ids"]
    lengths = [len(seq) for seq in input_ids_list]
    order = sorted(range(len(texts)), key=lambda i: lengths[i])
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)

    batches: List[Tuple[List[int], torch.Tensor, torch.Tensor]] = []
    step = max(1, int(batch_size))
    for start in range(0, len(order), step):
        idxs = order[start : start + step]
        local_max = max(1, max(lengths[i] for i in idxs))
        bucket_len = _pick_bucket(local_max, max_length)
        bsz = len(idxs)
        ids_t = torch.full((bsz, bucket_len), pad_id, dtype=torch.long)
        att_t = torch.zeros((bsz, bucket_len), dtype=torch.long)
        for row_i, orig_i in enumerate(idxs):
            seq = input_ids_list[orig_i][:bucket_len]
            if seq:
                ids_t[row_i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
                att_t[row_i, : len(seq)] = 1
        batches.append((idxs, ids_t, att_t))
    return batches


@dataclass
class ModernBERTBatchOutput:
    raw_unsafe_probs: torch.Tensor  # [B]
    layer_mean_states: torch.Tensor  # [B, 23, 768]
    layer_max_states: torch.Tensor  # [B, 23, 768]


@dataclass
class NeMoDomainBatchOutput:
    domain_labels: List[str]
    domain_probs: torch.Tensor  # [B, 26]
    all_layer_pooled_states: torch.Tensor  # [B, 13, 768]
    id2label: Dict[int, str]


@dataclass
class ProtectAIBatchOutput:
    injection_probs: torch.Tensor  # [B]
    all_layer_pooled_states: torch.Tensor  # [B, 13, 768]


@dataclass(frozen=True)
class CodeSecuritySignals:
    is_code: bool
    is_python: bool
    is_c_like: bool
    ast_risk: float
    cwe_sink_risk: float
    mitre_ttp_risk: float
    structural_risk: float
    matched_patterns: Tuple[str, ...]


@dataclass
class CodeBERTBatchOutput:
    insecure_probs: torch.Tensor  # [B] calibrated security score
    raw_margins: torch.Tensor  # [B] raw logits[:, 1] - logits[:, 0]
    calibrated_c_margin_probs: torch.Tensor  # [B]
    structural_signals: List[CodeSecuritySignals]
    all_layer_pooled_states: torch.Tensor  # [B, 13, 768]


@dataclass(frozen=True)
class CCPlusPlusSampleTrace:
    sample_index: int
    cascade_decision: str  # "PASS", "ESCALATE", "REFUSE"
    cascade_escalated: bool
    stage_a_probe_score: float
    stage_a_nl_score: float
    stage_a_cyber_score: float
    stage_a_dual_use_score: float
    mahalanobis_ood: float
    ood_elevated: bool
    stage_b_nl_score: float
    protectai_pi_prob: float
    codebert_risk_score: float
    nemo_domain_label: str
    nemo_domain_probs: Dict[str, float]
    dual_use_domain: str
    dual_use_prob: float
    exchange_risk_score: float
    final_harm_score: float
    reasons: Tuple[str, ...] = ()


def _extract_python_blocks(text: str) -> List[str]:
    blocks = re.findall(r"```(?:python|py)?\s*\n([\s\S]*?)```", text, flags=re.IGNORECASE)
    if blocks:
        return blocks
    return [text]


def _inspect_python_ast(code_text: str) -> Tuple[bool, float, List[str]]:
    """Inspect Python AST for dangerous execution, deserialization, or weak crypto sinks."""
    matched: List[str] = []
    max_risk = 0.0
    parsed_any = False

    for candidate in _extract_python_blocks(code_text):
        stripped = candidate.strip()
        if not stripped:
            continue
        try:
            tree = ast.parse(stripped)
        except SyntaxError:
            continue
        parsed_any = True
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name):
                    if func.id in _DANGEROUS_PY_CALLS:
                        matched.append(f"py_ast:{func.id}")
                        max_risk = max(max_risk, 0.85)
                    elif func.id in _MEDIUM_PY_CALLS:
                        matched.append(f"py_ast:{func.id}")
                        max_risk = max(max_risk, 0.45)
                elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                    qualified = f"{func.value.id}.{func.attr}"
                    if qualified in _DANGEROUS_PY_ATTR_CALLS:
                        matched.append(f"py_ast:{qualified}")
                        max_risk = max(max_risk, 0.82)
                    elif qualified in _MEDIUM_PY_ATTR_CALLS:
                        matched.append(f"py_ast:{qualified}")
                        max_risk = max(max_risk, 0.45)
                    elif func.value.id == "subprocess":
                        for kw in node.keywords:
                            if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                                matched.append("py_ast:subprocess_shell_true")
                                max_risk = max(max_risk, 0.88)
                    elif func.value.id == "yaml" and func.attr == "load":
                        has_safe_loader = any(
                            kw.arg == "Loader"
                            and isinstance(kw.value, ast.Name)
                            and "Safe" in kw.value.id
                            for kw in node.keywords
                        )
                        if not has_safe_loader:
                            matched.append("py_ast:yaml_unsafe_load")
                            max_risk = max(max_risk, 0.78)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in {"ctypes", "socket"}:
                        matched.append(f"py_ast:import_{alias.name}")
                        max_risk = max(max_risk, 0.45)
            elif isinstance(node, ast.ImportFrom) and node.module in {"ctypes", "socket"}:
                matched.append(f"py_ast:from_{node.module}")
                max_risk = max(max_risk, 0.45)
    return parsed_any, max_risk, matched


def extract_code_security_signals(text: str) -> CodeSecuritySignals:
    """Detect programming language structure, Python AST dangers, C/C++ CWEs, and MITRE TTPs."""
    if not isinstance(text, str):
        text = str(text)

    parsed_py, ast_risk, matched = _inspect_python_ast(text)
    is_py = parsed_py and bool(_PYTHON_DEF_RE.search(text) or "def " in text or "import " in text)
    is_c = bool(_C_CODE_RE.search(text)) and not is_py
    is_code = bool(is_py or is_c or "```" in text)

    cwe_risk = 0.0
    if is_c or "{" in text or ";" in text:
        for label, pattern, weight in _C_SINK_PATTERNS:
            if pattern.search(text):
                matched.append(label)
                cwe_risk = max(cwe_risk, weight)

    mitre_hits = _MITRE_TTP_RE.findall(text)
    mitre_risk = 0.0
    if mitre_hits:
        matched.append(f"mitre_ttp:{len(mitre_hits)}")
        mitre_risk = min(0.95, 0.72 + 0.08 * max(0, len(mitre_hits) - 1))

    structural = max(ast_risk, cwe_risk, mitre_risk)
    dedup_matched = tuple(dict.fromkeys(matched))
    return CodeSecuritySignals(
        is_code=is_code,
        is_python=is_py,
        is_c_like=is_c,
        ast_risk=float(ast_risk),
        cwe_sink_risk=float(cwe_risk),
        mitre_ttp_risk=float(mitre_risk),
        structural_risk=float(structural),
        matched_patterns=dedup_matched,
    )


class _NeMoDomainDeberta(nn.Module, PyTorchModelHubMixin):
    """Pure-PyTorch reproduction of NVIDIA NeMo Curator DeBERTa-v3 26-domain classifier."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        base_model = str(config.get("base_model", "microsoft/deberta-v3-base"))
        base_cfg = _load_hf_with_fallback(AutoConfig.from_pretrained, base_model, local_files_only=True)
        base_cfg.output_hidden_states = True
        self.model = AutoModel.from_config(base_cfg)
        self.dropout = nn.Dropout(float(config.get("fc_dropout", 0.2)))
        id2label = config.get("id2label", {})
        self.fc = nn.Linear(int(self.model.config.hidden_size), len(id2label))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        cls_feat = out.last_hidden_state[:, 0, :]
        logits = self.fc(self.dropout(cls_feat))
        probs = torch.softmax(logits, dim=-1)
        return probs, out.hidden_states


class ZeroCudaNeMoDomainClassifier:
    """Zero-CUDA Apple Silicon MPS / CPU wrapper for nvidia/domain-classifier."""

    def __init__(
        self,
        model_id: str = DEFAULT_NEMO_DOMAIN_MODEL,
        *,
        device: str = "auto",
        local_files_only: bool = True,
        max_length: int = 192,
    ) -> None:
        self.model_id = model_id
        self.device = resolve_portable_device(device)
        self.max_length = int(max_length)

        import json

        cfg_path = _download_hf_with_fallback(model_id, "config.json", local_files_only=local_files_only)
        self.config: Dict[str, Any] = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
        self.id2label: Dict[int, str] = {int(k): str(v) for k, v in self.config["id2label"].items()}
        self.label2id: Dict[str, int] = {str(v): int(k) for k, v in self.id2label.items()}

        self.tokenizer = _load_hf_with_fallback(
            AutoTokenizer.from_pretrained, model_id, local_files_only=local_files_only
        )
        self.model = _NeMoDomainDeberta(self.config)
        wt_path = _download_hf_with_fallback(
            model_id, "model.safetensors", local_files_only=local_files_only
        )
        state_dict = load_file(wt_path)
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(device=self.device, dtype=torch.float32).eval()

    def predict(
        self,
        texts: Union[str, Sequence[str]],
        *,
        batch_size: int = 32,
        max_length: Optional[int] = None,
    ) -> Tuple[Union[str, List[str]], torch.Tensor, torch.Tensor]:
        """Return (domain_label_or_labels, domain_probs [B, 26], all_layer_pooled_states [B, 13, 768])."""
        single = isinstance(texts, str)
        text_list = [texts] if single else list(texts)
        out = self.predict_batch(text_list, batch_size=batch_size, max_length=max_length)
        if single:
            return out.domain_labels[0], out.domain_probs[0], out.all_layer_pooled_states[0]
        return out.domain_labels, out.domain_probs, out.all_layer_pooled_states

    def predict_batch(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        max_length: Optional[int] = None,
    ) -> NeMoDomainBatchOutput:
        n = len(texts)
        num_classes = len(self.id2label)
        if n == 0:
            return NeMoDomainBatchOutput(
                domain_labels=[],
                domain_probs=torch.empty((0, num_classes), dtype=torch.float32),
                all_layer_pooled_states=torch.empty((0, 13, 768), dtype=torch.float32),
                id2label=self.id2label,
            )

        cap = int(max_length or self.max_length)
        batches = _iter_length_buckets(self.tokenizer, texts, max_length=cap, batch_size=batch_size)
        probs_out = torch.empty((n, num_classes), dtype=torch.float32)
        states_out: Optional[torch.Tensor] = None

        with torch.no_grad():
            for idxs, ids_t, att_t in batches:
                ids_dev = ids_t.to(self.device)
                att_dev = att_t.to(self.device)
                probs, hs = self.model(ids_dev, att_dev)
                mean_p, _ = _masked_layer_pool(hs, att_dev, include_max=False)
                if states_out is None:
                    states_out = torch.empty((n, mean_p.shape[1], mean_p.shape[2]), dtype=torch.float32)
                idx_t = torch.tensor(idxs, dtype=torch.long)
                probs_out[idx_t] = probs.detach().cpu().to(dtype=torch.float32)
                states_out[idx_t] = mean_p.detach().cpu().to(dtype=torch.float32)

        assert states_out is not None
        pred_ids = probs_out.argmax(dim=-1).tolist()
        labels = [self.id2label[int(i)] for i in pred_ids]
        return NeMoDomainBatchOutput(
            domain_labels=labels,
            domain_probs=probs_out,
            all_layer_pooled_states=states_out,
            id2label=self.id2label,
        )


class ModernBERTAllLayerExtractor:
    """Extracts all 23 ModernBERT hidden states (mean + max pooling) safely on MPS/CPU."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODERNBERT_MODEL,
        *,
        device: str = "auto",
        local_files_only: bool = True,
        max_length: int = 192,
    ) -> None:
        self.model_id = model_id
        self.device = resolve_portable_device(device)
        self.max_length = int(max_length)

        cfg = _load_hf_with_fallback(AutoConfig.from_pretrained, model_id, local_files_only=local_files_only)
        cfg.reference_compile = False
        cfg.output_hidden_states = True

        self.tokenizer = _load_hf_with_fallback(
            AutoTokenizer.from_pretrained, model_id, local_files_only=local_files_only
        )
        self.model = _load_hf_with_fallback(
            AutoModelForSequenceClassification.from_pretrained,
            model_id,
            config=cfg,
            local_files_only=local_files_only,
        )
        self.model = self.model.to(device=self.device, dtype=torch.float32).eval()
        id2label = getattr(self.model.config, "id2label", {0: "safe", 1: "unsafe"})
        self.unsafe_idx = next(
            (int(k) for k, v in id2label.items() if "unsafe" in str(v).lower()),
            1,
        )

    def extract_batch(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        max_length: Optional[int] = None,
    ) -> ModernBERTBatchOutput:
        n = len(texts)
        if n == 0:
            return ModernBERTBatchOutput(
                raw_unsafe_probs=torch.empty(0, dtype=torch.float32),
                layer_mean_states=torch.empty((0, 23, 768), dtype=torch.float32),
                layer_max_states=torch.empty((0, 23, 768), dtype=torch.float32),
            )

        cap = int(max_length or self.max_length)
        batches = _iter_length_buckets(self.tokenizer, texts, max_length=cap, batch_size=batch_size)
        raw_probs = torch.empty((n,), dtype=torch.float32)
        mean_out: Optional[torch.Tensor] = None
        max_out: Optional[torch.Tensor] = None

        with torch.no_grad():
            for idxs, ids_t, att_t in batches:
                ids_dev = ids_t.to(self.device)
                att_dev = att_t.to(self.device)
                out = self.model(input_ids=ids_dev, attention_mask=att_dev, output_hidden_states=True)
                mean_p, max_p = _masked_layer_pool(out.hidden_states, att_dev, include_max=True)
                assert max_p is not None
                if mean_out is None:
                    mean_out = torch.empty((n, mean_p.shape[1], mean_p.shape[2]), dtype=torch.float32)
                    max_out = torch.empty((n, max_p.shape[1], max_p.shape[2]), dtype=torch.float32)
                idx_t = torch.tensor(idxs, dtype=torch.long)
                probs = torch.softmax(out.logits.to(dtype=torch.float32), dim=-1)[:, self.unsafe_idx]
                raw_probs[idx_t] = probs.detach().cpu()
                mean_out[idx_t] = mean_p.detach().cpu()
                max_out[idx_t] = max_p.detach().cpu()

        assert mean_out is not None and max_out is not None
        return ModernBERTBatchOutput(
            raw_unsafe_probs=raw_probs,
            layer_mean_states=mean_out,
            layer_max_states=max_out,
        )


class ProtectAIPromptInjectionClassifier:
    """DeBERTa-v3 prompt-injection detector exposing all 13 hidden layers on MPS/CPU."""

    def __init__(
        self,
        model_id: str = DEFAULT_PROTECTAI_PI_MODEL,
        *,
        device: str = "auto",
        local_files_only: bool = True,
        max_length: int = 192,
    ) -> None:
        self.model_id = model_id
        self.device = resolve_portable_device(device)
        self.max_length = int(max_length)

        cfg = _load_hf_with_fallback(AutoConfig.from_pretrained, model_id, local_files_only=local_files_only)
        cfg.output_hidden_states = True

        self.tokenizer = _load_hf_with_fallback(
            AutoTokenizer.from_pretrained, model_id, local_files_only=local_files_only
        )
        self.model = _load_hf_with_fallback(
            AutoModelForSequenceClassification.from_pretrained,
            model_id,
            config=cfg,
            local_files_only=local_files_only,
        )
        self.model = self.model.to(device=self.device, dtype=torch.float32).eval()
        id2label = getattr(self.model.config, "id2label", {0: "SAFE", 1: "INJECTION"})
        self.injection_idx = next(
            (int(k) for k, v in id2label.items() if "injection" in str(v).lower()),
            1,
        )

    def predict_batch(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        max_length: Optional[int] = None,
    ) -> ProtectAIBatchOutput:
        n = len(texts)
        if n == 0:
            return ProtectAIBatchOutput(
                injection_probs=torch.empty(0, dtype=torch.float32),
                all_layer_pooled_states=torch.empty((0, 13, 768), dtype=torch.float32),
            )

        cap = int(max_length or self.max_length)
        batches = _iter_length_buckets(self.tokenizer, texts, max_length=cap, batch_size=batch_size)
        probs_out = torch.empty((n,), dtype=torch.float32)
        states_out: Optional[torch.Tensor] = None

        with torch.no_grad():
            for idxs, ids_t, att_t in batches:
                ids_dev = ids_t.to(self.device)
                att_dev = att_t.to(self.device)
                out = self.model(input_ids=ids_dev, attention_mask=att_dev, output_hidden_states=True)
                mean_p, _ = _masked_layer_pool(out.hidden_states, att_dev, include_max=False)
                if states_out is None:
                    states_out = torch.empty((n, mean_p.shape[1], mean_p.shape[2]), dtype=torch.float32)
                idx_t = torch.tensor(idxs, dtype=torch.long)
                probs = torch.softmax(out.logits.to(dtype=torch.float32), dim=-1)[:, self.injection_idx]
                probs_out[idx_t] = probs.detach().cpu()
                states_out[idx_t] = mean_p.detach().cpu()

        assert states_out is not None
        return ProtectAIBatchOutput(
            injection_probs=probs_out,
            all_layer_pooled_states=states_out,
        )


def _eval_layer_meta_probe(
    states: torch.Tensor,  # [B, L, D]
    probe: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate standardized per-layer linear probes + meta aggregation -> (z_meta, p_meta, p_layers)."""
    mean = probe["mean"]  # [L, D]
    scale = probe["scale"]  # [L, D]
    weights = probe["weights"]  # [L, D]
    bias = probe["bias"]  # [L]
    meta_w = probe["meta_weights"]  # [L]
    meta_b = probe["meta_bias"]

    normed = (states - mean.unsqueeze(0)) / scale.unsqueeze(0)
    z_layers = torch.einsum("bld,ld->bl", normed, weights) + bias.unsqueeze(0)
    z_meta = torch.matmul(z_layers, meta_w) + meta_b
    return z_meta, torch.sigmoid(z_meta), torch.sigmoid(z_layers)


class CodeBERTSecurityClassifier:
    """CodeBERT + calibrated all-layer probe + AST/CWE structural cyber detector."""

    def __init__(
        self,
        model_id: str = DEFAULT_CODEBERT_MODEL,
        *,
        device: str = "auto",
        local_files_only: bool = True,
        max_length: int = 192,
        probe_bundle: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_id = model_id
        self.device = resolve_portable_device(device)
        self.max_length = int(max_length)

        cfg = _load_hf_with_fallback(AutoConfig.from_pretrained, model_id, local_files_only=local_files_only)
        cfg.output_hidden_states = True

        self.tokenizer = _load_hf_with_fallback(
            AutoTokenizer.from_pretrained, model_id, local_files_only=local_files_only
        )
        self.model = _load_hf_with_fallback(
            AutoModelForSequenceClassification.from_pretrained,
            model_id,
            config=cfg,
            local_files_only=local_files_only,
        )
        self.model = self.model.to(device=self.device, dtype=torch.float32).eval()
        self.probe: Optional[Dict[str, torch.Tensor]] = None
        if probe_bundle is not None and "stage_b_codebert" in probe_bundle:
            self.probe = probe_bundle["stage_b_codebert"]
        elif DEFAULT_CCPP_PROBES_PATH.is_file():
            loaded = torch.load(DEFAULT_CCPP_PROBES_PATH, map_location="cpu", weights_only=True)
            self.probe = loaded.get("stage_b_codebert")

    def predict_batch(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        max_length: Optional[int] = None,
    ) -> CodeBERTBatchOutput:
        n = len(texts)
        if n == 0:
            return CodeBERTBatchOutput(
                insecure_probs=torch.empty(0, dtype=torch.float32),
                raw_margins=torch.empty(0, dtype=torch.float32),
                calibrated_c_margin_probs=torch.empty(0, dtype=torch.float32),
                structural_signals=[],
                all_layer_pooled_states=torch.empty((0, 13, 768), dtype=torch.float32),
            )

        signals = [extract_code_security_signals(t) for t in texts]
        cap = int(max_length or self.max_length)
        batches = _iter_length_buckets(self.tokenizer, texts, max_length=cap, batch_size=batch_size)
        margins = torch.empty((n,), dtype=torch.float32)
        states_out: Optional[torch.Tensor] = None

        with torch.no_grad():
            for idxs, ids_t, att_t in batches:
                ids_dev = ids_t.to(self.device)
                att_dev = att_t.to(self.device)
                out = self.model(input_ids=ids_dev, attention_mask=att_dev, output_hidden_states=True)
                mean_p, _ = _masked_layer_pool(out.hidden_states, att_dev, include_max=False)
                if states_out is None:
                    states_out = torch.empty((n, mean_p.shape[1], mean_p.shape[2]), dtype=torch.float32)
                idx_t = torch.tensor(idxs, dtype=torch.long)
                logits_f = out.logits.to(dtype=torch.float32)
                margins[idx_t] = (logits_f[:, 1] - logits_f[:, 0]).detach().cpu()
                states_out[idx_t] = mean_p.detach().cpu()

        assert states_out is not None
        cal_c_probs = torch.sigmoid((margins - 3.2) / 1.0)
        if self.probe is not None:
            _, probe_probs, _ = _eval_layer_meta_probe(states_out, self.probe)
        else:
            probe_probs = cal_c_probs

        final_scores = torch.empty((n,), dtype=torch.float32)
        for i, sig in enumerate(signals):
            p_probe = float(probe_probs[i].item())
            s_struct = sig.structural_risk
            if sig.is_c_like and not sig.is_python:
                if s_struct == 0.0 and not _SAFE_C_OR_PURE_RE.search(texts[i]):
                    # Clean pure/bounded C (e.g. arithmetic or snprintf without raw alloc/unsafe sinks)
                    score = 0.30 * p_probe
                else:
                    score = max(s_struct, 0.72 * p_probe + 0.35 * s_struct)
            elif sig.is_python:
                score = max(s_struct, 0.55 * p_probe + 0.55 * s_struct)
            else:
                score = max(sig.mitre_ttp_risk, 0.78 * p_probe + 0.38 * s_struct)
            final_scores[i] = float(max(0.0, min(1.0, score)))

        return CodeBERTBatchOutput(
            insecure_probs=final_scores,
            raw_margins=margins,
            calibrated_c_margin_probs=cal_c_probs,
            structural_signals=signals,
            all_layer_pooled_states=states_out,
        )


def _eval_mahalanobis_pack(
    layer_states: torch.Tensor,  # [B, L_sub, D]
    pack: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Evaluate regularized PCA-Woodbury Mahalanobis distance across selected layers."""
    mu = pack["mu"]  # [L_sub, D]
    std = pack["std"]  # [L_sub, D]
    uk = pack["Uk"]  # [L_sub, D, k]
    inv_lk = pack["inv_lk"]  # [L_sub, k]
    inv_lp = pack["inv_lp"]  # [L_sub]
    median = pack["median"]  # [L_sub]

    xs = (layer_states - mu.unsqueeze(0)) / std.unsqueeze(0)  # [B, L_sub, D]
    proj = torch.einsum("bld,ldk->blk", xs, uk)  # [B, L_sub, k]
    d_in = (proj.square() * inv_lk.unsqueeze(0)).sum(dim=-1)  # [B, L_sub]
    recon = torch.einsum("blk,ldk->bld", proj, uk)
    d_out = (xs - recon).square().sum(dim=-1) * inv_lp.unsqueeze(0)  # [B, L_sub]
    dim = float(layer_states.shape[-1])
    dists = torch.sqrt(((d_in + d_out) / dim).clamp(min=1e-12))
    normed = dists / median.unsqueeze(0).clamp(min=1e-4)
    return normed.mean(dim=1)


class _EncoderStackCache:
    """Thread-safe lazy singleton pool so multiple firewall objects share MPS weights."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._modernbert: Dict[Tuple[str, bool], ModernBERTAllLayerExtractor] = {}
        self._nemo: Dict[Tuple[str, bool], ZeroCudaNeMoDomainClassifier] = {}
        self._protectai: Dict[Tuple[str, bool], ProtectAIPromptInjectionClassifier] = {}
        self._codebert: Dict[Tuple[str, bool, str], CodeBERTSecurityClassifier] = {}

    def get_modernbert(self, device: str, local_files_only: bool) -> ModernBERTAllLayerExtractor:
        dev_key = str(resolve_portable_device(device))
        key = (dev_key, bool(local_files_only))
        with self._lock:
            if key not in self._modernbert:
                self._modernbert[key] = ModernBERTAllLayerExtractor(
                    device=dev_key, local_files_only=local_files_only
                )
            return self._modernbert[key]

    def get_nemo(self, device: str, local_files_only: bool) -> ZeroCudaNeMoDomainClassifier:
        dev_key = str(resolve_portable_device(device))
        key = (dev_key, bool(local_files_only))
        with self._lock:
            if key not in self._nemo:
                self._nemo[key] = ZeroCudaNeMoDomainClassifier(
                    device=dev_key, local_files_only=local_files_only
                )
            return self._nemo[key]

    def get_protectai(self, device: str, local_files_only: bool) -> ProtectAIPromptInjectionClassifier:
        dev_key = str(resolve_portable_device(device))
        key = (dev_key, bool(local_files_only))
        with self._lock:
            if key not in self._protectai:
                self._protectai[key] = ProtectAIPromptInjectionClassifier(
                    device=dev_key, local_files_only=local_files_only
                )
            return self._protectai[key]

    def get_codebert(
        self,
        device: str,
        local_files_only: bool,
        probe_bundle: Optional[Dict[str, Any]],
        probe_tag: str,
    ) -> CodeBERTSecurityClassifier:
        dev_key = str(resolve_portable_device(device))
        key = (dev_key, bool(local_files_only), probe_tag)
        with self._lock:
            if key not in self._codebert:
                self._codebert[key] = CodeBERTSecurityClassifier(
                    device=dev_key,
                    local_files_only=local_files_only,
                    probe_bundle=probe_bundle,
                )
            return self._codebert[key]


_SHARED_POOL = _EncoderStackCache()


class AnthropicCCPlusPlusCascade:
    """Two-stage Constitutional Classifiers++ activation-probe + Mahalanobis OOD + deep ensemble."""

    def __init__(
        self,
        *,
        device: str = "auto",
        probe_weights_path: Optional[Union[str, Path]] = None,
        local_files_only: bool = True,
        tau_low: Optional[float] = None,
        tau_high: Optional[float] = None,
        tau_ood: Optional[float] = None,
        preload_stage_b: bool = False,
    ) -> None:
        self.device = resolve_portable_device(device)
        self.local_files_only = bool(local_files_only)
        self.probe_path = Path(probe_weights_path) if probe_weights_path is not None else DEFAULT_CCPP_PROBES_PATH
        if not self.probe_path.is_file():
            raise FileNotFoundError(
                f"CC++ probe weights not found at {self.probe_path}. "
                "Generate ccpp_probes.pt before initializing AnthropicCCPlusPlusCascade."
            )
        _st_path = Path(self.probe_path).with_suffix(".safetensors")
        if _st_path.exists():
            import json as _json
            from safetensors import safe_open as _safe_open
            from safetensors.torch import load_file as _load_st
            _flat = _load_st(str(_st_path), device="cpu")
            _unflat: Dict[str, Any] = {}
            def _insert_nested(root: Dict[str, Any], key: str, val: Any) -> None:
                parts = key.split("__")
                cur = root
                for part in parts[:-1]:
                    if part not in cur or not isinstance(cur[part], dict):
                        cur[part] = {}
                    cur = cur[part]
                cur[parts[-1]] = val
            for _k, _v in _flat.items():
                _insert_nested(_unflat, _k, _v)
            with _safe_open(str(_st_path), framework="pt", device="cpu") as _f:
                _meta = _f.metadata() or {}
            if "ojasx_non_tensors" in _meta:
                for _k, _v in _json.loads(_meta["ojasx_non_tensors"]).items():
                    _insert_nested(_unflat, _k, _v)
            self.probes: Dict[str, Any] = _unflat
        else:
            self.probes: Dict[str, Any] = torch.load(self.probe_path, map_location="cpu", weights_only=True)
        defaults = self.probes.get("cascade_defaults", {})
        self.tau_low = float(tau_low if tau_low is not None else min(0.20, float(defaults.get("tau_low", 0.20))))
        self.tau_high = float(tau_high if tau_high is not None else max(0.95, float(defaults.get("tau_high", 0.95))))
        base_ood = float(defaults.get("tau_ood", 1.35))
        self.tau_ood = float(tau_ood if tau_ood is not None else base_ood * 1.08)

        # Stage A encoder loaded lazily on first use (or immediately if preload)
        self._mb: Optional[ModernBERTAllLayerExtractor] = None
        self._pi: Optional[ProtectAIPromptInjectionClassifier] = None
        self._nm: Optional[ZeroCudaNeMoDomainClassifier] = None
        self._cb: Optional[CodeBERTSecurityClassifier] = None
        if preload_stage_b:
            self._ensure_stage_a()
            self._ensure_stage_b(need_codebert=True)

    def _ensure_stage_a(self) -> ModernBERTAllLayerExtractor:
        if self._mb is None:
            self._mb = _SHARED_POOL.get_modernbert(str(self.device), self.local_files_only)
        return self._mb

    def _ensure_stage_b(
        self, *, need_codebert: bool = False
    ) -> Tuple[ProtectAIPromptInjectionClassifier, ZeroCudaNeMoDomainClassifier, Optional[CodeBERTSecurityClassifier]]:
        if self._pi is None:
            self._pi = _SHARED_POOL.get_protectai(str(self.device), self.local_files_only)
        if self._nm is None:
            self._nm = _SHARED_POOL.get_nemo(str(self.device), self.local_files_only)
        if need_codebert and self._cb is None:
            self._cb = _SHARED_POOL.get_codebert(
                str(self.device),
                self.local_files_only,
                self.probes,
                str(self.probe_path),
            )
        return self._pi, self._nm, self._cb

    def _compute_mahalanobis_ood(self, mb_means: torch.Tensor) -> torch.Tensor:
        mahal_cfg = self.probes["stage_a_mahalanobis"]
        layers = list(mahal_cfg["layers"])
        sub = mb_means[:, layers, :]
        d_nl = _eval_mahalanobis_pack(sub, mahal_cfg["nl_manifold"])
        d_code = _eval_mahalanobis_pack(sub, mahal_cfg["code_manifold"])
        return torch.minimum(d_nl, d_code)

    def evaluate_batch(
        self,
        texts: Sequence[str],
        router_scores: Sequence[Dict[str, Any]],
        *,
        is_mcqa_flags: Optional[Sequence[bool]] = None,
        profile_value: str = "public_general",
        exchange_pairs: Optional[Sequence[Tuple[str, str, Tuple[str, ...]]]] = None,
        router_score_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
        active_indices: Optional[Sequence[int]] = None,
    ) -> List[CCPlusPlusSampleTrace]:
        """Run Stage A all-layer screening + 3-way routing + Stage B Deep Ensemble on escalated items."""
        n = len(texts)
        if n == 0:
            return []

        idx_list = list(range(n)) if active_indices is None else list(active_indices)
        if not idx_list:
            return [
                CCPlusPlusSampleTrace(
                    sample_index=i,
                    cascade_decision="PASS",
                    cascade_escalated=False,
                    stage_a_probe_score=0.0,
                    stage_a_nl_score=0.0,
                    stage_a_cyber_score=0.0,
                    stage_a_dual_use_score=0.0,
                    mahalanobis_ood=0.0,
                    ood_elevated=False,
                    stage_b_nl_score=0.0,
                    protectai_pi_prob=0.0,
                    codebert_risk_score=0.0,
                    nemo_domain_label="",
                    nemo_domain_probs={},
                    dual_use_domain="none",
                    dual_use_prob=0.0,
                    exchange_risk_score=0.0,
                    final_harm_score=float(router_scores[i]["harm_score"]),
                    reasons=("ccpp:tier0_shortcut",),
                )
                for i in range(n)
            ]

        mb = self._ensure_stage_a()
        sub_texts = [texts[i] for i in idx_list]
        mb_out = mb.extract_batch(sub_texts)

        # Stage A vectorized all-layer probes
        z_sa, p_sa, _ = _eval_layer_meta_probe(mb_out.layer_mean_states, self.probes["stage_a_harm"])
        z_mb_nl, p_mb_nl, _ = _eval_layer_meta_probe(mb_out.layer_mean_states, self.probes["stage_a_harm_nl"])
        _, p_cyber_a, _ = _eval_layer_meta_probe(mb_out.layer_mean_states, self.probes["stage_a_cyber"])

        du_a_cfg = self.probes["stage_a_dual_use"]
        du_l = int(du_a_cfg["layer"])
        du_feat = (mb_out.layer_mean_states[:, du_l, :] - du_a_cfg["mean"]) / du_a_cfg["scale"]
        p_dual_a = torch.sigmoid(torch.matmul(du_feat, du_a_cfg["weights"]) + du_a_cfg["bias"])

        ood_dists = self._compute_mahalanobis_ood(mb_out.layer_mean_states)
        code_sigs = [extract_code_security_signals(t) for t in sub_texts]

        is_sci = profile_value == "scientific_modular"
        is_code_ent = profile_value == "code_enterprise"
        mcqa_flags = is_mcqa_flags if is_mcqa_flags is not None else [False] * n

        # Stage B routing per active item
        stage_decisions: List[str] = []
        escalated_local: List[int] = []
        need_codebert_local: List[int] = []
        need_nl_stage_b_local: List[int] = []

        for loc_i, orig_i in enumerate(idx_list):
            rs = router_scores[orig_i]
            sig = code_sigs[loc_i]
            s_p = float(p_sa[loc_i].item())
            s_cy = float(p_cyber_a[loc_i].item())
            s_du = float(p_dual_a[loc_i].item())
            ood_v = float(ood_dists[loc_i].item())
            has_turns = bool(exchange_pairs and len(exchange_pairs[orig_i][2]) >= 2)
            has_resp = bool(exchange_pairs and exchange_pairs[orig_i][1])

            fast_refuse = (
                (s_p > self.tau_high)
                and (rs["harm_score"] >= 0.40 or rs["has_actionable_verb"])
                and (not rs["has_benign_framing"])
                and (not (is_sci and (s_du >= 0.30 or mcqa_flags[orig_i])))
                and (not sig.is_code)
                and (not has_turns)
            )
            if fast_refuse:
                stage_decisions.append("REFUSE")
                continue

            should_escalate = (
                (s_p >= self.tau_low)
                or (ood_v > self.tau_ood)
                or sig.is_code
                or (sig.structural_risk > 0.0)
                or (s_cy >= 0.30)
                or (rs["harm_score"] >= 0.35)
                or (rs["dual_use_score"] >= 0.22)
                or rs["absolute_redline"]
                or (is_sci and (s_du >= 0.25 or bool(mcqa_flags[orig_i])))
                or has_turns
                or has_resp
            )
            if should_escalate:
                stage_decisions.append("ESCALATE")
                escalated_local.append(loc_i)
                is_pure_code = sig.is_code and not sig.mitre_ttp_risk and not is_sci and not mcqa_flags[orig_i] and not has_turns
                if sig.is_code or sig.structural_risk > 0.0 or s_cy >= 0.25 or is_code_ent:
                    need_codebert_local.append(loc_i)
                if not is_pure_code or has_resp or s_p >= 0.55 or rs["harm_score"] >= 0.45:
                    need_nl_stage_b_local.append(loc_i)
            else:
                stage_decisions.append("PASS")

        # Run Stage B Deep Ensemble ONLY on escalated items
        stage_b_nl = {loc_i: float(p_mb_nl[loc_i].item()) for loc_i in range(len(sub_texts))}
        pi_probs_map: Dict[int, float] = {}
        cb_scores_map: Dict[int, float] = {}
        nemo_label_map: Dict[int, str] = {}
        nemo_probs_map: Dict[int, Dict[str, float]] = {}
        du_domain_map: Dict[int, str] = {}
        du_prob_map: Dict[int, float] = {}
        exchange_risk_map: Dict[int, float] = {}

        if escalated_local:
            nl_set = set(need_nl_stage_b_local)
            nl_esc_local = [loc_i for loc_i in escalated_local if loc_i in nl_set]

            # Fast path: run NL Stage B (ProtectAI + NeMo) only on items needing NL / dual-use evaluation
            if nl_esc_local:
                esc_texts = [sub_texts[loc_i] for loc_i in nl_esc_local]
                pi_cls, nm_cls, cb_cls = self._ensure_stage_b(need_codebert=bool(need_codebert_local))

                pi_out = pi_cls.predict_batch(esc_texts)
                nm_out = nm_cls.predict_batch(esc_texts)

                z_pi, _, _ = _eval_layer_meta_probe(pi_out.all_layer_pooled_states, self.probes["stage_b_protectai"])
                z_nm, _, _ = _eval_layer_meta_probe(nm_out.all_layer_pooled_states, self.probes["stage_b_nemo"])

                meta_cfg = self.probes["stage_b_meta"]
                x_meta_rows = []
                for j, loc_i in enumerate(nl_esc_local):
                    orig_i = idx_list[loc_i]
                    rs = router_scores[orig_i]
                    ood_exc = max(0.0, float(ood_dists[loc_i].item()) - self.tau_ood)
                    x_meta_rows.append(
                        [
                            float(z_mb_nl[loc_i].item()),
                            float(z_pi[j].item()),
                            float(z_nm[j].item()),
                            float(rs["harm_score"]),
                            float(rs["ml_harm_prob"]),
                            1.0 if rs["has_benign_framing"] else 0.0,
                            1.0 if rs["has_actionable_verb"] else 0.0,
                            float(pi_out.injection_probs[j].item()),
                            ood_exc,
                        ]
                    )
                x_meta_t = torch.tensor(x_meta_rows, dtype=torch.float32)
                x_meta_std = (x_meta_t - meta_cfg["mean"]) / meta_cfg["scale"]
                meta_probs = torch.sigmoid(torch.matmul(x_meta_std, meta_cfg["weights"]) + meta_cfg["bias"])

                du_b_cfg = self.probes["stage_b_dual_use"]
                mb_l16 = mb_out.layer_mean_states[nl_esc_local, int(du_b_cfg["mb_layer"]), :]
                nm_l8 = nm_out.all_layer_pooled_states[:, int(du_b_cfg["nemo_layer"]), :]
                x_du = torch.cat([mb_l16, nm_l8, nm_out.domain_probs], dim=1)
                x_du_std = (x_du - du_b_cfg["mean"]) / du_b_cfg["scale"]
                du_logits = torch.matmul(x_du_std, du_b_cfg["weights"].T) + du_b_cfg["bias"]
                du_probs_t = torch.softmax(du_logits, dim=-1)
                du_classes: List[str] = list(du_b_cfg["classes"])

                for j, loc_i in enumerate(nl_esc_local):
                    pi_p = float(pi_out.injection_probs[j].item())
                    pi_probs_map[loc_i] = pi_p
                    nl_s = float(meta_probs[j].item())
                    if pi_p >= 0.75:
                        nl_s = max(nl_s, 0.55 + 0.42 * pi_p)
                    stage_b_nl[loc_i] = nl_s

                    lbl = nm_out.domain_labels[j]
                    nemo_label_map[loc_i] = lbl
                    row_probs = nm_out.domain_probs[j].tolist()
                    top_sorted = sorted(
                        ((nm_cls.id2label[idx_c], float(val)) for idx_c, val in enumerate(row_probs) if val >= 0.02),
                        key=lambda kv: kv[1],
                        reverse=True,
                    )[:6]
                    nemo_probs_map[loc_i] = dict(top_sorted)

                    du_row = du_probs_t[j].tolist()
                    best_du_dom = "none"
                    best_du_p = 0.0
                    for c_name, c_p in zip(du_classes, du_row):
                        if c_name != "none" and c_p > best_du_p:
                            best_du_p = float(c_p)
                            best_du_dom = c_name
                    du_domain_map[loc_i] = best_du_dom
                    du_prob_map[loc_i] = best_du_p

            # Run CodeBERT on code / cyber escalated items
            cb_set = set(need_codebert_local)
            cb_subset_local = [
                loc_i
                for loc_i in escalated_local
                if (
                    loc_i in cb_set
                    or nemo_label_map.get(loc_i) in ("Computers_and_Electronics", "Internet_and_Telecom")
                )
            ]
            if cb_subset_local:
                if self._cb is None:
                    self._cb = _SHARED_POOL.get_codebert(
                        str(self.device),
                        self.local_files_only,
                        self.probes,
                        str(self.probe_path),
                    )
                cb_texts = [sub_texts[loc_i] for loc_i in cb_subset_local]
                cb_out = self._cb.predict_batch(cb_texts)
                for k_i, loc_i in enumerate(cb_subset_local):
                    cb_scores_map[loc_i] = float(cb_out.insecure_probs[k_i].item())

            # Exchange Context (prompt vs response / multi-turn escalation) on escalated items
            if exchange_pairs is not None and router_score_fn is not None:
                ex_checks: List[Tuple[int, str]] = []
                for loc_i in escalated_local:
                    orig_i = idx_list[loc_i]
                    p_txt, r_txt, turns = exchange_pairs[orig_i]
                    if r_txt and p_txt:
                        ex_checks.append((loc_i, r_txt))
                    if len(turns) >= 2:
                        ex_checks.append((loc_i, turns[-1]))
                        if len(turns) >= 3:
                            ex_checks.append((loc_i, "\n".join(turns[-2:])))
                if ex_checks:
                    if self._pi is None:
                        self._pi = _SHARED_POOL.get_protectai(str(self.device), self.local_files_only)
                    ex_mb = mb.extract_batch([txt for _, txt in ex_checks])
                    _, ex_mb_p, _ = _eval_layer_meta_probe(ex_mb.layer_mean_states, self.probes["stage_a_harm"])
                    ex_pi = self._pi.predict_batch([txt for _, txt in ex_checks])
                    for k_i, (loc_i, ex_txt) in enumerate(ex_checks):
                        ex_heur = router_score_fn(ex_txt)["harm_score"]
                        ex_code = extract_code_security_signals(ex_txt).structural_risk
                        ex_score = max(
                            float(ex_heur),
                            float(ex_code),
                            0.55 * float(ex_mb_p[k_i].item()) + 0.45 * float(ex_pi.injection_probs[k_i].item()),
                            float(ex_pi.injection_probs[k_i].item()),
                        )
                        exchange_risk_map[loc_i] = max(exchange_risk_map.get(loc_i, 0.0), ex_score)

        # Assemble per-sample CCPlusPlusSampleTrace aligned to full batch [0..N-1]
        traces: List[Optional[CCPlusPlusSampleTrace]] = [None] * n
        for loc_i, orig_i in enumerate(idx_list):
            dec = stage_decisions[loc_i]
            escalated = dec == "ESCALATE"
            s_p = float(p_sa[loc_i].item())
            s_nl_a = float(p_mb_nl[loc_i].item())
            s_cy_a = float(p_cyber_a[loc_i].item())
            s_du_a = float(p_dual_a[loc_i].item())
            ood_v = float(ood_dists[loc_i].item())
            ood_elev = ood_v > self.tau_ood
            sb_nl = float(stage_b_nl.get(loc_i, s_nl_a))
            pi_p = float(pi_probs_map.get(loc_i, 0.0))
            cb_s = float(cb_scores_map.get(loc_i, 0.0))
            ex_s = float(exchange_risk_map.get(loc_i, 0.0))
            nm_lbl = nemo_label_map.get(loc_i, "")
            nm_p = nemo_probs_map.get(loc_i, {})
            du_dom = du_domain_map.get(loc_i, "none")
            du_p = float(du_prob_map.get(loc_i, 0.0))

            if dec == "PASS":
                final_score = min(float(router_scores[orig_i]["harm_score"]), s_p)
            elif dec == "REFUSE":
                final_score = max(s_p, float(router_scores[orig_i]["harm_score"]))
            else:
                final_score = max(sb_nl, cb_s, ex_s)

            reason_tags: List[str] = []
            if dec == "PASS":
                reason_tags.append("ccpp:stage_a_fast_pass")
            elif dec == "REFUSE":
                reason_tags.append("ccpp:stage_a_fast_refuse")
                reason_tags.append(f"neural:modernbert_probe={s_p:.2f}")
            else:
                reason_tags.append("ccpp:escalated_stage_b")
                if ood_elev:
                    reason_tags.append(f"ccpp:ood_elevated={ood_v:.2f}")
                reason_tags.append(f"neural:stage_b_nl={sb_nl:.2f}")
                if pi_p >= 0.45:
                    reason_tags.append(f"neural:protectai_pi={pi_p:.2f}")
                if cb_s >= 0.45:
                    reason_tags.append(f"neural:codebert_sec={cb_s:.2f}")
                if nm_lbl:
                    reason_tags.append(f"neural:nemo_domain={nm_lbl}")
                if ex_s >= 0.50:
                    reason_tags.append(f"neural:exchange_ctx={ex_s:.2f}")

            traces[orig_i] = CCPlusPlusSampleTrace(
                sample_index=orig_i,
                cascade_decision=dec,
                cascade_escalated=escalated,
                stage_a_probe_score=s_p,
                stage_a_nl_score=s_nl_a,
                stage_a_cyber_score=s_cy_a,
                stage_a_dual_use_score=s_du_a,
                mahalanobis_ood=ood_v,
                ood_elevated=ood_elev,
                stage_b_nl_score=sb_nl,
                protectai_pi_prob=pi_p,
                codebert_risk_score=cb_s,
                nemo_domain_label=nm_lbl,
                nemo_domain_probs=nm_p,
                dual_use_domain=du_dom,
                dual_use_prob=du_p,
                exchange_risk_score=ex_s,
                final_harm_score=float(max(0.0, min(1.0, final_score))),
                reasons=tuple(reason_tags),
            )

        out_traces: List[CCPlusPlusSampleTrace] = []
        for i, tr in enumerate(traces):
            if tr is None:
                out_traces.append(
                    CCPlusPlusSampleTrace(
                        sample_index=i,
                        cascade_decision="PASS",
                        cascade_escalated=False,
                        stage_a_probe_score=0.0,
                        stage_a_nl_score=0.0,
                        stage_a_cyber_score=0.0,
                        stage_a_dual_use_score=0.0,
                        mahalanobis_ood=0.0,
                        ood_elevated=False,
                        stage_b_nl_score=0.0,
                        protectai_pi_prob=0.0,
                        codebert_risk_score=0.0,
                        nemo_domain_label="",
                        nemo_domain_probs={},
                        dual_use_domain="none",
                        dual_use_prob=0.0,
                        exchange_risk_score=0.0,
                        final_harm_score=float(router_scores[i]["harm_score"]),
                        reasons=("ccpp:tier0_shortcut",),
                    )
                )
            else:
                out_traces.append(tr)
        return out_traces


__all__ = [
    "AnthropicCCPlusPlusCascade",
    "CCPlusPlusSampleTrace",
    "CodeBERTBatchOutput",
    "CodeBERTSecurityClassifier",
    "CodeSecuritySignals",
    "DEFAULT_CCPP_PROBES_PATH",
    "DEFAULT_CODEBERT_MODEL",
    "DEFAULT_MODERNBERT_MODEL",
    "DEFAULT_NEMO_DOMAIN_MODEL",
    "DEFAULT_PROTECTAI_PI_MODEL",
    "ModernBERTAllLayerExtractor",
    "ModernBERTBatchOutput",
    "NeMoDomainBatchOutput",
    "ProtectAIBatchOutput",
    "ProtectAIPromptInjectionClassifier",
    "ZeroCudaNeMoDomainClassifier",
    "extract_code_security_signals",
]
