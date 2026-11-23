"""Portable PII and secret credential detector and redactor for OjasX-Aegis."""

from __future__ import annotations

import hashlib
import ipaddress
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Pattern, Tuple


@dataclass(frozen=True)
class PIIFinding:
    """Metadata describing a detected and redacted PII or secret span."""

    category: str
    placeholder: str
    start: int
    end: int
    severity: str  # "secret" or "pii"
    value_sha256_prefix: str


def luhn_is_valid(digits: str) -> bool:
    """Validate a candidate payment card digit sequence with Luhn mod-10 checksum."""
    if not digits.isdigit() or not (13 <= len(digits) <= 19):
        return False
    if len(set(digits)) <= 1:
        return False
    total = 0
    reverse_digits = digits[::-1]
    for idx, ch in enumerate(reverse_digits):
        val = ord(ch) - 48
        if idx % 2 == 1:
            val *= 2
            if val > 9:
                val -= 9
        total += val
    return (total % 10) == 0


@dataclass(frozen=True)
class _PatternRule:
    category: str
    placeholder: str
    severity: str
    priority: int
    pattern: Pattern[str]
    group: int = 0
    validator: Optional[Callable[[str], bool]] = None


def _validate_ipv4(text: str) -> bool:
    try:
        ipaddress.IPv4Address(text)
        return True
    except ValueError:
        return False


def _validate_ipv6(text: str) -> bool:
    if text.count(":") < 2:
        return False
    try:
        ipaddress.IPv6Address(text)
        return True
    except ValueError:
        return False


def _validate_card(text: str) -> bool:
    digits = re.sub(r"[ -]", "", text)
    return luhn_is_valid(digits)


def _validate_phone(text: str) -> bool:
    digits = re.sub(r"\D", "", text)
    return 8 <= len(digits) <= 15


_RULES: Tuple[_PatternRule, ...] = (
    _PatternRule(
        category="pem_private_key",
        placeholder="[REDACTED_PRIVATE_KEY]",
        severity="secret",
        priority=100,
        pattern=re.compile(
            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
            re.MULTILINE,
        ),
    ),
    _PatternRule(
        category="url_credentials",
        placeholder="[REDACTED_URL_CREDENTIALS]",
        severity="secret",
        priority=95,
        pattern=re.compile(
            r"(?:[a-zA-Z][a-zA-Z0-9+.-]*://)([^\s/@:]+:[^\s/@]+)@[A-Za-z0-9._:-]+"
        ),
        group=1,
    ),
    _PatternRule(
        category="aws_access_key",
        placeholder="[REDACTED_AWS_KEY]",
        severity="secret",
        priority=90,
        pattern=re.compile(r"\b(?:AKIA|ASIA|AROA|AIDA)[0-9A-Z]{16}\b"),
    ),
    _PatternRule(
        category="aws_secret_key",
        placeholder="[REDACTED_AWS_SECRET]",
        severity="secret",
        priority=90,
        pattern=re.compile(
            r"(?i)(?:aws_secret_access_key|aws_secret_key|secret_access_key)\s*[:=]\s*['\"]?([A-Za-z0-9/+=]{40})['\"]?"
        ),
        group=1,
    ),
    _PatternRule(
        category="github_token",
        placeholder="[REDACTED_GITHUB_TOKEN]",
        severity="secret",
        priority=90,
        pattern=re.compile(r"\b(?:(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    ),
    _PatternRule(
        category="openai_api_key",
        placeholder="[REDACTED_API_KEY]",
        severity="secret",
        priority=90,
        pattern=re.compile(r"\bsk-(?:proj-|svcacct-|admin-|ant-)?[A-Za-z0-9_-]{20,}\b"),
    ),
    _PatternRule(
        category="gcp_api_key",
        placeholder="[REDACTED_GCP_KEY]",
        severity="secret",
        priority=90,
        pattern=re.compile(r"\bAIza[0-9A-Za-z_-]{35,}\b"),
    ),
    _PatternRule(
        category="hf_token",
        placeholder="[REDACTED_HF_TOKEN]",
        severity="secret",
        priority=90,
        pattern=re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    ),
    _PatternRule(
        category="slack_token",
        placeholder="[REDACTED_SLACK_TOKEN]",
        severity="secret",
        priority=90,
        pattern=re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,}\b"),
    ),
    _PatternRule(
        category="stripe_key",
        placeholder="[REDACTED_STRIPE_KEY]",
        severity="secret",
        priority=90,
        pattern=re.compile(r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}\b"),
    ),
    _PatternRule(
        category="payment_card",
        placeholder="[REDACTED_CARD]",
        severity="pii",
        priority=80,
        pattern=re.compile(r"(?<![\dA-Za-z])(\d(?:[ -]?\d){12,18})(?![\dA-Za-z])"),
        group=1,
        validator=_validate_card,
    ),
    _PatternRule(
        category="national_tax_id",
        placeholder="[REDACTED_TAX_ID]",
        severity="pii",
        priority=78,
        pattern=re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"),
    ),
    _PatternRule(
        category="national_tax_id",
        placeholder="[REDACTED_TAX_ID]",
        severity="pii",
        priority=76,
        pattern=re.compile(
            r"(?i)\b(?:ssn|social\s+security(?:\s+number)?|ein|tin|tax\s*id|aadhaar)\s*[:=#]?\s*"
            r"(\d{3}[-\s]?\d{2}[-\s]?\d{4}|\d{2}-\d{7}|\d{4}\s?\d{4}\s?\d{4})\b"
        ),
        group=1,
    ),
    _PatternRule(
        category="national_tax_id",
        placeholder="[REDACTED_TAX_ID]",
        severity="pii",
        priority=75,
        pattern=re.compile(r"(?i)\b(?:pan(?:\s+card|\s+no|\s+number)?)\s*[:=#]?\s*([A-Z]{5}\d{4}[A-Z])\b"),
        group=1,
    ),
    _PatternRule(
        category="email",
        placeholder="[REDACTED_EMAIL]",
        severity="pii",
        priority=70,
        pattern=re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ),
    _PatternRule(
        category="phone",
        placeholder="[REDACTED_PHONE]",
        severity="pii",
        priority=65,
        pattern=re.compile(
            r"(?<!\d\.)(?<![\dA-Za-z-])"
            r"(?:\+\d{1,3}[-.\s]?(?:\(\d{1,4}\)[-.\s]?)?(?:\d{2,4}[-.\s]?){2,4}\d{2,4}"
            r"|(?:\(\d{3}\)[-.\s]?|\d{3}[-.\s])\d{3}[-.\s]\d{4})"
            r"(?!\.\d)(?![\dA-Za-z-])"
        ),
        validator=_validate_phone,
    ),
    _PatternRule(
        category="ipv6",
        placeholder="[REDACTED_IP]",
        severity="pii",
        priority=60,
        pattern=re.compile(
            r"(?<![0-9A-Fa-f:])"
            r"(?:"
            r"(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}"
            r"|(?:[0-9A-Fa-f]{1,4}:){1,7}:"
            r"|(?:[0-9A-Fa-f]{1,4}:){1,6}:[0-9A-Fa-f]{1,4}"
            r"|::(?:[0-9A-Fa-f]{1,4}:){0,6}[0-9A-Fa-f]{1,4}"
            r"|::1"
            r")"
            r"(?![0-9A-Fa-f:])"
        ),
        validator=_validate_ipv6,
    ),
    _PatternRule(
        category="ipv4",
        placeholder="[REDACTED_IP]",
        severity="pii",
        priority=58,
        pattern=re.compile(
            r"(?<![\d.vV])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])"
        ),
        validator=_validate_ipv4,
    ),
)


class PIIRedactor:
    """Detects and redacts PII and secret credentials with overlap-safe spans."""

    def __init__(self, extra_rules: Tuple[_PatternRule, ...] = ()) -> None:
        self.rules: Tuple[_PatternRule, ...] = _RULES + tuple(extra_rules)

    def _collect_candidates(self, text: str) -> List[Tuple[int, int, int, _PatternRule, str]]:
        candidates: List[Tuple[int, int, int, _PatternRule, str]] = []
        for rule in self.rules:
            for match in rule.pattern.finditer(text):
                start, end = match.span(rule.group)
                if start < 0 or end <= start:
                    continue
                matched = match.group(rule.group)
                if rule.validator is not None and not rule.validator(matched):
                    continue
                candidates.append((rule.priority, start, end, rule, matched))
        return candidates

    def detect(self, text: str) -> List[PIIFinding]:
        """Return non-overlapping PII/secret findings sorted by start offset."""
        if not text:
            return []

        candidates = self._collect_candidates(text)
        candidates.sort(key=lambda item: (item[0], item[2] - item[1], -item[1]), reverse=True)

        selected: List[PIIFinding] = []
        occupied: List[Tuple[int, int]] = []
        for _, start, end, rule, matched in candidates:
            if any(not (end <= occ_s or start >= occ_e) for occ_s, occ_e in occupied):
                continue
            occupied.append((start, end))
            digest_prefix = hashlib.sha256(matched.encode("utf-8")).hexdigest()[:12]
            selected.append(
                PIIFinding(
                    category=rule.category,
                    placeholder=rule.placeholder,
                    start=start,
                    end=end,
                    severity=rule.severity,
                    value_sha256_prefix=digest_prefix,
                )
            )

        selected.sort(key=lambda f: f.start)
        return selected

    def redact(self, text: str) -> Tuple[str, List[PIIFinding]]:
        """Detect and replace all PII/secrets in text, returning (redacted_text, findings)."""
        if not isinstance(text, str):
            text = str(text)
        findings = self.detect(text)
        if not findings:
            return text, []

        chunks: List[str] = []
        cursor = 0
        for finding in findings:
            chunks.append(text[cursor : finding.start])
            chunks.append(finding.placeholder)
            cursor = finding.end
        chunks.append(text[cursor:])
        return "".join(chunks), findings


__all__ = ["PIIFinding", "PIIRedactor", "luhn_is_valid"]
