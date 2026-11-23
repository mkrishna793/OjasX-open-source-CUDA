"""Portable multimodal safety extractor: Apple Vision OCR, PNG/LSB forensics, and audio inspection."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from scipy.stats import chi2 as scipy_chi2

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PRIME_KERNEL_DIR = _REPO_ROOT / ".prime-kernel"

_SWIFT_OCR_SOURCE = r"""import Foundation
import Vision
import AppKit

struct OcrRecord: Codable {
    let path: String
    let lines: [String]
    let text: String
    let error: String?
}

func recognize(path: String) -> OcrRecord {
    let url = URL(fileURLWithPath: path)
    guard let img = NSImage(contentsOf: url),
          let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        return OcrRecord(path: path, lines: [], text: "", error: "failed_to_load_image")
    }

    let req = VNRecognizeTextRequest()
    req.recognitionLevel = .accurate
    req.usesLanguageCorrection = true

    let handler = VNImageRequestHandler(cgImage: cg, options: [:])
    do {
        try handler.perform([req])
        let lines = (req.results ?? []).compactMap { $0.topCandidates(1).first?.string }
        return OcrRecord(path: path, lines: lines, text: lines.joined(separator: "\n"), error: nil)
    } catch {
        return OcrRecord(path: path, lines: [], text: "", error: "\(error)")
    }
}

let encoder = JSONEncoder()
for arg in CommandLine.arguments.dropFirst() {
    autoreleasepool {
        let rec = recognize(path: arg)
        if let data = try? encoder.encode(rec),
           let line = String(data: data, encoding: .utf8) {
            print(line)
        }
    }
}
"""

# Matches standalone blank numbered list markers produced by Vision OCR (e.g. "1.", "2 •", "3°", "2 )")
_BLANK_LIST_ITEM_RE = re.compile(r"^\s*([1-9])\s*[.\u2022\u00b0:)\-]?\s*$")
_INLINE_TRAILING_LIST_RE = re.compile(r"(?:^|\n)\s*1\s*[.\u2022\u00b0:)]\s*\n\s*2\s*[.\u2022\u00b0:)]\s*(?:\n\s*3\s*[.\u2022\u00b0:)]\s*)?$", re.MULTILINE)


@dataclass
class ImageForensics:
    """PNG metadata, LSB steganography, and typographic layout inspection details."""

    png_metadata: Dict[str, str] = field(default_factory=dict)
    lsb_entropy: float = 0.0
    bp1_entropy: float = 0.0
    pov_chi2_p_value: float = 0.0
    pov_active_pairs: int = 0
    stego_detected: bool = False
    stego_reason: str = ""
    stroke_contrast_ratio: float = 0.0
    typographic_list_attack: bool = False
    blank_list_numbers: Tuple[int, ...] = ()


@dataclass
class MultimodalExtractionResult:
    """Combined OCR text and visual/metadata forensic inspection output."""

    extracted_text: str
    ocr_lines: List[str]
    ocr_backend: str
    forensics: ImageForensics
    combined_inspection_text: str


@dataclass
class AudioExtractionResult:
    """Extracted transcript and metadata from an audio sample."""

    extracted_text: str
    source_kind: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def _bitplane_entropy(plane: np.ndarray) -> float:
    p1 = float(np.mean(plane))
    p0 = 1.0 - p1
    ent = 0.0
    if p0 > 1e-12:
        ent -= p0 * math.log2(p0)
    if p1 > 1e-12:
        ent -= p1 * math.log2(p1)
    return float(ent)


def extract_png_text_chunks(path: Path) -> Dict[str, str]:
    """Parse tEXt, zTXt, and iTXt metadata chunks directly from PNG bytes + PIL info."""
    chunks: Dict[str, str] = {}
    try:
        with Image.open(path) as im:
            for k, v in (im.info or {}).items():
                if isinstance(v, (str, bytes)):
                    val = v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v
                    if val.strip():
                        chunks[str(k)] = val
    except Exception:
        pass

    try:
        raw = path.read_bytes()
    except OSError:
        return chunks

    if len(raw) < 8 or raw[:8] != b"\x89PNG\r\n\x1a\n":
        return chunks

    pos = 8
    while pos + 12 <= len(raw):
        length = struct.unpack(">I", raw[pos : pos + 4])[0]
        ctype = raw[pos + 4 : pos + 8]
        data_start = pos + 8
        data_end = data_start + length
        if data_end + 4 > len(raw):
            break
        payload = raw[data_start:data_end]
        pos = data_end + 4

        if ctype == b"tEXt" and b"\x00" in payload:
            key_b, val_b = payload.split(b"\x00", 1)
            key = key_b.decode("latin-1", errors="replace")
            val = val_b.decode("latin-1", errors="replace")
            if val.strip():
                chunks[key] = val
        elif ctype == b"zTXt" and b"\x00" in payload:
            key_b, rest = payload.split(b"\x00", 1)
            if len(rest) >= 2 and rest[0] == 0:
                try:
                    val = zlib.decompress(rest[1:]).decode("latin-1", errors="replace")
                    if val.strip():
                        chunks[key_b.decode("latin-1", errors="replace")] = val
                except zlib.error:
                    pass
        elif ctype == b"iTXt" and b"\x00" in payload:
            parts = payload.split(b"\x00", 5)
            if len(parts) >= 6:
                key = parts[0].decode("utf-8", errors="replace")
                comp_flag = parts[1][:1]
                text_bytes = parts[5]
                try:
                    if comp_flag == b"\x01":
                        text_bytes = zlib.decompress(text_bytes)
                    val = text_bytes.decode("utf-8", errors="replace")
                    if val.strip():
                        chunks[key] = val
                except Exception:
                    pass
        elif ctype == b"IEND":
            break

    return chunks


def analyze_lsb_steganography(rgb_array: np.ndarray) -> Tuple[float, float, float, int, bool, str]:
    """Run Pairs-of-Values chi-square and bit-plane entropy anomaly analysis on RGB channels."""
    if rgb_array.ndim == 2:
        arr = rgb_array[:, :, None]
    elif rgb_array.ndim == 3:
        arr = rgb_array[:, :, :3]
    else:
        return 0.0, 0.0, 0.0, 0, False, ""

    arr_u8 = np.asarray(arr, dtype=np.uint8)
    lsb_entropies: List[float] = []
    bp1_entropies: List[float] = []
    p_values: List[float] = []
    total_active_pairs = 0
    channel_stego_reason = ""

    for ch in range(arr_u8.shape[2]):
        chan = arr_u8[:, :, ch]
        lsb = chan & 1
        bp1 = (chan >> 1) & 1
        ch_lsb = _bitplane_entropy(lsb)
        ch_bp1 = _bitplane_entropy(bp1)
        lsb_entropies.append(ch_lsb)
        bp1_entropies.append(ch_bp1)

        hist = np.bincount(chan.ravel(), minlength=256).astype(np.float64)
        even = hist[0::2]
        odd = hist[1::2]
        pair_sum = even + odd
        active = pair_sum >= 8.0
        n_active = int(np.sum(active))
        total_active_pairs += n_active
        ch_pval = 0.0
        if n_active >= 8:
            expected = 0.5 * pair_sum[active]
            chi2_stat = float(
                np.sum(((even[active] - expected) ** 2 + (odd[active] - expected) ** 2) / expected)
            )
            ch_pval = float(scipy_chi2.sf(chi2_stat, df=n_active - 1))
            p_values.append(ch_pval)

        # Anomaly 1: Low-entropy host channel whose LSB plane was injected with high-entropy bits
        ch_div = ch_lsb - ch_bp1
        if not channel_stego_reason and ch_lsb >= 0.45 and ch_bp1 <= 0.12 and ch_div >= 0.35:
            channel_stego_reason = f"lsb_plane_divergence(ch={ch},lsb={ch_lsb:.3f},bp1={ch_bp1:.3f})"

        # Anomaly 2: Textured/natural channel showing Pairs-of-Values equalization under LSB embedding
        if not channel_stego_reason and ch_pval >= 0.90 and ch_lsb >= 0.85 and n_active >= 16:
            channel_stego_reason = f"pov_chi2_equalization(ch={ch},p={ch_pval:.3f},lsb={ch_lsb:.3f})"

    mean_lsb = float(np.mean(lsb_entropies)) if lsb_entropies else 0.0
    mean_bp1 = float(np.mean(bp1_entropies)) if bp1_entropies else 0.0
    max_pval = max(p_values) if p_values else 0.0

    if channel_stego_reason:
        return (
            mean_lsb,
            mean_bp1,
            max_pval,
            total_active_pairs,
            True,
            channel_stego_reason,
        )

    return mean_lsb, mean_bp1, max_pval, total_active_pairs, False, ""


def detect_typographic_list_attack(
    ocr_lines: Sequence[str],
    rgb_array: Optional[np.ndarray] = None,
) -> Tuple[bool, Tuple[int, ...], float]:
    """Detect blank numbered list templates (e.g., FigStep '1. / 2. / 3.') and stroke density."""
    found_nums: List[int] = []
    for line in ocr_lines:
        m = _BLANK_LIST_ITEM_RE.match(line)
        if m:
            num = int(m.group(1))
            if num not in found_nums:
                found_nums.append(num)

    joined = "\n".join(ocr_lines)
    if len(found_nums) < 2 and _INLINE_TRAILING_LIST_RE.search(joined):
        for n in (1, 2, 3):
            if n not in found_nums and re.search(rf"(?m)\b{n}\b", joined):
                found_nums.append(n)

    contrast_ratio = 0.0
    if rgb_array is not None and rgb_array.size > 0:
        gray = rgb_array.mean(axis=2) if rgb_array.ndim == 3 else rgb_array
        white_frac = float(np.mean(gray > 235))
        dark_frac = float(np.mean(gray < 80))
        contrast_ratio = min(white_frac, 1.0) * (1.0 if dark_frac > 0.002 else 0.0)

    is_list_attack = len(set(found_nums) & {1, 2, 3}) >= 2
    return is_list_attack, tuple(sorted(found_nums)), float(contrast_ratio)


def _tensor_or_array_to_pil(image: Union[torch.Tensor, np.ndarray, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, torch.Tensor):
        t = image.detach().cpu()
        if t.ndim == 4:
            t = t[0]
        if t.ndim == 3 and t.shape[0] in (1, 3, 4) and t.shape[-1] not in (1, 3, 4):
            t = t.permute(1, 2, 0)
        arr = t.numpy()
    elif isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
    else:
        raise TypeError(f"Unsupported image input type: {type(image).__name__}")

    if np.issubdtype(arr.dtype, np.floating):
        if float(np.min(arr)) < 0.0:
            arr = (arr + 1.0) * 127.5
        elif float(np.max(arr)) <= 1.0 + 1e-6:
            arr = arr * 255.0
        arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L").convert("RGB")
    if arr.shape[2] == 1:
        return Image.fromarray(arr[:, :, 0], mode="L").convert("RGB")
    return Image.fromarray(arr[:, :, :3], mode="RGB")


class MultimodalExtractor:
    """Extracts text and safety forensics from images (Apple Vision OCR) and audio samples."""

    def __init__(
        self,
        *,
        cache_dir: Optional[Path] = None,
        ocr_timeout_sec: float = 60.0,
        enable_vision_ocr: bool = True,
    ) -> None:
        self.ocr_timeout_sec = float(ocr_timeout_sec)
        self.enable_vision_ocr = bool(enable_vision_ocr)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else self._resolve_cache_dir()
        self._compiled_bin: Optional[Path] = None
        self._swift_src: Optional[Path] = None

    @staticmethod
    def _resolve_cache_dir() -> Path:
        try:
            _PRIME_KERNEL_DIR.mkdir(parents=True, exist_ok=True)
            gitignore = _PRIME_KERNEL_DIR / ".gitignore"
            if not gitignore.exists():
                gitignore.write_text("*\n", encoding="utf-8")
            probe = _PRIME_KERNEL_DIR / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return _PRIME_KERNEL_DIR
        except OSError:
            fallback = Path(tempfile.gettempdir()) / "ojasx_aegis_cache"
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback

    def _ensure_vision_helper(self) -> Tuple[Optional[Path], Optional[Path]]:
        """Compile or locate cached Apple Vision Swift OCR helper in .prime-kernel/ or tmp."""
        if not self.enable_vision_ocr:
            return None, None
        swift_bin = shutil.which("swift")
        swiftc_bin = shutil.which("swiftc")
        if swift_bin is None and swiftc_bin is None:
            return None, None

        src_path = self.cache_dir / "aegis_vision_ocr.swift"
        bin_path = self.cache_dir / "aegis_vision_ocr"

        if not src_path.exists() or src_path.read_text(encoding="utf-8") != _SWIFT_OCR_SOURCE:
            src_path.write_text(_SWIFT_OCR_SOURCE, encoding="utf-8")
        self._swift_src = src_path

        if bin_path.exists() and bin_path.stat().st_mtime_ns >= src_path.stat().st_mtime_ns:
            self._compiled_bin = bin_path
            return bin_path, src_path

        if swiftc_bin is not None:
            tmp_bin = self.cache_dir / f".tmp_ocr_{os.getpid()}"
            try:
                proc = subprocess.run(
                    [swiftc_bin, str(src_path), "-o", str(tmp_bin)],
                    capture_output=True,
                    text=True,
                    timeout=self.ocr_timeout_sec,
                    check=False,
                )
                if proc.returncode == 0 and tmp_bin.exists():
                    os.replace(tmp_bin, bin_path)
                    self._compiled_bin = bin_path
                    return bin_path, src_path
            except Exception:
                pass
            finally:
                if tmp_bin.exists():
                    try:
                        tmp_bin.unlink()
                    except OSError:
                        pass

        return None, src_path

    def _run_apple_vision_batch(self, image_paths: Sequence[Path]) -> Dict[str, Tuple[List[str], str]]:
        """Invoke Apple Vision VNRecognizeTextRequest (.accurate) on a batch of image paths."""
        results: Dict[str, Tuple[List[str], str]] = {}
        if not image_paths:
            return results

        bin_path, src_path = self._ensure_vision_helper()
        cmd: Optional[List[str]] = None
        backend_label = "fallback_pil_typography"
        if bin_path is not None and bin_path.exists():
            cmd = [str(bin_path)] + [str(p) for p in image_paths]
            backend_label = "apple_vision_compiled"
        elif src_path is not None and shutil.which("swift") is not None:
            cmd = ["swift", str(src_path)] + [str(p) for p in image_paths]
            backend_label = "apple_vision_swift"

        if cmd is None:
            return results

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=max(self.ocr_timeout_sec, 10.0 * len(image_paths)),
                check=False,
            )
            if proc.returncode != 0:
                return results
            for line in proc.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                p_str = rec.get("path", "")
                if not rec.get("error"):
                    results[p_str] = (list(rec.get("lines") or []), backend_label)
        except Exception:
            return results
        return results

    @staticmethod
    def _pil_typography_fallback(rgb_array: np.ndarray) -> Tuple[List[str], str]:
        """Graceful PIL/numpy structural typography fallback when Swift/Vision is unavailable."""
        if rgb_array.size == 0:
            return [], "fallback_empty"
        gray = rgb_array.mean(axis=2) if rgb_array.ndim == 3 else rgb_array
        dark_mask = gray < 100.0
        row_ink = dark_mask.mean(axis=1)
        active_rows = np.where(row_ink > 0.01)[0]
        white_frac = float(np.mean(gray > 235.0))
        if len(active_rows) > 0 and white_frac > 0.65:
            # Estimate active horizontal text band structure for telemetry
            return [f"[visual_typography_bands:{len(active_rows)}]"], "fallback_pil_typography"
        return [], "fallback_pil_typography"

    def inspect_images_batch(
        self,
        images: Sequence[Union[str, Path, Image.Image, torch.Tensor, np.ndarray]],
    ) -> List[MultimodalExtractionResult]:
        """Extract OCR text and forensics for multiple images in one Vision batch call."""
        temp_files: List[Path] = []
        resolved_paths: List[Path] = []
        rgb_arrays: List[np.ndarray] = []
        png_chunks_list: List[Dict[str, str]] = []

        try:
            for item in images:
                if isinstance(item, (str, Path)):
                    p = Path(item)
                    resolved_paths.append(p)
                    with Image.open(p) as im:
                        rgb_arrays.append(np.asarray(im.convert("RGB"), dtype=np.uint8))
                    png_chunks_list.append(extract_png_text_chunks(p))
                else:
                    pil_im = _tensor_or_array_to_pil(item)
                    rgb_arrays.append(np.asarray(pil_im, dtype=np.uint8))
                    fd, tmp_name = tempfile.mkstemp(prefix="aegis_mm_", suffix=".png")
                    os.close(fd)
                    tmp_path = Path(tmp_name)
                    pil_im.save(tmp_path, format="PNG")
                    temp_files.append(tmp_path)
                    resolved_paths.append(tmp_path)
                    png_chunks_list.append(extract_png_text_chunks(tmp_path))

            vision_out = self._run_apple_vision_batch(resolved_paths)
            outputs: List[MultimodalExtractionResult] = []
            for p, rgb_arr, meta_chunks in zip(resolved_paths, rgb_arrays, png_chunks_list):
                hit = vision_out.get(str(p))
                if hit is not None:
                    lines, backend = hit
                else:
                    lines, backend = self._pil_typography_fallback(rgb_arr)

                lsb_ent, bp1_ent, pval, active_pairs, stego_flag, stego_reason = analyze_lsb_steganography(rgb_arr)
                list_attack, blank_nums, contrast_ratio = detect_typographic_list_attack(lines, rgb_arr)

                ocr_text = "\n".join(lines).strip()
                meta_text = "\n".join(f"{k}: {v}" for k, v in sorted(meta_chunks.items())).strip()
                parts = [part for part in (ocr_text, meta_text) if part]
                combined_text = "\n".join(parts)

                forensics = ImageForensics(
                    png_metadata=meta_chunks,
                    lsb_entropy=lsb_ent,
                    bp1_entropy=bp1_ent,
                    pov_chi2_p_value=pval,
                    pov_active_pairs=active_pairs,
                    stego_detected=stego_flag,
                    stego_reason=stego_reason,
                    stroke_contrast_ratio=contrast_ratio,
                    typographic_list_attack=list_attack,
                    blank_list_numbers=blank_nums,
                )
                outputs.append(
                    MultimodalExtractionResult(
                        extracted_text=ocr_text,
                        ocr_lines=lines,
                        ocr_backend=backend,
                        forensics=forensics,
                        combined_inspection_text=combined_text,
                    )
                )
            return outputs
        finally:
            for tmp in temp_files:
                try:
                    tmp.unlink()
                except OSError:
                    pass

    def extract_image_text(
        self,
        image: Union[str, Path, Image.Image, torch.Tensor, np.ndarray],
    ) -> MultimodalExtractionResult:
        """Inspect a single image path, PIL image, numpy array, or PyTorch tensor."""
        return self.inspect_images_batch([image])[0]

    @staticmethod
    def extract_audio_text(audio_sample: Any) -> AudioExtractionResult:
        """Extract transcript and embedded header metadata from an audio sample."""
        if isinstance(audio_sample, str):
            return AudioExtractionResult(
                extracted_text=audio_sample.strip(),
                source_kind="transcript_str",
                metadata={},
            )

        if isinstance(audio_sample, dict):
            text_parts: List[str] = []
            for key in ("audio_transcript", "transcript", "text", "caption", "utterance"):
                val = audio_sample.get(key)
                if isinstance(val, str) and val.strip():
                    text_parts.append(val.strip())
            segments = audio_sample.get("segments")
            if isinstance(segments, list):
                for seg in segments:
                    if isinstance(seg, dict) and isinstance(seg.get("text"), str):
                        text_parts.append(seg["text"].strip())
                    elif isinstance(seg, str) and seg.strip():
                        text_parts.append(seg.strip())
            meta = {k: v for k, v in audio_sample.items() if k not in ("waveform", "audio", "tensor")}
            return AudioExtractionResult(
                extracted_text="\n".join(dict.fromkeys(text_parts)),
                source_kind="audio_dict",
                metadata=meta,
            )

        if isinstance(audio_sample, (Path, bytes, bytearray)):
            raw: bytes
            if isinstance(audio_sample, Path):
                raw = audio_sample.read_bytes() if audio_sample.is_file() else b""
            else:
                raw = bytes(audio_sample)
            # Extract ASCII metadata strings from RIFF/WAV LIST/INFO chunks if present
            ascii_chunks = re.findall(rb"[ -~]{8,}", raw[:65536])
            decoded = " ".join(c.decode("ascii", errors="ignore") for c in ascii_chunks if not c.startswith(b"RIFF"))
            return AudioExtractionResult(
                extracted_text=decoded.strip(),
                source_kind="audio_binary",
                metadata={"byte_length": len(raw)},
            )

        if isinstance(audio_sample, (torch.Tensor, np.ndarray)):
            shape = tuple(audio_sample.shape)
            return AudioExtractionResult(
                extracted_text="",
                source_kind="audio_waveform_no_transcript",
                metadata={"shape": shape},
            )

        return AudioExtractionResult(
            extracted_text=str(audio_sample).strip(),
            source_kind="audio_other",
            metadata={},
        )


__all__ = [
    "AudioExtractionResult",
    "ImageForensics",
    "MultimodalExtractor",
    "MultimodalExtractionResult",
    "analyze_lsb_steganography",
    "detect_typographic_list_attack",
    "extract_png_text_chunks",
]
