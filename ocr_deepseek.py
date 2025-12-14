from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Iterable, Any, Optional
import contextlib
import io
import re

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModel

# Reuse from Tesseract script
try:
    from ocr_tesseract import (
        PROJECT_ROOT,
        TEXT_SUFFIX,
        collect_image_paths,
        flatten_ocr_text,
    )
except ImportError:
    from ocr_tesseract import (  # type: ignore[no-redef]
        PROJECT_ROOT,
        TEXT_SUFFIX,
        collect_image_paths,
        flatten_ocr_text,
    )

SUPPORTED_PATH_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_MODEL_NAME = "deepseek-ai/DeepSeek-OCR"


class DeepSeekOcr:
    """
    Local DeepSeek-OCR wrapper, aligned with your existing OCR APIs.

    Uses the model's custom `infer` method (trust_remote_code=True).

    IMPORTANT:
    DeepSeek's infer() often prints OCR text to stdout instead of returning it.
    We capture stdout and parse it.

    TEXT NORMALIZATION CONTRACT (same as your Tesseract script):
    - fix hyphenated line breaks
    - remove all remaining newlines (turn them into spaces)
    - collapse multiple spaces
    - final output is ALWAYS single-line
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "auto",
        dtype: str = "auto",
        base_size: int = 1024,
        image_size: int = 1024,
        crop_mode: bool = False,
        save_results: bool = False,
        test_compress: bool = True,
        attn_implementation: Optional[str] = "flash_attention_2",
        output_dir: Optional[Path] = None,
        **model_kwargs: Any,
    ) -> None:
        self.model_name = model_name

        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(dtype)

        self.base_size = base_size
        self.image_size = image_size
        self.crop_mode = crop_mode
        self.save_results = save_results
        self.test_compress = test_compress
        self.attn_implementation = attn_implementation

        self.output_dir = output_dir or (PROJECT_ROOT / "deepseek_outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._tokenizer = None
        self._model = None
        self._model_kwargs = model_kwargs

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        return device

    @staticmethod
    def _resolve_dtype(dtype: str) -> torch.dtype:
        if dtype == "auto":
            if torch.cuda.is_available():
                return torch.bfloat16
            return torch.float32

        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        return mapping.get(dtype.lower(), torch.float32)

    def _ensure_loaded(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        model_load_kwargs: dict[str, Any] = dict(
            trust_remote_code=True,
            use_safetensors=True,
            **self._model_kwargs,
        )

        # Be explicit to avoid FA auto-toggles
        if self.attn_implementation:
            model_load_kwargs["attn_implementation"] = self.attn_implementation
            model_load_kwargs["_attn_implementation"] = self.attn_implementation
        else:
            model_load_kwargs["attn_implementation"] = "eager"

        self._model = AutoModel.from_pretrained(self.model_name, **model_load_kwargs)
        self._model = self._model.eval().to(self.device).to(self.dtype)

    def _normalize_image_if_needed(self, image_path: Path) -> Path:
        suffix = image_path.suffix.lower()
        if suffix in SUPPORTED_PATH_SUFFIXES:
            return image_path

        with Image.open(image_path) as img:
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            tmp = self.output_dir / f"{image_path.stem}_converted.png"
            img.save(tmp)
            return tmp

    def _extract_text(self, result: Any) -> str:
        if result is None:
            return ""

        if isinstance(result, str):
            return result.strip()

        if isinstance(result, dict):
            for key in ("text", "ocr_text", "content", "result", "texts", "pred_text"):
                val = result.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
                if isinstance(val, list):
                    joined = "\n".join(str(x) for x in val if x)
                    if joined.strip():
                        return joined.strip()

        if isinstance(result, (list, tuple)):
            parts: list[str] = []
            for item in result:
                t = self._extract_text(item)
                if t:
                    parts.append(t)
            if parts:
                return "\n".join(parts).strip()

        return ""

    def _extract_text_from_stdout(self, printed: str) -> str:
        """
        Parse DeepSeek's verbose stdout.
        We filter debug lines and keep likely OCR text.

        Then apply the SAME single-line normalization as Tesseract
        via flatten_ocr_text().
        """
        if not printed:
            return ""

        lines = [ln.rstrip() for ln in printed.splitlines() if ln.strip()]

        skip_patterns = [
            r"^=+$",
            r"^BASE:",
            r"^PATCHES:",
            r"^NO PATCHES",
            r"^Miscellaneous$",
            r"^image size:",
            r"^valid image tokens:",
            r"^output texts tokens",
            r"^compression ratio:",
            r"^Setting `pad_token_id`",
            r"^The attention mask",
            r"^The `seen_tokens` attribute is deprecated",
            r"^`get_max_cache\(\)` is deprecated",
            r"^The attention layers in this model are transitioning",
            r"^One time compensation",
        ]

        def is_skip(line: str) -> bool:
            return any(re.search(p, line) for p in skip_patterns)

        candidates = [ln for ln in lines if not is_skip(ln)]
        if not candidates:
            return ""

        # De-duplicate consecutive identical lines
        deduped: list[str] = []
        for ln in candidates:
            if not deduped or deduped[-1] != ln:
                deduped.append(ln)

        # ✅ Important: join with newlines then flatten using shared logic
        joined = "\n".join(deduped)
        return flatten_ocr_text(joined).strip()

    def ocr_image(
        self,
        image_path: Path,
        *,
        prompt: str = "<image>\nFree OCR.",
    ) -> str:
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image: {image_path}")

        self._ensure_loaded()
        normalized_path = self._normalize_image_if_needed(image_path)

        # Capture stdout so DeepSeek doesn't spam console,
        # and so we can extract OCR text from it.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = self._model.infer(  # type: ignore[attr-defined]
                self._tokenizer,
                prompt=prompt,
                image_file=str(normalized_path),
                output_path=str(self.output_dir),
                base_size=self.base_size,
                image_size=self.image_size,
                crop_mode=self.crop_mode,
                save_results=self.save_results,
                test_compress=self.test_compress,
            )

        # 1) Try return object
        text = self._extract_text(result)

        # 2) Fallback to captured stdout
        if not text:
            text = self._extract_text_from_stdout(buf.getvalue())

        # ✅ Final normalization (identical to Tesseract contract)
        text = flatten_ocr_text(text)

        return text.strip()

    def ocr_images_to_txt(
        self,
        images: Iterable[Path],
        *,
        suffix: str = TEXT_SUFFIX,
        prompt: str = "<image>\nFree OCR.",
    ) -> None:
        for image in images:
            text = self.ocr_image(image, prompt=prompt)
            destination = image.with_name(f"{image.stem}{suffix}.txt")
            destination.write_text(text + "\n", encoding="utf-8")

    def run_on_default_dataset(
        self,
        *,
        prompt: str = "<image>\nFree OCR.",
    ) -> float:
        start = perf_counter()
        images = collect_image_paths()
        self.ocr_images_to_txt(images, prompt=prompt)
        total_seconds = perf_counter() - start

        script_name = Path(__file__).stem
        time_file = PROJECT_ROOT / f"{script_name}.txt"
        time_file.write_text(
            f"Total OCR processing time: {total_seconds:.2f} seconds\n",
            encoding="utf-8",
        )
        return total_seconds


def main() -> None:
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    ocr = DeepSeekOcr(
        device="auto",
        attn_implementation=None,  # keep as you already set
    )

    total_seconds = ocr.run_on_default_dataset(prompt="<image>\nFree OCR.")

    print("OCR (DeepSeek-OCR) complete. Text files written alongside source images.")
    print(f"Total OCR processing time: {total_seconds:.2f} seconds")
    script_name = Path(__file__).stem
    time_file = PROJECT_ROOT / f"{script_name}.txt"
    print(f"Processing time saved to: {time_file}")


if __name__ == "__main__":
    main()
