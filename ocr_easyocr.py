from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Iterable, Any

import numpy as np
from PIL import Image
import easyocr

# Reuse from Tesseract script to avoid duplicate code
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

# EasyOCR can read common raster formats by path.
# Your dataset collector in ocr_tesseract already restricts to:
# (".jpg", ".jpeg", ".png", ".tif", ".tiff")
SUPPORTED_PATH_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}

EASYOCR_LANGS = ["en"]


class EasyOcr:
    """
    EasyOCR implementation analogous to your Tesseract and PaddleOCR scripts.

    - Reuses constants and helpers from ocr_tesseract:
      PROJECT_ROOT, TEXT_SUFFIX, collect_image_paths, flatten_ocr_text
    - Writes output files alongside source images with the same suffix.
    - Uses the same dataset folder structure via collect_image_paths().
    """

    def __init__(
        self,
        *,
        langs: list[str] | None = None,
        gpu: bool = False,
        **reader_kwargs: Any,
    ) -> None:
        self.langs = langs or EASYOCR_LANGS
        self.gpu = gpu
        self._reader = easyocr.Reader(self.langs, gpu=self.gpu, **reader_kwargs)

    def _extract_text_from_result(self, result) -> str:
        """
        EasyOCR's readtext returns a list of:
        [(bbox, text, confidence), ...]
        """
        if not result:
            return ""

        texts: list[str] = []
        for item in result:
            try:
                # item: (bbox, text, conf)
                txt = (item[1] or "").strip()
            except (TypeError, IndexError):
                continue

            if txt:
                texts.append(txt)

        return flatten_ocr_text(" ".join(texts))

    def ocr_image(self, image_path: Path) -> str:
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image: {image_path}")

        suffix = image_path.suffix.lower()

        # Prefer path-based OCR when possible
        if suffix in SUPPORTED_PATH_SUFFIXES:
            result = self._reader.readtext(str(image_path))
            return self._extract_text_from_result(result)

        # Fallback: load with PIL and pass numpy array
        with Image.open(image_path) as img:
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            arr = np.array(img)

        result = self._reader.readtext(arr)
        return self._extract_text_from_result(result)

    def ocr_images_to_txt(self, images: Iterable[Path], *, suffix: str = TEXT_SUFFIX) -> None:
        for image in images:
            try:
                text = self.ocr_image(image)
            except Exception as exc:
                print(f"[WARN] Skipping {image.name}: {exc}")
                continue

            destination = image.with_name(f"{image.stem}{suffix}.txt")
            destination.write_text(text + "\n", encoding="utf-8")

    def run_on_default_dataset(self) -> float:
        start = perf_counter()
        images = collect_image_paths()
        self.ocr_images_to_txt(images)
        total_seconds = perf_counter() - start

        script_name = Path(__file__).stem
        time_file = PROJECT_ROOT / f"{script_name}.txt"
        time_file.write_text(
            f"Total OCR processing time: {total_seconds:.2f} seconds\n",
            encoding="utf-8",
        )
        return total_seconds


def main() -> None:
    # Keep default consistent with your Paddle script (CPU)
    ocr = EasyOcr(gpu=False)
    total_seconds = ocr.run_on_default_dataset()

    print("OCR (EasyOCR) complete. Text files written alongside source images.")
    print(f"Total OCR processing time: {total_seconds:.2f} seconds")
    script_name = Path(__file__).stem
    time_file = PROJECT_ROOT / f"{script_name}.txt"
    print(f"Processing time saved to: {time_file}")


if __name__ == "__main__":
    main()
