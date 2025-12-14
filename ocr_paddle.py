from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Iterable, Any

import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

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

SUPPORTED_PATH_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".pdf"}
PADDLE_LANG = "en"


class PaddleOcr:
    def __init__(self, *, lang: str = PADDLE_LANG, device: str = "cpu", **ocr_kwargs: Any) -> None:
        self.lang = lang
        self._ocr = PaddleOCR(
            lang=self.lang,
            use_textline_orientation=False,
            device=device,
            **ocr_kwargs,
        )

    def _extract_text_from_result(self, result) -> str:
        if not result:
            return ""

        texts: list[str] = []
        for res in result:
            try:
                rec_texts = res["rec_texts"]
                rec_scores = res["rec_scores"]
            except (KeyError, TypeError):
                continue

            for txt, score in zip(rec_texts, rec_scores):
                txt = (txt or "").strip()
                if txt:
                    texts.append(txt)

        return flatten_ocr_text(" ".join(texts))

    def ocr_image(self, image_path: Path) -> str:
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image: {image_path}")

        suffix = image_path.suffix.lower()

        if suffix in SUPPORTED_PATH_SUFFIXES:
            inp: Any = str(image_path)
        else:
            img = Image.open(image_path)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            inp = np.array(img)

        result = self._ocr.predict(inp)
        return self._extract_text_from_result(result)

    def ocr_images_to_txt(self, images: Iterable[Path], *, suffix: str = TEXT_SUFFIX) -> None:
        for image in images:
            text = self.ocr_image(image)
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
    ocr = PaddleOcr(device="cpu")
    total_seconds = ocr.run_on_default_dataset()

    print("OCR (PaddleOCR) complete. Text files written alongside source images.")
    print(f"Total OCR processing time: {total_seconds:.2f} seconds")
    script_name = Path(__file__).stem
    time_file = PROJECT_ROOT / f"{script_name}.txt"
    print(f"Processing time saved to: {time_file}")


if __name__ == "__main__":
    main()
