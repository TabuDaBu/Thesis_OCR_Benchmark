
from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter
import re

import pytesseract
from PIL import Image
from pytesseract import TesseractNotFoundError

PROJECT_ROOT = Path(__file__).parent
TEST_DATA_DIR = PROJECT_ROOT / "dataset_to_process"
DEFAULT_LANG = "eng"
TEXT_SUFFIX = "_ocred"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def _configure_tesseract() -> None:
    """Allow users to point to a custom tesseract.exe via TESSERACT_CMD env."""
    custom_cmd = os.environ.get("TESSERACT_CMD")
    if custom_cmd:
        pytesseract.pytesseract.tesseract_cmd = custom_cmd

    try:
        pytesseract.get_tesseract_version()
    except TesseractNotFoundError as exc:
        raise SystemExit(
            "Tesseract binary not found. Install it or set TESSERACT_CMD to the executable."
        ) from exc


def _ocr_pil_image(image: Image.Image, *, lang: str = DEFAULT_LANG) -> str:
    raw_text = pytesseract.image_to_string(image, lang=lang)
    cleaned = flatten_ocr_text(raw_text)
    return cleaned.strip()



def ocr_image(image_path: Path, *, lang: str = DEFAULT_LANG) -> str:
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing image: {image_path}")

    with Image.open(image_path) as img:
        return _ocr_pil_image(img, lang=lang)


def ocr_images_to_txt(images: list[Path], *, suffix: str = TEXT_SUFFIX, lang: str = DEFAULT_LANG) -> None:
    for image in images:
        text = ocr_image(image, lang=lang)
        destination = image.with_name(f"{image.stem}{suffix}.txt")
        destination.write_text(text + "\n", encoding="utf-8")



def collect_image_paths() -> list[Path]:
    if not TEST_DATA_DIR.is_dir():
        raise SystemExit(f"Test data directory not found: {TEST_DATA_DIR}")
    
    paths: list[Path] = []
    for suffix in IMAGE_SUFFIXES:
        paths.extend(sorted(TEST_DATA_DIR.rglob(f"*{suffix}")))
    
    if not paths:
        raise SystemExit(f"No images found in {TEST_DATA_DIR}")
    return paths



def flatten_ocr_text(text: str) -> str:
    """
    Make OCR text single-line:
    - fix hyphenated line breaks: 'in-\\nvestigate' -> 'investigate'
    - remove all remaining newlines (turn them into spaces)
    - collapse multiple spaces
    """
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Join hyphenated line-breaks: "in-\nvestigate" -> "investigate"
    text = re.sub(r"-\s*\n\s*", "", text)

    # Replace all remaining newlines (including blank ones) with a single space
    text = re.sub(r"\s*\n\s*", " ", text)

    # Collapse multiple spaces/tabs into a single space
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()




def main() -> None:
    _configure_tesseract()
    
    overall_start = perf_counter()
    images = collect_image_paths()
    ocr_images_to_txt(images)
    total_seconds = perf_counter() - overall_start
    
    print("OCR complete. Text files written alongside source images.")
    print(f"Total OCR processing time: {total_seconds:.2f} seconds")
    
    # Write processing time to file
    script_name = Path(__file__).stem
    time_file = PROJECT_ROOT / f"{script_name}.txt"
    time_file.write_text(f"Total OCR processing time: {total_seconds:.2f} seconds\n", encoding="utf-8")
    print(f"Processing time saved to: {time_file}")


if __name__ == "__main__":
    main()

