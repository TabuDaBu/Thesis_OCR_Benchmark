from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Iterable, Optional

from google.cloud import vision
from google.oauth2 import service_account

# Reuse helpers from  Tesseract script
from ocr_tesseract import (
    PROJECT_ROOT,
    TEXT_SUFFIX,
    collect_image_paths,
    flatten_ocr_text,
)
SUPPORTED_PATH_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
}


class GoogleVisionOcr:
    """
    Google Cloud Vision OCR wrapper for document text extraction.

    Uses the official Python client library:
    - ImageAnnotatorClient
    - document_text_detection() / text_detection()

    Auth options:
    - If credentials_json is given:
        * Loads a service account key from that file explicitly.
    """

    def __init__(
        self,
        *,
        use_document_text_detection: bool = True,
        credentials_json: Optional[Path | str] = None,
    ) -> None:
        if credentials_json:
            creds = service_account.Credentials.from_service_account_file(
                str(credentials_json)
            )
            self.client = vision.ImageAnnotatorClient(credentials=creds)
        else:
            self.client = vision.ImageAnnotatorClient()

        self.use_document_text_detection = use_document_text_detection

    # ---------- internal helpers ---------- #

    def _build_image(self, image_path: Path) -> vision.Image:
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image: {image_path}")

        with image_path.open("rb") as f:
            content = f.read()

        # Official pattern: vision.Image(content=bytes) :contentReference[oaicite:2]{index=2}
        return vision.Image(content=content)

    def _extract_text_from_response(
        self,
        response: vision.AnnotateImageResponse,
    ) -> str:
        # If Vision signals an error, surface it
        if getattr(response, "error", None) and response.error.message:
            raise RuntimeError(f"Google Vision API error: {response.error.message}")

        # Prefer full_text_annotation (document text OCR) :contentReference[oaicite:3]{index=3}
        annotation = getattr(response, "full_text_annotation", None)
        if annotation and getattr(annotation, "text", None):
            raw = annotation.text or ""
        else:
            # Fallback: first text_annotations entry
            anns = getattr(response, "text_annotations", None)
            if anns:
                raw = anns[0].description or ""
            else:
                raw = ""

        return flatten_ocr_text(raw)

    # ---------- public API (same shape as your other OCR classes) ---------- #

    def ocr_image(self, image_path: Path) -> str:
        suffix = image_path.suffix.lower()
        if suffix not in SUPPORTED_PATH_SUFFIXES:
            raise ValueError(
                f"Unsupported file type for Google Vision OCR: {suffix} "
                f"(supported: {sorted(SUPPORTED_PATH_SUFFIXES)})"
            )

        image = self._build_image(image_path)

        if self.use_document_text_detection:
            response = self.client.document_text_detection(image=image)
        else:
            response = self.client.text_detection(image=image)

        text = self._extract_text_from_response(response)
        return text.strip()

    def ocr_images_to_txt(
        self,
        images: Iterable[Path],
        *,
        suffix: str = TEXT_SUFFIX,
    ) -> None:
        for image in images:
            try:
                text = self.ocr_image(image)
            except Exception as exc:
                print(f"[WARN] Skipping {image.name}: {exc}")
                continue

            out_path = image.with_name(f"{image.stem}{suffix}.txt")
            out_path.write_text(text + "\n", encoding="utf-8")

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
    ocr = GoogleVisionOcr(
        use_document_text_detection=True,
        credentials_json=Path(r"C:\Users\Daniel\gcloud_keys\vision-ocr-sa.json"),
    )
    total_seconds = ocr.run_on_default_dataset()

    print("OCR (Google Cloud Vision) complete.")
    print(f"Total OCR processing time: {total_seconds:.2f} seconds")
    script_name = Path(__file__).stem
    time_file = PROJECT_ROOT / f"{script_name}.txt"
    print(f"Processing time saved to: {time_file}")


if __name__ == "__main__":
    main()
