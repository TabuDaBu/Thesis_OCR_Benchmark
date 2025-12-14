from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import evaluate
import pandas as pd

PROJECT_ROOT = Path(__file__).parent


@dataclass
class cer:
    """Utility to compute Character Error Rate (CER) for OCR outputs."""

    dataset_dir: Path = PROJECT_ROOT / "dataset_to_process"
    output_excel: Path = PROJECT_ROOT / "dataset_to_process/cer_results.xlsx"
    _cer_metric = evaluate.load("cer")

    @classmethod
    def calculate_cer(cls, ground_truth: str, prediction: str) -> float:
        """Return the CER given two strings using Hugging Face evaluate library."""
        ground_truth = ground_truth.strip()
        prediction = prediction.strip()

        # Fall 1: beide leer -> perfekte Übereinstimmung
        if not ground_truth and not prediction:
            return 0.0

        # Fall 2: eine Seite leer, die andere nicht -> schlechtester Fall, aber max. 1.0
        if not ground_truth or not prediction:
            return 1.0

        # Normalfall: mit HF-Metrik berechnen
        result = cls._cer_metric.compute(
            predictions=[prediction],
            references=[ground_truth],
        )

        cer_value = float(result)

        # Sicherheit: CER in [0.0, 1.0] clampen
        if cer_value < 0.0:
            return 0.0
        if cer_value > 1.0:
            return 1.0
        return cer_value

    def _iter_test_pairs(self, folder: Path) -> Iterable[tuple[str, Path, Path]]:
        """Yield (sample_id, ocred_path, gt_path) tuples present in folder."""
        for ocred_path in sorted(folder.glob("*_ocred.txt")):
            sample_id = ocred_path.name.split("_ocred", 1)[0]
            gt_path = folder / f"{sample_id}.gt.txt"
            if gt_path.exists():
                yield sample_id, ocred_path, gt_path

    def evaluate_folder(
        self,
        folder: Path | None = None,
        output_path: Path | None = None,
    ) -> Path:
        """Compute CER for every *_ocred.txt vs *.gt.txt pair and write Excel results."""
        folder_path = Path(folder) if folder else self.dataset_dir
        destination = Path(output_path) if output_path else self.output_excel

        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        rows: list[dict[str, object]] = []
        for sample_id, ocred_path, gt_path in self._iter_test_pairs(folder_path):
            ocr_text = ocred_path.read_text(encoding="utf-8").strip()
            gt_text = gt_path.read_text(encoding="utf-8").strip()
            rows.append(
                {
                    "sample": sample_id,
                    "cer": self.calculate_cer(gt_text, ocr_text),
                    "ocr_text": ocr_text,
                    "ground_truth": gt_text,
                    "ocr_file": ocred_path.name,
                    "gt_file": gt_path.name,
                }
            )

        if not rows:
            raise RuntimeError(
                f"No matching *_ocred.txt / *.gt.txt pairs found in {folder_path}"
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_excel(destination, index=False)
        return destination


def main() -> None:
    """Main entry point for running CER evaluation."""
    evaluator = cer()
    evaluator.evaluate_folder()


if __name__ == "__main__":
    main()
