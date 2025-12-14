from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
TEST_DATA_DIR = PROJECT_ROOT / "dataset_to_process"
NORMALIZED_DIR = PROJECT_ROOT / "dataset_to_process" / "ground_truth_normalized"


class gt_normalization:

    def __init__(self, input_dir: Path = TEST_DATA_DIR, output_dir: Path = NORMALIZED_DIR) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir

    @staticmethod
    def flatten_text(text: str) -> str:
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Join hyphenated line-breaks: "in-\nvestigate" -> "investigate"
        text = re.sub(r"-\s*\n\s*", "", text)

        # Replace all remaining newlines (including blank ones) with a single space
        text = re.sub(r"\s*\n\s*", " ", text)

        # Collapse multiple spaces/tabs into a single space
        text = re.sub(r"\s{2,}", " ", text)

        return text.strip()

    def collect_text_paths(self) -> list[Path]:
        """Collect all .txt files under the input directory."""
        if not self.input_dir.is_dir():
            raise SystemExit(f"Test data directory not found: {self.input_dir}")

        paths = sorted(self.input_dir.rglob("*.txt"))

        if not paths:
            raise SystemExit(f"No text files found in {self.input_dir}")

        return paths

    def normalize_file(self, source: Path) -> None:
        relative = source.relative_to(self.input_dir)
        gt_name = f"{relative.stem}.gt{relative.suffix}"
        relative_with_suffix = relative.with_name(gt_name)

        target = self.output_dir / relative_with_suffix
        target.parent.mkdir(parents=True, exist_ok=True)

        raw_text = source.read_text(encoding="utf-8")
        normalized = self.flatten_text(raw_text)

        target.write_text(normalized + "\n", encoding="utf-8")

    def run(self) -> None:
        """Normalize all .txt files from input_dir into output_dir."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        text_files = self.collect_text_paths()

        for source in text_files:
            self.normalize_file(source)


def main() -> None:
    normalizer = gt_normalization()
    normalizer.run()
    print(f"Normalization complete. Normalized files written to: {NORMALIZED_DIR}")


if __name__ == "__main__":
    main()
