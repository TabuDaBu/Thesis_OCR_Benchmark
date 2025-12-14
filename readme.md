# OCR Thesis Project

Projekt zur Ausführung und Auswertung verschiedener OCR-Pipelines auf einem gemeinsamen Datensatz.

## Klassenübersicht

- `gt_normalization`: Liest Ground-Truth-Textdateien aus `dataset_to_process`, entfernt Zeilenumbrüche und löst Silbentrennungen auf, um normalisierte Referenztexte zu erzeugen.  
- `GoogleVisionOcr`: Kapselt die Google-Cloud-Vision-API, führt (Document-)Text-Detection auf Bildern aus und speichert den erkannten Text als `.txt` neben den Quelldateien.  
- `DeepSeekOcr`: Lädt das lokale Modell `deepseek-ai/DeepSeek-OCR`, führt Inferenz auf Bildern durch und bereitet den erkannten Text im gleichen Format wie die anderen OCR-Skripte auf.  
- `EasyOcr`: Verwendet EasyOCR für gängige Bildformate, extrahiert Text und schreibt die Ergebnisse als `_ocred.txt` passend zur bestehenden Tesseract-/Paddle-Logik.  
- `PaddleOcr`: Nutzt `PaddleOCR`, extrahiert Text aus Bildern und speichert ihn mit gemeinsamer Helper-Logik aus `ocr_tesseract`.  
- `wer`: Berechnet die Word Error Rate (WER) zwischen Ground-Truth-Dateien und OCR-Ergebnissen und schreibt die Resultate tabellarisch in eine Excel-Datei.  
- `cer`: Berechnet die Character Error Rate (CER) mit der Hugging-Face-`evaluate`-Bibliothek und exportiert die Ergebnisse ebenfalls als Excel-Tabelle.

## Installation

```bash
pip install -r requirements.txt
