# Bachelorprojekt - OCR-Benchmark

## Idee des Projekts
Mehrere OCR-Engines auf einem lokalen Test-Datensatz laufen lassen und die Ergebnisse gegen Ground Truth per **CER**, **WER** und der **Verarbeitungszeit** auswerten.

## Ordnerstruktur
- Datensatz liegt unter: `dataset_to_process/`
- OCR-Ausgaben werden **neben den Bildern** als `*_ocred.txt` gespeichert (Suffix: `_ocred`)
- Für die Auswertung werden Paare erwartet:
  - OCR: `SAMPLE_ocred.txt`
  - Ground Truth: `SAMPLE.gt.txt`

## OCR-Skripte
### Tesseract (`ocr_tesseract.py`)
- Sprache: `DEFAULT_LANG = "eng"`

### EasyOCR (`ocr_easyocr.py`)
- Sprache: `langs=["en"]`, `gpu=False`

### PaddleOCR (`ocr_paddle.py`)
- Sprache: `lang="en"`, `device="cpu"`

### Google Cloud Vision (`ocr_google.py`)
- Authentifizierung erfordert Service-Account JSON 

### DeepSeek-OCR (`ocr_deepseek.py`)
- Model: `deepseek-ai/DeepSeek-OCR`
- Prompt: `str = "<image>\nFree OCR."`

**Hinweis:** Alle OCR-Skripte normalisieren Text (Zeilenumbrüche → Leerzeichen, doppelte Leerzeichen entfernen, Trennstriche beheben).

## Optionale Ground Truth Normalisierung
- Skript: `gt_normalization.py`
- Erzeugt normalisierte `.gt.txt` Dateien im Ordner `dataset_to_process/ground_truth_normalized/` falls erforderlich

## Auswertung
### CER (`cer.py`)
- Schreibt Ergebnis nach `dataset_to_process/cer_results.xlsx`

### WER (`wer.py`)
- Schreibt Ergebnis nach `dataset_to_process/wer_results.xlsx`

Beide Skripte suchen automatisch passende OCR/GT-Paare im Zielordner.

## Requirements
Alle Abhängigkeiten in `requirements.txt`:

## Alle Projektdateien als Link
Aufgrund der GitHub-Uploadbegrenzung kann das gesamte Projekt inklusive der Ergebnisse unter folgendem Link heruntergeladen werden:
https://aauklagenfurt-my.sharepoint.com/:u:/g/personal/dabuttazoni_edu_aau_at/IQBoY1omDVLjQr8EiiKT2ufsAWfNMNIHpEwg9N_hzmCMxLw?e=iMDZC6
