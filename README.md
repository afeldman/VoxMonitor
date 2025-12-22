# VoxMonitor

Trainingspipeline für Klassifikation von Schweine-Lauten (Grunzen/Quieken/…):

- Preprocessing: FFT → Mel-Spektrogramm
- CNN-Backbone (MobileNetV2)
- Multi-Task Heads: age, sex, valence, context
- Augmentierung: über aptt-Framework (dynamisch ladbar)
  - Optional: aptt-Lightning-Pipeline integrierbar (Modell-Factory, Augmentor, ONNX-Exporter)

## Setup (uv + pyproject)

1. Optional: Python-Umgebung aktivieren/erstellen

```bash
uv venv
source .venv/bin/activate
```

2. Abhängigkeiten installieren

```bash
uv pip install -e .
```

3. aptt lokal installieren (Editable Mode)

```bash
uv pip install -e /Users/anton.feldmann/Projects/ai/aptt/
```

4. Konfiguration prüfen: siehe [config/config.yaml](config/config.yaml)

## Training mit aptt

Die Trainings- und Export-Pipeline wird vollständig über das aptt-Projekt ausgeführt (Lightning, Augmentor, ONNX-Export). Installiere aptt im Editable-Mode und nutze dessen Trainings-CLI oder Module.

Beispiel (Platzhalter, bitte durch aptt-spezifische Kommandos ersetzen):

```bash
# aptt installieren
uv pip install -e /Users/anton.feldmann/Projects/ai/aptt/

# Training starten (Beispiel)
uv run python -m aptt.train --config config/config.yaml

# ONNX-Export (Beispiel)
uv run python -m aptt.export.onnx --ckpt checkpoints/best.pt --out exports/onnx/model.onnx
```

uv run python src/voxmonitor/train_pl.py
uv run voxmonitor-train-pl

````

Checkpoints werden in [checkpoints](checkpoints) abgelegt. Der beste Stand liegt in `best.pt`.

## Hinweise

- Die Spaltennamen für Labels sind in [config/config.yaml](config/config.yaml) unter `labels.columns` konfiguriert.
- Die Audio-Dateiliste wird aus [data/SoundwelDatasetKey.xlsx](data/SoundwelDatasetKey.xlsx) gelesen; die Spalte mit Dateinamen wird heuristisch erkannt oder kann über `data.audio_column_candidates` erzwungen werden.
- Für on-device und Server-Inferenz folgen separate Module (TFLite/Core ML und FastAPI). Zuerst wird die Klassifikations-Pipeline stabilisiert.
 Optionaler aptt-Exporter: setze in [config/config.yaml](config/config.yaml) `onnx.exporter_path` auf den entsprechenden aptt-Pfad.

## ONNX Export

```bash
uv run voxmonitor-export-onnx
## Struktur

- [config/config.yaml](config/config.yaml): zentrale Konfiguration (Datenpfade, Labels, Augmentor-/Modell-Factory-Pfade, ONNX-Exporter).
- [data/SoundwelDatasetKey.xlsx](data/SoundwelDatasetKey.xlsx): Label- und Dateischlüssel.
- [pyproject.toml](pyproject.toml): minimale Projektmetadaten; Abhängigkeiten über aptt.

## Referenzen

- Nature-Artikel (Konzeptgrundlage): https://www.nature.com/articles/s41598-022-07174-8
- Datensatz/Material (Zenodo): https://zenodo.org/records/8252482

Die Ausgabe landet in `exports/onnx/model.onnx`. Konfigurierbar über `onnx.export_dir`, `onnx.opset` und optionalen aptt-Exporter.
````
