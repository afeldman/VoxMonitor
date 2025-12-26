# 🚀 VoxMonitor Training - Implementation Summary

## Status: ✅ READY FOR 1-EPOCH TEST

Das VoxMonitor-Projekt wurde vollständig mit DeepSuite integriert und kann jetzt mit den lokalen Soundwell-Daten trainiert werden.

---

## 📋 Durchgeführte Implementierungen

### 1. **VoxMonitorTrainer** (train.py)
- ✅ Erweitert DeepSuite's BaseTrainer
- ✅ Vollständige Initialisierung mit super().__init__()
- ✅ Automatische ONNX-Export-Unterstützung
- ✅ MLflow-Integration
- ✅ Checkpoint-Management

### 2. **SoundwelDataset Enhancement** (data.py)
- ✅ Flexible CSV-Datei-Unterstützung
- ✅ Pre-loaded DataFrame-Unterstützung
- ✅ Automatische Spalten-Name-Zuordnung
- ✅ Audio-Datei-Loading mit torchaudio
- ✅ Mel-Spektrogramm-Extraktion

**Neue Parameter:**
```python
SoundwelDataset(
    root_dir="/Volumes/.../Soundwel",
    csv_path="/Volumes/.../SoundwelDatasetKey.csv",  # ← CSV-Pfad
    metadata=df,                                      # ← oder pre-loaded
    label_columns=["age", "sex", "valence", "context"],
    sample_rate=16000,
    max_length_sec=3.0,
    download=False,  # Local data
)
```

### 3. **Registry Cleanup**
- ✅ VoxMonitor registry.py nutzt nur noch DeepSuite HeadRegistry
- ✅ Keine Duplikation mehr
- ✅ Re-Export für Convenience-Imports

### 4. **Lightning Module Testing**
- ✅ VoxMonitorLightningModule vollständig getestet
- ✅ Multi-Task Learning mit 4 Klassifikations-Zielen
- ✅ Per-Task Loss & Accuracy Tracking
- ✅ Hyperparameter Saving

---

## 🎯 Lokale Daten-Struktur

```
data/
├── audio/                       # Audio-Dateien (WAV)
│   ├── sample1.wav
│   ├── sample2.wav
│   └── ...
├── SoundwelDatasetKey.csv       # Metadaten + Labels
└── training/                    # Training-Artefakte
```

**CSV-Struktur:**
- Audio Filename: Dateiname (z.B. "ETHZETHZPositivePositivepig1510.wav")
- Age Category: Piglet, Weaner, Grower, Finisher
- Sex: male, female
- Valence: Pos, Neg, (Neutral)
- Context: Enriched, Barren, Isolation, etc.

---

## 🚀 Training starten

### Option 1: Quick 1-Epoch Test
```bash
cd /Users/anton.feldmann/Projects/priv/VoxMonitor
uv run python train_local.py
```

**Was passiert:**
1. Config wird erstellt (config_local.yaml)
2. CSV mit 3000+ Samples wird geladen
3. Dataset wird initialisiert
4. 1 Epoch Training startet
5. Metriken werden geloggt
6. ONNX-Export erfolgt automatisch

### Option 2: Mit Custom Config
```bash
uv run voxmonitor-train --config config/my_config.yaml
```

---

## 📊 Training-Parameter

```yaml
data:
  audio_dir: /Volumes/.../Soundwel
  csv_path: /Volumes/.../SoundwelDatasetKey.csv
  sample_rate: 16000
  max_length_sec: 3.0

train:
  batch_size: 16          # Increase für schneller Training
  max_epochs: 1           # Quick test
  lr: 1e-3
  weight_decay: 1e-5
  device: auto            # GPU/MPS/CPU
  checkpoint_dir: ckpt/soundwell_quick
  export_formats: [onnx]  # Für OmniEngine
```

---

## 🔍 Expected Output

```
======================================================================
🚀 VoxMonitor Training - Local Soundwell Data (1 Epoch)
======================================================================

📂 Audio:    /Volumes/.../Soundwel
📄 CSV:      /Volumes/.../SoundwelDatasetKey.csv

📋 Loading metadata...
   Total samples: 3000+

🔧 Creating dataset...
   ✅ 3000+ samples loaded

🏷️  Classes:
   age: 4 classes
   sex: 2 classes
   valence: 3 classes
   context: 5 classes

📊 DataLoader: 188 batches of 16

🧠 Creating Lightning module...
   Classes: {'age': 4, 'sex': 2, 'valence': 3, 'context': 5}

🏋️  Creating trainer...
   ✅ Trainer ready

⏱️  Starting training...

----------------------------------------------------------------------
Epoch 1: [████████████████████] 100% Loss: 1.23
----------------------------------------------------------------------

✅ Training completed successfully!

======================================================================
✨ 1-epoch training test PASSED
======================================================================
```

---

## 📈 Output-Dateien

Nach dem Training:
```
ckpt/soundwell_quick/
├── epoch-00-val_loss=0.xxx.ckpt    # PyTorch Lightning Checkpoint
├── soundwell_final.pt                # PyTorch Model
├── soundwell.onnx                    # ONNX Export (für OmniEngine!)
└── training.log                      # Training-Logs
```

**ONNX-Modell für OmniEngine:**
- Input: Mel-Spektrogramm [1, 64, T]
- Output: Multi-Task Logits
  - age: [1, 4]
  - sex: [1, 2]
  - valence: [1, 3]
  - context: [1, 5]

---

## ✅ Checkliste vor dem Training

- [x] Data-Dateien vorhanden (/Volumes/Backup/.../Soundwell)
- [x] CSV-Metadaten verfügbar (SoundwelDatasetKey.csv)
- [x] SoundwelDataset CSV-Support implementiert
- [x] VoxMonitorTrainer basiert auf DeepSuite BaseTrainer
- [x] VoxMonitorLightningModule multi-task fertig
- [x] Registry bereinigt (nur DeepSuite)
- [x] Training-Skript erstellt (train_local.py)
- [x] ONNX-Export konfiguriert

---

## 🎓 Nächste Schritte

1. **Test ausführen:** `uv run python train_local.py`
2. **ONNX validieren:** Modell mit OmniEngine laden
3. **Full Training:** Config anpassen für alle Samples
4. **Hyperparameter tuning:** Learning rate, Batch size, etc.
5. **Inference Pipeline:** VoxMonitor → OmniEngine

---

## 📚 Referenzen

- Soundwell Dataset: https://zenodo.org/records/8252482
- Paper: https://doi.org/10.1038/s41598-022-07174-8 (Briefer et al., 2022)
- DeepSuite: Multi-task learning framework
- OmniEngine: Production inference engine

---

**Status: READY TO TRAIN** ✨
