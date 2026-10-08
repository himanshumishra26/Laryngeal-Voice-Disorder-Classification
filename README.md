# Automated Multi-Class Laryngeal Voice Disorder Classification Using Log-Mel Spectrograms and CNN-LSTM

## 1. Project Overview
This project is an independent BTech final-year research and engineering project focused on developing an automated computer-aided diagnostic (CAD) system for classifying human voice recordings into three clinical categories:
1. **Normal** (healthy control voices)
2. **Laryngozele** (pathological voice disorder caused by herniation/dilation of the laryngeal saccule)
3. **Vox senilis** (presbyphonia; voice changes associated with physiological aging of the vocal folds)

---

## 2. Research Motivation & Paradigm Shift

### Baseline / Prior Approach:
- **Feature Extraction**: Mel-Frequency Cepstral Coefficients (MFCCs). MFCCs compress acoustic energy into a compact 1D discrete cosine transform (DCT) representation, which often discards fine-grained harmonic, formant, and continuous time-frequency geometric relationships.
- **Classification Model**: 1D Convolutional Neural Network (1D CNN). While 1D CNNs can capture local temporal patterns in 1D vectors, they lack recurrent memory to track long-term phonatory drift, tremor, and sequential vocal instabilities over time.

### Proposed Independent Approach:
- **Feature Extraction: 2D Log-Mel Spectrograms**
  - Converts raw voice signals into 2D time-frequency acoustic energy distributions using the Short-Time Fourier Transform (STFT) mapped onto the non-linear human auditory Mel scale.
  - Preserves harmonic structures, spectral tilt, formant trajectories, and energy dynamics without lossy DCT compression.
- **Classification Model: CNN-LSTM Hybrid Architecture**
  - **2D CNN Subnetwork**: Functions as a spatial feature extractor over the spectrogram image, learning local invariant patterns in formant transitions and harmonic intervals.
  - **Bidirectional LSTM Subnetwork**: Functions as a temporal sequence model, capturing long-term phonatory dynamics, pitch irregularities, voice breaks, and temporal vocal cord oscillations across time.
  - **Dense Classification Head**: Maps the recurrent temporal representations to multi-class posterior class probabilities with Softmax.

---

## 3. End-to-End Pipeline Architecture

```
+---------------------------+
|    Raw Audio (.wav)       |
+-------------+-------------+
              |
              v
+---------------------------+
|    Audio Preprocessing    |
| - Channel downmix (mono)  |
| - Resampling (22,050 Hz)  |
| - Silence trimming        |
| - Fixed length padding    |
+-------------+-------------+
              |
              v
+---------------------------+
| Log-Mel Spectrogram (2D)  |
| - STFT (n_fft=1024)       |
| - Mel filterbank (n=128)  |
| - Logarithmic dB scale    |
| - Normalization           |
+-------------+-------------+
              |
              v
+---------------------------+
|    CNN-LSTM Deep Model    |
| - 2D CNN Conv Blocks      |
| - Max-Pooling & Dropout   |
| - Time-sequence Reshape   |
| - Bidirectional LSTM      |
| - Dense Classifier Head   |
+-------------+-------------+
              |
              v
+---------------------------+
|    3-Class Prediction     |
| [Normal, Laryngozele,     |
|       Vox Senilis]        |
+---------------------------+
```

---

## 4. Project Directory Structure

```text
D:\Laryngeal_Voice_Disorder_Project\
├── app/                              # Interactive web application (Streamlit)
│   ├── __init__.py
│   └── app.py                        # Diagnostic web dashboard
├── configs/                          # Experiment configuration and hyperparameters
│   └── config.yaml                   # Audio sampling rate, n_mels, batch size, etc.
├── data/                             # Dataset storage (kept uncommitted & clean)
│   ├── raw/                          # Raw voice audio recordings (.wav) by class
│   │   ├── normal/
│   │   ├── laryngozele/
│   │   └── vox_senilis/
│   ├── processed/                    # Standardized and preprocessed audio files
│   └── splits/                       # Train / Validation / Test manifest files
├── results/                          # Output artifacts generated during development
│   ├── checkpoints/                  # Trained neural network model checkpoints (.pt)
│   ├── figures/                      # Plots: spectrograms, loss curves, confusion matrices
│   └── reports/                      # Evaluation metrics (CSV / JSON classification reports)
├── src/                              # Main Python package source code
│   ├── __init__.py
│   ├── preprocessing/                # Audio loading, silence trimming, resampling, padding
│   │   ├── __init__.py
│   │   └── audio_preprocessor.py
│   ├── features/                     # Log-Mel Spectrogram feature extraction
│   │   ├── __init__.py
│   │   └── feature_extractor.py
│   ├── models/                       # CNN-LSTM deep learning model architecture
│   │   ├── __init__.py
│   │   └── cnn_lstm.py
│   ├── training/                     # Training engine, loss functions, learning rate scheduling
│   │   ├── __init__.py
│   │   └── trainer.py
│   ├── evaluation/                   # Evaluation engine, metrics calculation, visualization
│   │   ├── __init__.py
│   │   ├── evaluate.py
│   │   └── metrics.py
│   └── utils/                        # Utilities (seed reproducibility, device selection, config)
│       ├── __init__.py
│       └── utils.py
├── .gitignore                        # Git exclusion rules
├── README.md                         # Project documentation and architectural guide
└── requirements.txt                  # Python dependencies
```

---

## 5. Description of Folders and Files

| Folder / File | Purpose |
| :--- | :--- |
| `configs/config.yaml` | Centralized parameter configuration (sample rates, FFT windows, model layers, training epochs). |
| `data/raw/` | Storage for authentic raw audio files categorized into subdirectories (`normal`, `laryngozele`, `vox_senilis`). |
| `data/processed/` | Output storage for normalized, trimmed, and resampled audio waveforms. |
| `data/splits/` | Stratified train/val/test split manifests ensuring zero data leakage. |
| `src/preprocessing/` | Module containing audio I/O, mono conversion, silence elimination, and fixed-duration normalization. |
| `src/features/` | Module converting 1D audio time series into normalized 2D Log-Mel Spectrogram arrays. |
| `src/models/` | Neural network module implementing the hybrid 2D CNN + Bidirectional LSTM architecture. |
| `src/training/` | Training pipeline managing epoch iteration, backpropagation, optimizer, loss, and checkpointing. |
| `src/evaluation/` | Performance evaluation calculating multi-class confusion matrix, precision, recall, specificity, and macro F1. |
| `src/utils/` | Shared utilities including GPU/device detection, strict random seed fixing, and config parsing. |
| `results/checkpoints/`| Destination folder for best-performing model weights (`.pt`). |
| `results/figures/` | Destination folder for plots (spectrograms, training loss/accuracy curves, confusion matrices). |
| `results/reports/` | Destination folder for numerical evaluation reports (JSON/CSV). |
| `app/app.py` | Interactive diagnostic demo interface allowing end-users to upload audio and view predictions. |

---

## 6. Environment Setup

### Prerequisites
- Python 3.12.x
- NVIDIA GPU with CUDA support (Optional, but recommended. GPU detected: NVIDIA GeForce RTX 3050 Laptop GPU).

### Installation
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

---

## 7. Next Implementation Stages
- **Stage 1 (Complete)**: Environment validation, project architecture design, modular file scaffolding.
- **Stage 2**: Dataset ingestion, audio preprocessing, and Log-Mel Spectrogram feature extraction.
- **Stage 3**: CNN-LSTM model implementation and forward pass verification.
- **Stage 4**: Training pipeline execution on authentic data.
- **Stage 5**: Comprehensive multi-class evaluation and publication-grade visualization.
- **Stage 6**: Interactive diagnostic application deployment.
