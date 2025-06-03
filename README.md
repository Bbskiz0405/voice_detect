# Voice Detect

本專案為語音偵測（Voice/Non-Voice Classification），使用深度學習模型（CNN 與 FFNN）進行音訊分類。

## 專案用途與學術價值

本專案旨在設計並實作一套自動化語音偵測系統，能夠判斷音訊片段中是否包含人聲，進而將「語音」與「非語音」自動分類。此系統可廣泛應用於語音前處理（如語音辨識、語者分離）、監控錄音過濾、會議記錄整理、智慧助理啟動、語音資料標註等多種場景。

### 研究動機

隨著語音技術的普及，從龐大的音訊資料中自動篩選出含有人聲的片段，已成為語音處理領域的重要前置步驟。傳統方法多仰賴能量閾值或簡單特徵，易受雜訊干擾，準確率有限。因此，本專案採用深度學習方法，結合卷積神經網路（CNN）與前饋神經網路（FFNN），自動學習音訊特徵，提升語音偵測的準確性與魯棒性。

### 技術特色
- **多模型設計**：同時實作 CNN 與 FFNN，並比較其於語音偵測任務的效能差異。
- **資料自動前處理**：具備音檔轉換、特徵萃取（如 MFCC、Mel-spectrogram）及資料標註自動化流程。
- **模組化架構**：程式碼結構清晰，便於擴充與維護。
- **實驗可重現性**：所有訓練、測試、資料處理腳本皆可直接執行，方便學術研究與後續改良。

### 學術價值
本專案可作為語音前處理、音訊分類、深度學習聲音應用等課程或研究的實作範例。透過本專案，學生能夠學習：
- 如何設計語音分類系統
- 深度學習於聲音領域的應用
- 資料前處理與特徵工程技巧
- 模型訓練、評估與比較方法

此外，專案架構亦適合用於延伸研究，如多類別音訊事件偵測、噪音環境下的語音辨識前處理等。

---

## 目錄結構

```
voice_detect-main/
├── data/                  # 音訊與處理後資料
├── train.csv              # FFNN 訓練資料
├── train_cnn_model.py     # CNN 訓練腳本
├── train_ffnn_model.py    # FFNN 訓練腳本
├── eval_cnn_model.py      # CNN 評估腳本
├── eval_ffnn_model.py     # FFNN 評估腳本
├── prepare_cnn_dataset.py # CNN 資料前處理
├── prepare_ffnn_dataset.py# FFNN 資料前處理
└── voice_detect/
    ├── cnn/               # CNN 模型與相關程式
    ├── ffnn/              # FFNN 模型與相關程式
    ├── utils/             # 工具程式
    └── config.py          # 參數設定
```

## 安裝需求

請先安裝 Python 3.8+，並安裝以下套件：

```bash
pip install torch torchvision pandas numpy librosa matplotlib scikit-learn pydub
```

## 資料準備

1. 將原始 `.wav` 檔案放入 `data/AUDIO/voice` 與 `data/AUDIO/not_voice`。
2. 若有標註檔，請放入 `data/anotations`。

### 產生 CNN 訓練用資料（Mel-spectrogram 圖片）

```bash
python prepare_cnn_dataset.py
```

### 產生 FFNN 訓練用資料（CSV）

```bash
python prepare_ffnn_dataset.py
```

## 模型訓練

### 訓練 FFNN

```bash
python train_ffnn_model.py
```
- 預設會使用 `train.csv` 作為資料來源。

### 訓練 CNN

```bash
python train_cnn_model.py
```
- 預設會使用 `data/output_plots/train` 目錄下的圖片。

## 模型評估

### 評估 FFNN

```bash
python eval_ffnn_model.py
```

### 評估 CNN

```bash
python eval_cnn_model.py
```

## 專案說明

- `voice_detect/ffnn/` 與 `voice_detect/cnn/`：分別包含 FFNN 與 CNN 的模型、訓練、評估程式。
- `voice_detect/utils/`：資料處理、特徵萃取、音檔轉換等工具。
- 所有訓練好的模型會儲存在 `voice_detect/saved_models/`。

## 聯絡方式

作者：  
如有問題請聯絡 [你的聯絡方式]

---

# English Version

## Voice Detect

This project is for voice detection (Voice/Non-Voice Classification) using deep learning models (CNN and FFNN) for audio classification.

## Directory Structure

```
voice_detect-main/
├── data/                  # Audio and processed data
├── train.csv              # FFNN training data
├── train_cnn_model.py     # CNN training script
├── train_ffnn_model.py    # FFNN training script
├── eval_cnn_model.py      # CNN evaluation script
├── eval_ffnn_model.py     # FFNN evaluation script
├── prepare_cnn_dataset.py # CNN data preprocessing
├── prepare_ffnn_dataset.py# FFNN data preprocessing
└── voice_detect/
    ├── cnn/               # CNN model and scripts
    ├── ffnn/              # FFNN model and scripts
    ├── utils/             # Utility scripts
    └── config.py          # Configuration
```

## Requirements

Python 3.8+ is required. Install dependencies with:

```bash
pip install torch torchvision pandas numpy librosa matplotlib scikit-learn pydub
```

## Data Preparation

1. Put raw `.wav` files in `data/AUDIO/voice` and `data/AUDIO/not_voice`.
2. If you have annotation files, put them in `data/anotations`.

### Generate CNN Training Data (Mel-spectrogram Images)

```bash
python prepare_cnn_dataset.py
```

### Generate FFNN Training Data (CSV)

```bash
python prepare_ffnn_dataset.py
```

## Model Training

### Train FFNN

```bash
python train_ffnn_model.py
```
- Uses `train.csv` by default.

### Train CNN

```bash
python train_cnn_model.py
```
- Uses images in `data/output_plots/train` by default.

## Model Evaluation

### Evaluate FFNN

```bash
python eval_ffnn_model.py
```

### Evaluate CNN

```bash
python eval_cnn_model.py
```

## Project Description

- `voice_detect/ffnn/` and `voice_detect/cnn/`: FFNN and CNN models, training and evaluation scripts.
- `voice_detect/utils/`: Data processing, feature extraction, audio conversion utilities.
- Trained models are saved in `voice_detect/saved_models/`.

## Contact

Author:  
For questions, contact [your contact info]
