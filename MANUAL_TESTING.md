# EcoPin Image Validation — Manual Testing Guide

> **For:** Developers and teammates who want to run and test the classifier service locally.
> **Level:** Beginner-friendly — assumes you have never run this service before.

---

## Table of Contents

1. [What the service does](#1-what-the-service-does)
2. [Prerequisites](#2-prerequisites)
3. [Getting the ML repository](#3-getting-the-ml-repository)
4. [Python virtual environment](#4-python-virtual-environment)
5. [Installing dependencies](#5-installing-dependencies)
6. [Environment variables](#6-environment-variables)
7. [Starting the classifier service](#7-starting-the-classifier-service)
8. [API reference](#8-api-reference)
9. [Manual image testing](#9-manual-image-testing)
10. [End-to-end backend testing](#10-end-to-end-backend-testing)
11. [Troubleshooting](#11-troubleshooting)
12. [Fresh machine setup (quick reference)](#12-fresh-machine-setup-quick-reference)

---

## 1. What the service does

The classifier service is a lightweight HTTP server that:

1. **Accepts an uploaded image** (JPEG, PNG, WebP, or any PIL-supported format).
2. **Runs it through an EfficientNet-B0 neural network** trained on the EcoPin image dataset.
3. **Returns a prediction** — one of four classes, with a confidence score.

### Output classes

| Class | Meaning |
|---|---|
| `flooding` | Image shows a flooding / water hazard event |
| `pollution` | Image shows air, water, or land pollution |
| `waste` | Image shows illegal waste / garbage dumping |
| `non_environmental` | Image is not an environmental concern |

### How the EcoPin backend uses this

When a resident submits a report through the EcoPin app, the backend automatically sends the attached image to this service. Based on the prediction and the confidence threshold (default `0.70`):

| Result | Action |
|---|---|
| Environmental class + confidence ≥ 0.70 | Report → **APPROVED** automatically |
| `non_environmental` + confidence ≥ 0.70 | Report → **REJECTED** automatically |
| Any class + confidence < 0.70 | Report → **MANUAL REVIEW** queue |

---

## 2. Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.10+** | Tested with Python 3.10 and 3.11 |
| **Git** | To clone the repository |
| **~2 GB disk space** | For PyTorch and model weights |
| **CPU** | GPU is **not required**. The service defaults to CPU inference. |
| **NVIDIA GPU + CUDA** | Optional. Only if you explicitly set `DEVICE=cuda`. |
| **Internet access** | Needed once during `pip install` to download packages. |

> **No GPU required.** CPU inference is the default and works correctly for all four classes.

---

## 3. Getting the ML repository

Clone the repository to your machine:

```powershell
git clone <repository-url> ecopin_ml_models
cd ecopin_ml_models
```

You only need the `image_validation/` folder to run the classifier service:

```
image_validation/
├── inference_service.py    ← The HTTP server / entry point
├── requirements.txt        ← Runtime Python dependencies
├── .env.example            ← Environment variable template
├── checkpoints/
│   └── exp8/
│       └── best.pt         ← Default model checkpoint (Exp 8, ~350-image dataset)
└── _vendor/                ← Bundled packages (loaded automatically)
```

> **You do NOT need:**
> - The `c:\dev\datasets\ecopin_dataset\` training/split dataset
> - The `final_1000_dataset/` experiment directories
> - The `archive_original_350_dataset/` directory
> - Any training scripts

### Which model checkpoint to use

| Checkpoint | Location | Dataset | Val Acc | Test Acc |
|---|---|---|---|---|
| **Exp 8 (default)** | `checkpoints/exp8/best.pt` | ~350 images | 69.81% | — |
| **Exp 02 (recommended)** | `final_1000_dataset/experiments/experiment_02_baseline_finetune/best_model.pt` | 1,000 images | 74.00% | 62.00% |

The service defaults to Exp 8 unless you override `MODEL_CHECKPOINT_PATH` (see §6).

---

## 4. Python virtual environment

Open **PowerShell** from the repository root (or from `image_validation/`):

```powershell
# Navigate to the inference service directory
cd image_validation

# Create a virtual environment named .venv
python -m venv .venv

# Activate it
.venv\Scripts\Activate.ps1
```

You should see `(.venv)` appear in your prompt. **All subsequent commands assume the venv is active.**

> **Execution policy error?** If PowerShell blocks the script, run:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```
> Then try activating again.

---

## 5. Installing dependencies

The `requirements.txt` lists only the runtime inference dependencies (no training libraries).

### Option A — CPU-only install (recommended for most teammates)

```powershell
# Install PyTorch CPU build first (smaller download, ~200 MB vs ~2 GB for CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install the remaining inference dependencies
pip install timm Pillow numpy
```

### Option B — GPU install (only if you have an NVIDIA GPU)

```powershell
# CUDA 12.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm Pillow numpy
```

### Option C — Let pip resolve automatically

```powershell
pip install -r requirements.txt
```

> Option C installs the generic `torch` build, which may pull in CUDA even if you do not need it.
> Option A is preferred for machines without a GPU.

### Verify installation

```powershell
python -c "import torch, torchvision, timm, PIL; print('All imports OK')"
```

Expected output:
```
All imports OK
```

---

## 6. Environment variables

The service reads all configuration from environment variables. None are required — the defaults work for local testing.

### Supported variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_CHECKPOINT_PATH` | `checkpoints/exp8/best.pt` (relative to `inference_service.py`) | Path to the EfficientNet-B0 `.pt` checkpoint |
| `MODEL_NAME` | `efficientnet_b0` | Model architecture name (timm registry) |
| `NUM_CLASSES` | `4` | Number of output classes |
| `CLASS_NAMES` | `flooding,non_environmental,pollution,waste` | Comma-separated class labels (order must match training) |
| `IMAGE_SIZE` | `224` | Input image size in pixels (square) |
| `HOST` | `127.0.0.1` | Bind address |
| `PORT` | `8000` | Listening port |
| `DEVICE` | `cpu` | Compute device (`cpu` or `cuda`) |
| `MAX_BODY_BYTES` | `20971520` (20 MB) | Maximum accepted request body size |

### Using the recommended Exp 02 model

Copy `.env.example` to `.env` and set the model path:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and uncomment/set:

```env
MODEL_CHECKPOINT_PATH=C:\dev\ecopin_ml_models\final_1000_dataset\experiments\experiment_02_baseline_finetune\best_model.pt
```

> **Adjust this path to wherever the checkpoint lives on your machine.**
> The checkpoint file is **not** committed to the repository and must be obtained separately
> (see the Git Cleanup Report for the distribution strategy).

### Setting the variable without a .env file

You can also set the variable directly in PowerShell before starting the service:

```powershell
$env:MODEL_CHECKPOINT_PATH = "C:\dev\ecopin_ml_models\final_1000_dataset\experiments\experiment_02_baseline_finetune\best_model.pt"
```

---

## 7. Starting the classifier service

> Make sure your virtual environment is active and you are inside `image_validation/`.

```powershell
python inference_service.py
```

### Expected startup output

```
[classifier] Loading model: efficientnet_b0
[classifier] Checkpoint: checkpoints\exp8\best.pt
[classifier] Device: cpu
[classifier] Classes: ['flooding', 'non_environmental', 'pollution', 'waste']
[classifier] Model loaded successfully.
[classifier] Listening on http://127.0.0.1:8000
```

> If you set `MODEL_CHECKPOINT_PATH`, the checkpoint line will show that path instead.
> Loading on CPU takes 1–5 seconds. The service is ready as soon as "Listening on" appears.

### Stopping the service

Press `Ctrl+C`.

---

## 8. API reference

> **Important:** This service uses Python's built-in HTTP server — it is **NOT** a FastAPI application.
> There is **no `/docs` Swagger UI**. Use the endpoints below directly.

### `GET /health`

Returns the model and class configuration. Use this to verify the service is running.

```powershell
Invoke-WebRequest -Uri http://127.0.0.1:8000/health | Select-Object -ExpandProperty Content
```

Or with curl (if installed):

```bash
curl http://127.0.0.1:8000/health
```

**Response:**

```json
{
  "status": "ok",
  "model": "efficientnet_b0",
  "classes": ["flooding", "non_environmental", "pollution", "waste"]
}
```

---

### `POST /classify`

**Primary classification endpoint.** Send raw image bytes as the request body.

This is the endpoint the EcoPin backend uses.

| Field | Value |
|---|---|
| Method | `POST` |
| URL | `http://127.0.0.1:8000/classify` |
| Body | Raw image bytes (JPEG, PNG, WebP, etc.) |
| Content-Type | Any (e.g. `image/jpeg`) |
| Max body size | 20 MB (default) |

**Response:**

```json
{
  "predicted_class": "flooding",
  "confidence": 0.8723,
  "probabilities": {
    "flooding": 0.8723,
    "non_environmental": 0.0441,
    "pollution": 0.0512,
    "waste": 0.0324
  }
}
```

| Field | Description |
|---|---|
| `predicted_class` | The class with the highest probability |
| `confidence` | Probability of the predicted class (0.0 – 1.0) |
| `probabilities` | Softmax probability for all four classes |

---

### `POST /classify-multipart`

Accepts `multipart/form-data` with the image in a field named **`file`**.

Useful for manual testing with tools that send form uploads (e.g. Postman, HTML forms).

| Field | Value |
|---|---|
| Method | `POST` |
| URL | `http://127.0.0.1:8000/classify-multipart` |
| Content-Type | `multipart/form-data` |
| Form field name | `file` |

```powershell
# Using curl (if available)
curl -X POST http://127.0.0.1:8000/classify-multipart -F "file=@C:\path\to\test_image.jpg"
```

**Response:** Same JSON structure as `/classify`.

---

## 9. Manual image testing

### Using PowerShell (raw POST to /classify)

```powershell
# Read image as raw bytes and POST to /classify
$bytes   = [System.IO.File]::ReadAllBytes("C:\path\to\test_image.jpg")
$uri     = "http://127.0.0.1:8000/classify"
$headers = @{ "Content-Type" = "image/jpeg" }

$response = Invoke-WebRequest -Uri $uri -Method POST -Body $bytes -Headers $headers
$response.Content | ConvertFrom-Json
```

### Using Python

```python
import requests

with open(r"C:\path\to\test_image.jpg", "rb") as f:
    image_bytes = f.read()

response = requests.post(
    "http://127.0.0.1:8000/classify",
    data=image_bytes,
    headers={"Content-Type": "image/jpeg"},
)
result = response.json()
print(f"Predicted: {result['predicted_class']} ({result['confidence']:.2%} confidence)")
print(f"All probabilities: {result['probabilities']}")
```

### Recommended test cases

Use these categories to exercise and observe model behavior. Test images can be found anywhere in `c:\dev\datasets\ecopin_dataset\raw\` if available locally, or use your own photos.

| Test image | Expected class | Folder hint |
|---|---|---|
| Flooded street or waterlogged road | `flooding` | `raw/flooding/` |
| Pile of illegally dumped garbage | `waste` | `raw/waste/` |
| Smoke from a factory or burning pile | `pollution` | `raw/pollution/` |
| Normal street scene, no hazard | `non_environmental` | `raw/non_environmental/` |
| Muddy river (ambiguous: flooding or pollution?) | Either — observe confidence | — |

> **Goal:** Observe model behavior, not to judge correctness from a single prediction.
> A low confidence score (< 0.70) means the model is uncertain — the backend will route that report to manual review.

### Interpreting results

```
predicted_class = "waste"
confidence      = 0.8723   → Backend action: APPROVED (≥ 0.70, environmental)
confidence      = 0.5500   → Backend action: MANUAL REVIEW (< 0.70)

predicted_class = "non_environmental"
confidence      = 0.9100   → Backend action: REJECTED (high confidence non-env)
confidence      = 0.4800   → Backend action: MANUAL REVIEW
```

---

## 10. End-to-end backend testing

The EcoPin backend (`ecopin_backend_node`) connects to this service automatically when `CLASSIFIER_SERVICE_URL` is set.

### Backend environment variable

In `ecopin_backend_node/.env`:

```env
CLASSIFIER_SERVICE_URL=http://127.0.0.1:8000
CLASSIFIER_HIGH_CONFIDENCE=0.70
```

The backend calls `POST http://127.0.0.1:8000/classify` with the raw image buffer when a report with an image is submitted.

### Startup order

Start services in this order:

1. **Classifier service** (this guide)
   ```powershell
   # In image_validation/ with .venv active
   python inference_service.py
   ```

2. **EcoPin backend**
   ```powershell
   # In ecopin_backend_node/
   npm run dev
   ```

3. **EcoPin frontend / mobile app**
   ```powershell
   # In ecopin_app/
   npm run dev   # or expo start
   ```

4. **Submit a report** with an image through the app or frontend.

5. **Observe** the report's validation status in the database or admin panel.
   - `APPROVED` — image classified as environmental with ≥ 70% confidence
   - `REJECTED` — image classified as non-environmental with ≥ 70% confidence
   - `MANUAL_REVIEW` — model was uncertain (confidence < 70%)

### Verifying the backend → classifier connection

After starting both services, check the classifier health from within the backend environment:

```powershell
Invoke-WebRequest -Uri http://127.0.0.1:8000/health | Select-Object -ExpandProperty Content
```

If you see `{"status":"ok",...}`, the backend can reach the classifier.

---

## 11. Troubleshooting

### `Connection refused` when calling the API

The classifier service is not running. Start it with:
```powershell
python inference_service.py
```

### Port 8000 already in use

Find and stop the process using the port:
```powershell
# Find what's using port 8000
netstat -ano | findstr :8000

# Kill by PID (replace 1234 with the actual PID)
taskkill /PID 1234 /F
```

Or change the classifier port:
```powershell
$env:PORT = "8001"
python inference_service.py
```
Then update `CLASSIFIER_SERVICE_URL=http://127.0.0.1:8001` in the backend `.env`.

### `python` not found

Ensure Python 3.10+ is installed and on your PATH:
```powershell
python --version
```
If this fails, [download Python from python.org](https://www.python.org/downloads/) and ensure "Add to PATH" is checked during installation.

### Missing package error (`ModuleNotFoundError`)

Make sure your virtual environment is active (you should see `(.venv)` in the prompt):
```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### `torch` import failure / DLL error on Windows

This often means the wrong PyTorch build was installed. Reinstall the CPU build:
```powershell
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Checkpoint not found

```
FileNotFoundError: [Errno 2] No such file or directory: 'checkpoints\exp8\best.pt'
```

The model checkpoint file is missing. Either:
- Obtain `best.pt` from your team and place it at `image_validation/checkpoints/exp8/best.pt`, OR
- Set `MODEL_CHECKPOINT_PATH` to point to the Exp 02 checkpoint:
  ```powershell
  $env:MODEL_CHECKPOINT_PATH = "C:\path\to\best_model.pt"
  ```

### Invalid checkpoint architecture

```
RuntimeError: Error(s) in loading state_dict
```

The checkpoint was saved with a different model architecture or number of classes. Verify that `MODEL_NAME=efficientnet_b0` and `NUM_CLASSES=4` match the checkpoint.

### Model running on CPU but you expected GPU

Check the startup log:
```
[classifier] Device: cpu
```
This is correct for CPU inference. To use a GPU, set:
```powershell
$env:DEVICE = "cuda"
```
Note: CUDA PyTorch must be installed (Option B in §5).

### CUDA unavailable (after setting `DEVICE=cuda`)

If no CUDA-capable GPU is detected, the service automatically falls back to CPU. Check the log:
```
[classifier] Device: cpu
```
This is safe — inference will still work.

### Image upload failure (400 Bad Request from `/classify-multipart`)

Ensure:
- The form field is named exactly **`file`** (case-sensitive).
- The `Content-Type` header includes `multipart/form-data` with a valid boundary.

### Backend reports `CLASSIFIER_SERVICE_URL is not configured`

Set the variable in `ecopin_backend_node/.env`:
```env
CLASSIFIER_SERVICE_URL=http://127.0.0.1:8000
```

### Backend cannot reach classifier (timeout / ECONNREFUSED)

- Confirm the classifier service is running and listening on port 8000.
- Confirm `CLASSIFIER_SERVICE_URL` matches the host/port.
- The default timeout is 30 seconds — if inference is extremely slow on CPU, this should still be sufficient for a single image.

### `/docs` not found (404)

The current classifier service is **not** built on FastAPI and does not expose a Swagger UI. Use the `GET /health` endpoint to verify the service is running, and use the API reference in §8 or tools like Postman for manual testing.

---

## 12. Fresh machine setup (quick reference)

> A teammate should be able to follow these steps from a fresh clone and start classifying images.

```powershell
# 1. Clone the repo
git clone <repository-url> ecopin_ml_models
cd ecopin_ml_models\image_validation

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install CPU-only PyTorch (fastest, no GPU required)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install timm Pillow numpy

# 4. (Optional) Point to the Exp 02 model
# Set MODEL_CHECKPOINT_PATH in .env or as an environment variable.
# See .env.example for details.

# 5. Start the service
python inference_service.py

# 6. Verify it's running (in a separate terminal)
Invoke-WebRequest -Uri http://127.0.0.1:8000/health | Select-Object -ExpandProperty Content
# Expected: {"status":"ok","model":"efficientnet_b0","classes":[...]}

# 7. Test with an image
$bytes = [System.IO.File]::ReadAllBytes("C:\path\to\test_image.jpg")
(Invoke-WebRequest -Uri http://127.0.0.1:8000/classify -Method POST -Body $bytes -Headers @{"Content-Type"="image/jpeg"}).Content
```

---

*Model: EfficientNet-B0 trained on the EcoPin 1,000-image dataset (Experiment 02 — fine-tuned from historical Exp 8 baseline).*
*Best validation accuracy: 74.00% | Best validation Macro F1: 0.7387 | Test accuracy: 62.00% | Test Macro F1: 0.6105*
