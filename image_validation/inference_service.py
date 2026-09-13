import os
import io
import sys
import json
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from timm import create_model
from PIL import Image

VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_vendor")
if os.path.isdir(VENDOR_DIR) and VENDOR_DIR not in sys.path:
    sys.path.insert(0, VENDOR_DIR)

MODEL_CHECKPOINT_PATH = os.environ.get(
    "MODEL_CHECKPOINT_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "exp2_2", "best_model.pt"),
)
MODEL_NAME = os.environ.get("MODEL_NAME", "efficientnet_b0")
NUM_CLASSES = int(os.environ.get("NUM_CLASSES", "4"))
CLASS_NAMES = os.environ.get(
    "CLASS_NAMES",
    "flooding,non_environmental,pollution,waste",
).split(",")
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "224"))
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
DEVICE_STR = os.environ.get("DEVICE", "cpu")
MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", str(20 * 1024 * 1024)))
# Mirrors the backend CLASSIFIER_HIGH_CONFIDENCE default (diagnostics only — does not affect the API response)
HIGH_CONFIDENCE_THRESHOLD = float(os.environ.get("CLASSIFIER_HIGH_CONFIDENCE", "0.70"))

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def _sep(char="─", width=64):
    """Print a separator line to stderr."""
    sys.stderr.write(char * width + "\n")


def _diag(msg):
    """Write a single diagnostic line to stderr."""
    sys.stderr.write(f"[diag] {msg}\n")


# ---------------------------------------------------------------------------
# Preprocessing — UNCHANGED from original
# ---------------------------------------------------------------------------

def _pad_to_square(img):
    w, h = img.size
    max_side = max(w, h)
    pad_left = (max_side - w) // 2
    pad_top = (max_side - h) // 2
    pad_right = max_side - w - pad_left
    pad_bottom = max_side - h - pad_top
    return TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=0)


def build_inference_transform(image_size):
    return T.Compose([
        T.Lambda(_pad_to_square),
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Startup — model loading with detailed diagnostics
# ---------------------------------------------------------------------------

_sep("═")
sys.stderr.write("[classifier] EcoPin Image Validation Classifier — Starting Up\n")
_sep("═")
sys.stderr.write(f"[classifier] Architecture   : {MODEL_NAME}\n")
sys.stderr.write(f"[classifier] Num classes    : {NUM_CLASSES}\n")
sys.stderr.write(f"[classifier] Class order    : {CLASS_NAMES}\n")
sys.stderr.write(f"[classifier] Image size     : {IMAGE_SIZE} × {IMAGE_SIZE}\n")
sys.stderr.write(f"[classifier] Preprocessing  : PadToSquare(fill=0) → Resize({IMAGE_SIZE}×{IMAGE_SIZE}) "
                 f"→ ToTensor → ImageNet Normalize\n")
sys.stderr.write(f"[classifier]                  mean={IMAGENET_MEAN}\n")
sys.stderr.write(f"[classifier]                  std ={IMAGENET_STD}\n")
sys.stderr.write(f"[classifier] Conf. threshold: {HIGH_CONFIDENCE_THRESHOLD:.2f} "
                 f"(backend threshold, shown in diagnostics only)\n")
_sep()

# Device selection (same logic as original)
if torch.cuda.is_available() and DEVICE_STR != "cpu":
    device = torch.device(DEVICE_STR)
    sys.stderr.write(f"[classifier] Device         : {device} (CUDA GPU)\n")
else:
    device = torch.device("cpu")
    if DEVICE_STR != "cpu":
        sys.stderr.write(
            f"[classifier] Device         : cpu  "
            f"(CUDA was requested via DEVICE={DEVICE_STR!r} but is unavailable — fell back to CPU)\n"
        )
    else:
        sys.stderr.write(f"[classifier] Device         : cpu\n")

# Checkpoint path — printed explicitly so there is never any ambiguity about which file is loaded
sys.stderr.write(f"[classifier] Checkpoint     : {MODEL_CHECKPOINT_PATH}\n")
if os.path.isfile(MODEL_CHECKPOINT_PATH):
    ckpt_mb = os.path.getsize(MODEL_CHECKPOINT_PATH) / (1024 * 1024)
    sys.stderr.write(f"[classifier] Checkpoint size: {ckpt_mb:.1f} MB  ✓ file exists\n")
else:
    sys.stderr.write(f"[classifier] Checkpoint     : ✗ FILE NOT FOUND — startup will fail\n")

_sep()
sys.stderr.write("[classifier] Loading model weights...\n")

model = create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
_raw = torch.load(MODEL_CHECKPOINT_PATH, map_location=device, weights_only=False)

# Detect checkpoint format and extract state_dict
if isinstance(_raw, dict) and "model_state_dict" in _raw:
    # Checkpoint dictionary format (Exp 02 and later)
    state_dict = _raw["model_state_dict"]
    sys.stderr.write("[classifier] Checkpoint format: dictionary (model_state_dict key found)\n")
    _ckpt_epoch      = _raw.get("epoch", "unknown")
    _ckpt_macro_f1   = _raw.get("best_macro_f1")
    _ckpt_val_acc    = _raw.get("best_val_acc")
    _ckpt_base_ckpt  = _raw.get("base_checkpoint", "unknown")
    _ckpt_cls        = _raw.get("class_names")
    sys.stderr.write(f"[classifier]   Saved epoch    : {_ckpt_epoch}\n")
    if _ckpt_macro_f1 is not None:
        sys.stderr.write(f"[classifier]   Best Macro F1  : {_ckpt_macro_f1:.4f}\n")
    if _ckpt_val_acc is not None:
        sys.stderr.write(f"[classifier]   Best Val Acc   : {_ckpt_val_acc * 100:.2f}%\n")
    sys.stderr.write(f"[classifier]   Base checkpoint: {_ckpt_base_ckpt}\n")
    if _ckpt_cls is not None:
        sys.stderr.write(f"[classifier]   Embedded classes: {_ckpt_cls}\n")
        if _ckpt_cls != CLASS_NAMES:
            sys.stderr.write(
                f"[classifier] ⚠ CLASS NAME MISMATCH — checkpoint has {_ckpt_cls}, "
                f"service expects {CLASS_NAMES}\n"
            )
        else:
            sys.stderr.write("[classifier]   Class names match service config  ✓\n")
    else:
        sys.stderr.write("[classifier]   Embedded classes: (not stored in checkpoint)\n")
else:
    # Raw state_dict format (Exp 8 / legacy)
    state_dict = _raw
    sys.stderr.write("[classifier] Checkpoint format: raw state_dict (legacy format)\n")

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

transform = build_inference_transform(IMAGE_SIZE)
_model_lock = threading.Lock()

sys.stderr.write("[classifier] Checkpoint loaded into model  ✓\n")
sys.stderr.write("[classifier] Model set to eval() mode      ✓\n")
_sep("═")


# ---------------------------------------------------------------------------
# Decision logic (mirrors classifierClient.service.js — diagnostics only)
# ---------------------------------------------------------------------------

_ENVIRONMENTAL_CLASSES = {"flooding", "pollution", "waste"}


def _backend_decision(predicted_class, confidence):
    """
    Reproduce the backend's classification-to-validation-status mapping.
    Used ONLY for terminal diagnostics. Does NOT affect the JSON API response.
    """
    high = confidence >= HIGH_CONFIDENCE_THRESHOLD
    if predicted_class in _ENVIRONMENTAL_CLASSES and high:
        return "APPROVED"
    if predicted_class == "non_environmental" and high:
        return "REJECTED"
    return "MANUAL_REVIEW"


# ---------------------------------------------------------------------------
# Inference with step-by-step diagnostics
# ---------------------------------------------------------------------------

def run_inference(image_bytes):
    """
    Run preprocessing + model inference. Returns the same dict as the original.
    Diagnostic output goes to stderr only — the JSON API response is unchanged.

    Raises RuntimeError with a descriptive message if any stage fails, allowing
    the caller to distinguish:
      • image decode failure
      • preprocessing failure
      • model inference failure
    """

    # ── Stage 1: Image decode ───────────────────────────────────────────────
    try:
        img_buf = io.BytesIO(image_bytes)
        image = Image.open(img_buf)
        img_format = image.format or "unknown"
        orig_w, orig_h = image.size
        orig_mode = image.mode
        image = image.convert("RGB")
        _diag("Image decode    : OK")
        _diag(f"  Raw bytes     : {len(image_bytes):,} bytes")
        _diag(f"  Format        : {img_format}")
        _diag(f"  Original size : {orig_w} × {orig_h}  mode={orig_mode}")
        _diag(f"  After convert : {image.size[0]} × {image.size[1]}  mode=RGB")
    except Exception as exc:
        _diag(f"Image decode    : ✗ FAILED — {exc}")
        raise RuntimeError(f"Image decode failed: {exc}") from exc

    # ── Stage 2: Preprocessing ──────────────────────────────────────────────
    try:
        # Show what PadToSquare will add (computed here for logging; transform does the actual work)
        max_side = max(orig_w, orig_h)
        pad_l = (max_side - orig_w) // 2
        pad_t = (max_side - orig_h) // 2
        pad_r = max_side - orig_w - pad_l
        pad_b = max_side - orig_h - pad_t

        tensor = transform(image).unsqueeze(0).to(device)
        t_cpu = tensor.cpu()

        _diag("Preprocessing   :")
        _diag(f"  PadToSquare   : left={pad_l} top={pad_t} right={pad_r} bottom={pad_b}  →  {max_side}×{max_side}")
        _diag(f"  Resize        : {max_side}×{max_side}  →  {IMAGE_SIZE}×{IMAGE_SIZE}")
        _diag(f"  Normalize     : ImageNet mean/std")
        _diag(f"  Tensor shape  : {list(tensor.shape)}  dtype={tensor.dtype}")
        _diag(f"  Tensor stats  : min={t_cpu.min().item():.4f}  "
              f"max={t_cpu.max().item():.4f}  "
              f"mean={t_cpu.mean().item():.4f}")
    except Exception as exc:
        _diag(f"Preprocessing   : ✗ FAILED — {exc}")
        raise RuntimeError(f"Preprocessing failed: {exc}") from exc

    # ── Stage 3: Model inference ────────────────────────────────────────────
    try:
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        _diag("Inference       : OK")
    except Exception as exc:
        _diag(f"Inference       : ✗ FAILED — {exc}")
        raise RuntimeError(f"Model inference failed: {exc}") from exc

    # ── Stage 4: Results & decision ─────────────────────────────────────────
    idx = int(probs.argmax())
    predicted_class = CLASS_NAMES[idx]
    confidence = float(probs[idx])
    decision = _backend_decision(predicted_class, confidence)

    _diag("Probabilities   :")
    for i, cls in enumerate(CLASS_NAMES):
        marker = "  ◀ predicted" if i == idx else ""
        _diag(f"  {cls:<22}: {probs[i] * 100:6.2f}%{marker}")
    _diag(f"Predicted class : {predicted_class}")
    _diag(f"Confidence      : {confidence * 100:.2f}%  (threshold {HIGH_CONFIDENCE_THRESHOLD:.0%})")
    _diag(f"Backend decision: {decision}")

    # Return value is identical to original — API contract unchanged
    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
    }


# ---------------------------------------------------------------------------
# HTTP handler — UNCHANGED logic, diagnostics added
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("[classifier] %s - %s\n" % (self.address_string(), fmt % args))

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length_header = self.headers.get("Content-Length")
        if length_header is None:
            return None
        try:
            length = int(length_header)
        except ValueError:
            return None
        if length > MAX_BODY_BYTES:
            self._send_json(413, {"error": f"Payload too large ({length} > {MAX_BODY_BYTES})"})
            return None
        if length <= 0:
            return b""
        return self.rfile.read(length)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "model": MODEL_NAME, "classes": CLASS_NAMES})
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path == "/classify":
            _sep()
            _diag(f"POST /classify  from {self.address_string()}")
            _diag(f"  Content-Type  : {self.headers.get('Content-Type', '(none)')}")

            body = self._read_body()
            if body is None:
                _diag("  Body read     : ✗ Missing or oversized Content-Length — request rejected")
                return
            if not body:
                _diag("  Body read     : ✗ Empty body — returning 400")
                self._send_json(400, {"error": "Empty request body"})
                return
            _diag(f"  Body size     : {len(body):,} bytes")

            try:
                with _model_lock:
                    result = run_inference(body)
            except RuntimeError as exc:
                # run_inference already logged the stage that failed
                _diag(f"  Result        : ✗ ERROR — {exc}")
                _sep()
                self._send_json(500, {"error": str(exc)})
                return
            except Exception as exc:
                _diag(f"  Result        : ✗ UNEXPECTED ERROR — {exc}")
                _diag(traceback.format_exc())
                _sep()
                self._send_json(500, {"error": f"Inference failed: {exc}"})
                return

            _diag("  Result        : 200 OK — response sent")
            _sep()
            self._send_json(200, result)

        elif self.path == "/classify-multipart":
            _sep()
            _diag(f"POST /classify-multipart  from {self.address_string()}")

            body = self._read_body()
            if body is None:
                _diag("  Body read     : ✗ Missing or oversized Content-Length — request rejected")
                return
            ctype = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in ctype:
                _diag(f"  Content-Type  : ✗ Expected multipart/form-data, got: {ctype!r}")
                self._send_json(400, {"error": "Expected multipart/form-data"})
                return
            try:
                boundary = None
                for part in ctype.split(";"):
                    part = part.strip()
                    if part.lower().startswith("boundary="):
                        boundary = part.split("=", 1)[1].strip().strip('"').encode("ascii")
                        break
                if not boundary:
                    _diag("  Boundary      : ✗ Missing boundary in Content-Type")
                    self._send_json(400, {"error": "Missing boundary in Content-Type"})
                    return

                delimiter = b"--" + boundary
                sections = body.split(delimiter)
                file_bytes = None
                for sec in sections:
                    if b"Content-Disposition:" in sec and b'name="file"' in sec:
                        head_end = sec.find(b"\r\n\r\n")
                        if head_end == -1:
                            head_end = sec.find(b"\n\n")
                            if head_end != -1:
                                head_end += 2
                        else:
                            head_end += 4
                        if head_end != -1:
                            payload = sec[head_end:]
                            if payload.endswith(b"\r\n"):
                                payload = payload[:-2]
                            if payload.endswith(b"--"):
                                payload = payload[:-2]
                            if payload.endswith(b"\r\n"):
                                payload = payload[:-2]
                            file_bytes = payload
                            break
                if not file_bytes:
                    _diag("  Form field    : ✗ 'file' field not found in multipart body")
                    self._send_json(400, {"error": "Missing 'file' field"})
                    return

                _diag(f"  Multipart     : {len(file_bytes):,} bytes extracted from 'file' field")
                with _model_lock:
                    result = run_inference(file_bytes)

            except RuntimeError as exc:
                _diag(f"  Result        : ✗ ERROR — {exc}")
                _sep()
                self._send_json(500, {"error": str(exc)})
                return
            except Exception as exc:
                _diag(f"  Result        : ✗ UNEXPECTED ERROR — {exc}")
                _diag(traceback.format_exc())
                _sep()
                self._send_json(500, {"error": f"Request handling failed: {exc}"})
                return

            _diag("  Result        : 200 OK — response sent")
            _sep()
            self._send_json(200, result)

        else:
            self._send_json(404, {"error": "Not found"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    sys.stderr.write(f"[classifier] Listening on http://{HOST}:{PORT}\n")
    sys.stderr.write(f"[classifier]   GET  /health\n")
    sys.stderr.write(f"[classifier]   POST /classify           (raw image bytes)\n")
    sys.stderr.write(f"[classifier]   POST /classify-multipart (multipart, field='file')\n")
    _sep("═")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        sys.stderr.write("\n[classifier] Shutting down.\n")
        server.server_close()


if __name__ == "__main__":
    main()
