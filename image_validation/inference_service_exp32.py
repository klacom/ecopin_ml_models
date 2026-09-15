"""
EcoPin Image Validation — Experiment 3.2 Inference Service
===========================================================
Pipeline : image  →  CLIP ViT-B/32 (frozen)  →  512-d embedding
           →  Regularized MLP  →  P(valid)  →  VALID / INVALID / MANUAL_REVIEW

Model    : Experiment 3.2 Regularized MLP
           artifacts/exp03_2_clip_refined/reg_mlp/best.pt
           Architecture: 512 → 128 → ReLU → Dropout(0.3) → 32 → ReLU → Dropout(0.3) → 1
           Best epoch  : 21
           Val macro-F1: 0.8958   Test macro-F1: 0.9110

Classes  : VALID=1 (useful environmental evidence)
           INVALID=0 (not useful environmental evidence)

Thresholds (configurable via env vars):
  THRESHOLD        = 0.50   P(valid) >= threshold  →  VALID
  REVIEW_LO        = 0.35   P(valid) in [0.35, 0.65]  →  MANUAL_REVIEW zone
  REVIEW_HI        = 0.65

API (same conventions as inference_service.py):
  GET  /health
  POST /classify            (raw image bytes in body)
  POST /classify-multipart  (multipart/form-data, field="file")

Response JSON:
  {
    "predicted_class": "VALID" | "INVALID",
    "confidence":      0.0–1.0,               # P(valid)
    "probabilities": {
      "VALID":   <float>,
      "INVALID": <float>
    },
    "decision": "APPROVED" | "REJECTED" | "MANUAL_REVIEW",
    "in_review_zone": true | false
  }

Does NOT modify or replace inference_service.py (the 4-class EfficientNet service).
"""

import io, json, os, sys, threading, traceback, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
import torch.nn as nn
from PIL import Image

# ── CONFIG (env-var overridable) ──────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))

CHECKPOINT_PATH = os.environ.get(
    "EXP32_CHECKPOINT_PATH",
    os.path.join(_HERE, "artifacts", "exp03_2_clip_refined", "reg_mlp", "best.pt"),
)
CLIP_MODEL   = os.environ.get("EXP32_CLIP_MODEL",   "ViT-B/32")
THRESHOLD    = float(os.environ.get("EXP32_THRESHOLD",  "0.50"))
REVIEW_LO    = float(os.environ.get("EXP32_REVIEW_LO",  "0.35"))
REVIEW_HI    = float(os.environ.get("EXP32_REVIEW_HI",  "0.65"))
HOST         = os.environ.get("EXP32_HOST",  "127.0.0.1")
PORT         = int(os.environ.get("EXP32_PORT",  "8001"))
DEVICE_STR   = os.environ.get("EXP32_DEVICE", "cuda")
MAX_BODY     = int(os.environ.get("MAX_BODY_BYTES", str(20 * 1024 * 1024)))

EMB_DIM = 512

# ── DIAGNOSTIC HELPERS ────────────────────────────────────────────────────────
def _sep(c="─", w=66):
    sys.stderr.write(c * w + "\n")

def _log(msg):
    sys.stderr.write(f"[exp32] {msg}\n")

# ── REGULARIZED MLP (must match training definition exactly) ──────────────────
class RegMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 32),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

# ── STARTUP ───────────────────────────────────────────────────────────────────
_sep("═")
_log("EcoPin Binary Image Validation — Exp 3.2 Inference Service")
_sep("═")
_log(f"CLIP model     : {CLIP_MODEL}")
_log(f"Checkpoint     : {CHECKPOINT_PATH}")
_log(f"Threshold      : {THRESHOLD}")
_log(f"Review zone    : [{REVIEW_LO}, {REVIEW_HI}]")
_log(f"Port           : {PORT}")

# Device
if torch.cuda.is_available() and DEVICE_STR != "cpu":
    device = torch.device(DEVICE_STR)
    _log(f"Device         : {device} ({torch.cuda.get_device_name(0)})")
else:
    device = torch.device("cpu")
    _log(f"Device         : cpu")

# Verify checkpoint exists
if not os.path.isfile(CHECKPOINT_PATH):
    _log(f"ERROR: checkpoint not found: {CHECKPOINT_PATH}")
    sys.exit(1)
ckpt_kb = os.path.getsize(CHECKPOINT_PATH) / 1024
_log(f"Checkpoint size: {ckpt_kb:.1f} KB  ✓ exists")

# Load CLIP
_sep()
_log("Loading CLIP ViT-B/32 …")
try:
    import clip as _clip
    clip_model, clip_preprocess = _clip.load(CLIP_MODEL, device=device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)
    trainable = sum(1 for p in clip_model.parameters() if p.requires_grad)
    total_p   = sum(1 for p in clip_model.parameters())
    assert trainable == 0, f"CLIP has {trainable} trainable params — must be 0"
    _log(f"CLIP loaded    : {total_p} params, {trainable} trainable  ✓ frozen")
except Exception as exc:
    _log(f"ERROR loading CLIP: {exc}")
    sys.exit(1)

# Load RegMLP classifier
_log("Loading Regularized MLP classifier …")
try:
    classifier = RegMLP().to(device)
    state = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    classifier.load_state_dict(state)
    classifier.eval()
    # Sanity: check parameter shapes match
    w0 = state["net.0.weight"]
    assert w0.shape == (128, 512), f"Unexpected weight shape: {w0.shape}"
    _log(f"Classifier     : RegMLP 512→128→32→1  ✓ loaded")
    _log(f"  net.0.weight : {w0.shape}  (512-D CLIP embedding input confirmed)")
except Exception as exc:
    _log(f"ERROR loading classifier: {exc}")
    sys.exit(1)

# Confirm CLIP preprocessing (sanity forward pass)
_log("Running sanity forward pass …")
try:
    with torch.no_grad():
        _dummy = clip_preprocess(Image.new("RGB", (224, 224))).unsqueeze(0).to(device)
        _feat  = clip_model.encode_image(_dummy)
        _feat  = _feat / _feat.norm(dim=-1, keepdim=True)
        _prob  = torch.sigmoid(classifier(_feat.float())).item()
    _log(f"Sanity pass    : OK  embedding_dim={_feat.shape[1]}  dummy_prob={_prob:.4f}")
except Exception as exc:
    _log(f"ERROR in sanity pass: {exc}")
    sys.exit(1)

_model_lock = threading.Lock()
_sep("═")

# ── DECISION LOGIC ────────────────────────────────────────────────────────────
def _decision(prob_valid: float) -> tuple:
    """
    Returns (predicted_class, decision, in_review_zone).
    predicted_class: 'VALID' or 'INVALID'
    decision:        'APPROVED' | 'REJECTED' | 'MANUAL_REVIEW'
    """
    in_review_zone = REVIEW_LO <= prob_valid <= REVIEW_HI
    if prob_valid >= THRESHOLD:
        predicted_class = "VALID"
        decision = "MANUAL_REVIEW" if in_review_zone else "APPROVED"
    else:
        predicted_class = "INVALID"
        decision = "MANUAL_REVIEW" if in_review_zone else "REJECTED"
    return predicted_class, decision, in_review_zone

# ── INFERENCE ─────────────────────────────────────────────────────────────────
def run_inference(image_bytes: bytes) -> dict:
    t0 = time.perf_counter()

    # Stage 1: decode
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        _log(f"  Image decode : {img.size[0]}x{img.size[1]} RGB  ({len(image_bytes):,} bytes)")
    except Exception as exc:
        _log(f"  Image decode : FAILED — {exc}")
        raise RuntimeError(f"Image decode failed: {exc}") from exc

    # Stage 2: CLIP preprocess + encode
    try:
        tensor = clip_preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = clip_model.encode_image(tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            embedding = embedding.float()
        _log(f"  CLIP embed   : shape={list(embedding.shape)}")
    except Exception as exc:
        _log(f"  CLIP embed   : FAILED — {exc}")
        raise RuntimeError(f"CLIP embedding failed: {exc}") from exc

    # Stage 3: RegMLP classifier
    try:
        with torch.no_grad():
            logit = classifier(embedding)
            prob_valid = torch.sigmoid(logit).item()
        _log(f"  Classifier   : logit={logit.item():.4f}  P(valid)={prob_valid:.4f}")
    except Exception as exc:
        _log(f"  Classifier   : FAILED — {exc}")
        raise RuntimeError(f"Classifier inference failed: {exc}") from exc

    # Stage 4: decision
    predicted_class, decision, in_review_zone = _decision(prob_valid)
    prob_invalid = 1.0 - prob_valid

    elapsed_ms = (time.perf_counter() - t0) * 1000
    _log(f"  Decision     : {predicted_class}  ({decision})  "
         f"review_zone={in_review_zone}  {elapsed_ms:.1f}ms")

    return {
        "predicted_class": predicted_class,
        "confidence":      round(prob_valid, 6),
        "probabilities": {
            "VALID":   round(prob_valid,   6),
            "INVALID": round(prob_invalid, 6),
        },
        "decision":       decision,
        "in_review_zone": in_review_zone,
        "threshold":      THRESHOLD,
        "review_zone":    [REVIEW_LO, REVIEW_HI],
        "inference_ms":   round(elapsed_ms, 1),
    }

# ── HTTP HANDLER ──────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write(f"[exp32] {self.address_string()} — {fmt % args}\n")

    def _send_json(self, status, payload):
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        hdr = self.headers.get("Content-Length")
        if hdr is None:
            return None
        try:
            n = int(hdr)
        except ValueError:
            return None
        if n > MAX_BODY:
            self._send_json(413, {"error": f"Payload too large ({n} > {MAX_BODY})"})
            return None
        return self.rfile.read(n) if n > 0 else b""

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {
                "status":      "ok",
                "experiment":  "3.2",
                "model":       "RegMLP",
                "clip_model":  CLIP_MODEL,
                "threshold":   THRESHOLD,
                "review_zone": [REVIEW_LO, REVIEW_HI],
                "classes":     ["VALID", "INVALID"],
                "checkpoint":  CHECKPOINT_PATH,
            })
        else:
            self._send_json(404, {"error": "Not found"})

    def _handle_classify(self, image_bytes):
        _sep()
        _log(f"POST /classify  from {self.address_string()}")
        try:
            with _model_lock:
                result = run_inference(image_bytes)
        except RuntimeError as exc:
            _sep()
            self._send_json(500, {"error": str(exc)})
            return
        except Exception as exc:
            _log(f"  Unexpected: {exc}\n{traceback.format_exc()}")
            _sep()
            self._send_json(500, {"error": f"Inference failed: {exc}"})
            return
        _sep()
        self._send_json(200, result)

    def do_POST(self):
        if self.path == "/classify":
            body = self._read_body()
            if body is None:
                return
            if not body:
                self._send_json(400, {"error": "Empty request body"})
                return
            self._handle_classify(body)

        elif self.path == "/classify-multipart":
            body = self._read_body()
            if body is None:
                return
            ctype = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in ctype:
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
                    self._send_json(400, {"error": "Missing boundary"}); return
                delimiter = b"--" + boundary
                file_bytes = None
                for sec in body.split(delimiter):
                    if b"Content-Disposition:" in sec and b'name="file"' in sec:
                        head_end = sec.find(b"\r\n\r\n")
                        if head_end == -1:
                            head_end = sec.find(b"\n\n")
                            if head_end != -1: head_end += 2
                        else:
                            head_end += 4
                        if head_end != -1:
                            payload = sec[head_end:]
                            for suffix in (b"\r\n--", b"--", b"\r\n"):
                                if payload.endswith(suffix):
                                    payload = payload[:-len(suffix)]
                            file_bytes = payload
                            break
                if not file_bytes:
                    self._send_json(400, {"error": "Missing 'file' field"}); return
                self._handle_classify(file_bytes)
            except Exception as exc:
                self._send_json(500, {"error": f"Request handling failed: {exc}"})

        else:
            self._send_json(404, {"error": "Not found"})

# ── ENTRY POINT ───────────────────────────────────────────────────────────────
def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    _log(f"Listening on http://{HOST}:{PORT}")
    _log(f"  GET  /health")
    _log(f"  POST /classify            (raw image bytes)")
    _log(f"  POST /classify-multipart  (multipart/form-data, field='file')")
    _sep("═")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log("Shutting down.")
        server.server_close()

if __name__ == "__main__":
    main()
