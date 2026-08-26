import os
import io
import sys
import json
import threading
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
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "exp8", "best.pt"),
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

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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


print(f"[classifier] Loading model: {MODEL_NAME}")
print(f"[classifier] Checkpoint: {MODEL_CHECKPOINT_PATH}")
if torch.cuda.is_available() and DEVICE_STR != "cpu":
    device = torch.device(DEVICE_STR)
else:
    device = torch.device("cpu")
print(f"[classifier] Device: {device}")
print(f"[classifier] Classes: {CLASS_NAMES}")

model = create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
state_dict = torch.load(MODEL_CHECKPOINT_PATH, map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

transform = build_inference_transform(IMAGE_SIZE)
_model_lock = threading.Lock()

print(f"[classifier] Model loaded successfully.")


def run_inference(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(probs.argmax())
    return {
        "predicted_class": CLASS_NAMES[idx],
        "confidence": float(probs[idx]),
        "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
    }


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
            body = self._read_body()
            if body is None:
                return
            if not body:
                self._send_json(400, {"error": "Empty request body"})
                return
            try:
                with _model_lock:
                    result = run_inference(body)
            except Exception as e:
                self._send_json(500, {"error": f"Inference failed: {e}"})
                return
            self._send_json(200, result)
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
                    self._send_json(400, {"error": "Missing 'file' field"})
                    return
                with _model_lock:
                    result = run_inference(file_bytes)
            except Exception as e:
                self._send_json(500, {"error": f"Request handling failed: {e}"})
                return
            self._send_json(200, result)
        else:
            self._send_json(404, {"error": "Not found"})


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"[classifier] Listening on http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[classifier] Shutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
