"""Local web dashboard for chatting with a trained gowtham-gpt checkpoint.

Usage:
    python webapp/app.py
    open http://127.0.0.1:5001
"""

import sys
from pathlib import Path

# Let this script import the top-level package (model/, data/, generate.py, ...)
# whether it's launched as `python webapp/app.py` or `python -m webapp.app`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from flask import Flask, jsonify, request, send_from_directory

from generate import generate, load_checkpoint
from utils import pick_device

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "checkpoints" / "model.pt"
STATIC_DIR = Path(__file__).resolve().parent / "static"
STOP_SEQUENCE = "\nYou:"
MAX_NEW_CHARS = 800

app = Flask(__name__, static_folder=None)
device = pick_device()
_state = {}


def get_model_state():
    """Lazily load the checkpoint once, on first request. Returns None if untrained."""
    if "model" not in _state:
        if not CHECKPOINT_PATH.exists():
            return None
        model, config, stoi, itos = load_checkpoint(CHECKPOINT_PATH, device)
        _state.update(model=model, config=config, stoi=stoi, itos=itos)
    return _state


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)


@app.get("/api/info")
def info():
    state = get_model_state()
    if state is None:
        return jsonify({"ready": False})
    num_params = sum(p.numel() for p in state["model"].parameters())
    return jsonify({"ready": True, "device": str(device), "num_params": num_params, **state["config"]})


@app.post("/api/generate")
def api_generate():
    state = get_model_state()
    if state is None:
        return jsonify({"error": "No trained model found. Run `python train.py` first, then reload."}), 400

    body = request.get_json(force=True)
    transcript = body.get("prompt", "")
    user_text = body.get("message", transcript)
    new_chars = max(1, min(int(body.get("new_chars", 200)), MAX_NEW_CHARS))
    temperature = max(0.05, float(body.get("temperature", 0.8)))

    model, config, stoi, itos = state["model"], state["config"], state["stoi"], state["itos"]

    # The model can only encode characters it saw during training. Rather than reject
    # the whole message, drop unseen characters and tell the caller which ones — but
    # only report drops from what the *user* typed, not from our own "You: " / "GPT: "
    # wrapper text (a small training corpus may simply be missing e.g. capital "G").
    dropped = sorted(set(user_text) - set(stoi))
    filtered = "".join(ch for ch in transcript if ch in stoi)
    if not filtered:
        return jsonify({"error": "Message contains no characters this model was trained on."}), 400

    context = torch.tensor([[stoi[ch] for ch in filtered]], dtype=torch.long, device=device)
    reply = generate(model, context, new_chars, config["context_length"], itos, temperature=temperature)

    if STOP_SEQUENCE in reply:
        reply = reply[: reply.index(STOP_SEQUENCE)]

    return jsonify({"reply": reply, "dropped_chars": dropped})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
