"""Entry point: YOLO26 + SAM2 Label Studio ML backend with optional Active Learning."""

import argparse
import json
import logging
import os
from pathlib import Path

from label_studio_ml.api import init_app
from model import YoloSamBackend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ── Load .env ────────────────────────────────────────────────────────────────
_env_file = Path(__file__).with_name(".env")
try:
    from dotenv import load_dotenv

    load_dotenv(_env_file)
except ImportError:
    # Minimal fallback — parse .env without python-dotenv
    if _env_file.exists():
        for _line in _env_file.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _key, _val = _line.lstrip("export ").split("=", 1)
            _val = _val.strip().strip("'\"")
            os.environ.setdefault(_key.strip(), _val)

# ── Select model class based on configuration ────────────────────────────────
if os.getenv("ACTIVE_LEARNING_ENABLED", "true").lower() in ("true", "1", "yes"):
    from active_model import ActiveYoloSamBackend

    model_class = ActiveYoloSamBackend
    logging.info("Using ActiveYoloSamBackend (Active Learning enabled)")
else:
    from model import YoloSamBackend

    model_class = YoloSamBackend
    logging.info("Using YoloSamBackend (standard)")

# ── label-studio-ml compat flags ─────────────────────────────────────────────
os.environ.setdefault("LABEL_STUDIO_ML_BACKEND_V2", "")  # force V1 mode
os.environ.setdefault("AUTO_UPDATE", "")

# ── Model dir setup ───────────────────────────────────────────────────────────
_base = Path(__file__).resolve().parent
_model_dir = Path(os.getenv("MODEL_DIR", ".ls-models"))
if not _model_dir.is_absolute():
    _model_dir = (_base / _model_dir).resolve()
model_dir = str(_model_dir)


def _ensure_initial_job_result(base_model_dir: str) -> None:
    """Create stub job-result files so Label Studio finds a valid model version."""
    root = Path(base_model_dir)
    for d in (root / "INITIAL", root / root.name):
        d.mkdir(parents=True, exist_ok=True)
        result = d / "job_result.json"
        if not result.exists():
            result.write_text(
                json.dumps({"status": "initial", "job_id": root.name}),
                encoding="utf-8",
            )


_ensure_initial_job_result(model_dir)

app = init_app(model_class=model_class, model_dir=model_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO+SAM2 Label Studio ML backend")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", 9090)))
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    app.run(host=args.host, port=args.port, debug=args.debug)
