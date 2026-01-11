#!/usr/bin/env python3
import argparse
import sys
import os
import logging
import traceback
from datetime import datetime
import yaml

from src.app import App


def setup_logging(config_path: str):
    """Initialize logging with file + stdout handlers, honoring config.logging."""
    log_dir = "logs"
    enabled = True
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        log_cfg = cfg.get("logging", {}) or {}
        enabled = log_cfg.get("enabled", True)
        log_dir = log_cfg.get("dir", log_dir)
    except Exception:
        # Fallback to defaults if config missing or unreadable
        pass

    if not enabled:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        return None

    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"app-{ts}.log")

    handlers = [
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )
    logging.info(f"Logging initialized at {log_file}")
    return log_file

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cameras.yaml", help="Path to camera config file")
    ap.add_argument("--fullscreen", action="store_true", help="Start in fullscreen mode")
    args = ap.parse_args()

    setup_logging(args.config)

    try:
        app = App(args.config, args.fullscreen)
        app.run()
    except Exception as e:
        logging.exception("Unhandled exception")
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
