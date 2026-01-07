#!/usr/bin/env python3
import argparse
import sys
import traceback
from src.app import App

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cameras.yaml", help="Path to camera config file")
    ap.add_argument("--fullscreen", action="store_true", help="Start in fullscreen mode")
    args = ap.parse_args()

    try:
        app = App(args.config, args.fullscreen)
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
