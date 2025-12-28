import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.train_pipeline import build_and_save_model

def main():
    print("Starting non-interactive model build...")
    _ = build_and_save_model()
    print("Model build finished successfully.")

if __name__ == "__main__":
    main()
