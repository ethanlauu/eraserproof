import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from eraserproof.verification.evaluate import main


if __name__ == "__main__":
    main()
