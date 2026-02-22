from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isynkgr.evaluation.benchmark_runner import main

if __name__ == '__main__':
    main()
