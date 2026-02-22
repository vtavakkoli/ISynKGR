from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isynkgr.evaluation.data_gen.generate_samples import generate as gen_samples
from isynkgr.evaluation.data_gen.generate_ground_truth import generate as gen_gt
from isynkgr.evaluation.data_gen.validate_data import main as validate


def run() -> None:
    samples = Path('data/samples')
    gt = Path('data/ground_truth')
    gen_samples(samples, 100)
    gen_gt(samples, gt)
    validate(samples, gt)


if __name__ == '__main__':
    run()
