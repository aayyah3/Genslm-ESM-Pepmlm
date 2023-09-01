from argparse import ArgumentParser
import weightwatcher as ww
from transformers import EsmForMaskedLM
from pathlib import Path
import pandas as pd


def main(model_path: Path, output_path: Path) -> None:
    model = EsmForMaskedLM.from_pretrained(model_path)
    watcher = ww.WeightWatcher(model=model)
    details = watcher.analyze(plot=False, savefig=str(output_path), randomize=True)
    summary = watcher.get_summary()
    output_path.mkdir(exist_ok=True)
    details.to_csv(str(output_path / "details.csv"))
    pd.DataFrame(summary, index=[0]).to_csv(str(output_path / "summary.csv"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-m", "--model", required=True, type=Path, help="model checkpoint path"
    )
    parser.add_argument(
        "-o", "--output", required=True, type=Path, help="weight watcher output path"
    )
    args = parser.parse_args()
    main(args.model, args.output)
