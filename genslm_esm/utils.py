from pathlib import Path
from typing import Tuple


def best_checkpoint(output_dir: Path) -> Tuple[float, Path]:
    """Utility to report the best checkpoint from a training run (measured by smallest eval_loss).

    Parameters
    ----------
    output_dir : Path
        The directory containing the training run checkpoints.
    """
    from pathlib import Path

    import numpy as np
    from transformers.trainer_callback import TrainerState

    # Get the checkpoint directories in order
    ckpt_dirs = list(Path(output_dir).glob("checkpoint-*"))
    ckpt_dirs = list(sorted(ckpt_dirs, key=lambda x: int(str(x).split("-")[1])))
    ckpt_steps = [int(str(x).split("-")[1]) for x in ckpt_dirs]
    last_ckpt = ckpt_dirs[-1]

    # Load the trainer state wih the full log history
    state = TrainerState.load_from_json(f"{last_ckpt / 'trainer_state.json'}")

    # Get the eval losses and steps from each checkpoint
    eval_losses, steps = [], []
    for item in state.log_history:
        if "eval_loss" in item:
            eval_losses.append(item["eval_loss"])
            steps.append(item["step"])

    # Get the best eval loss and step
    best_ind = np.argmin(eval_losses)
    best_step = steps[best_ind]
    best_loss = eval_losses[best_ind]

    # Since checkpoint report steps are not necessarily the same as eval steps,
    # we need to find the closest checkpoint report step to the best eval step.
    best_step = ckpt_steps[np.argmin(np.abs(np.array(ckpt_steps) - best_step))]

    best_ckpt = output_dir / f"checkpoint-{best_step}"

    return best_loss, best_ckpt


def write_loss_curve(run_dir: Path, csv_file: Path) -> None:
    """Utility to parse hugging face trainer state and write loss curve to CSV file.

    Parameters
    ----------
    output_dir : Path
        The directory to write the gathered checkpoints to.

    """
    import pandas as pd
    from collections import defaultdict
    from transformers.trainer_callback import TrainerState

    # Check if the trainer state exists
    trainer_state_json = run_dir / "trainer_state.json"

    # If trainer state doesn't exist then load the best checkpoint instead
    if not trainer_state_json.exists():
        print(
            f"Trainer state does not exist for {run_dir}. "
            "Loading best checkpoint instead ..."
        )
        # Get the checkpoint directories in order
        ckpt_dirs = list(Path(run_dir).glob("checkpoint-*"))
        ckpt_dirs = list(sorted(ckpt_dirs, key=lambda x: int(str(x).split("-")[1])))
        last_ckpt = ckpt_dirs[-1]

        # Load the trainer state wih the full log history
        trainer_state_json = last_ckpt / "trainer_state.json"

    # Load the trainer state wih the full log history
    state = TrainerState.load_from_json(str(trainer_state_json))

    # Aggregate data by logging step using defaultdict
    data = defaultdict(dict)
    for entry in state.log_history:
        data[entry["step"]].update(entry)

    # Convert to DataFrame
    df = pd.DataFrame(list(data.values()))

    # Write to CSV file
    df.to_csv(csv_file, index=False)


def aggregate_loss_curves(training_runs_dir: Path, output_dir: Path) -> None:
    """Utility to write the loss curve to CSV for each training run directory.

    Parameters
    ----------
    training_runs_dir : Path
        The directory containing different training run directories.

    output_dir : Path
        The directory to write the gathered loss curve CSV files to.
    """
    # Make the output directory
    output_dir.mkdir(exist_ok=True)

    # Loop through each training run directory and write the loss curve to CSV
    for run_dir in filter(lambda x: x.is_dir(), training_runs_dir.glob("*")):
        write_loss_curve(run_dir, output_dir / f"{run_dir.name}.csv")
