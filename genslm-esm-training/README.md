# genslm-esm-training

This package contains the original training code for the GenSLM-ESM model.

## Usage

We provide a command line interface to the package. To see the available commands, run:
```console
genslm-esm-training --help
```

### Write aminoacid fasta given a nucleotide fasta file
```console
genslm-esm-training write-aminoacid-fasta -f data/mdh/valid.fasta -o data/mdh/valid_aminoacid.fasta
```

### Make a train/validation split given a fasta file
```console
genslm-esm-training random-split-fasta -f data/mdh/mdh_natural_sequences.fnn -o data/mdh/
```

### Get the best checkpoint according to the validation loss
```console
genslm-esm-training best-checkpoint --train_output_dir <training_dir>
```

## Submit a training run to Polaris
This submits a 2 node job for 2 hours to the `your_queue` queue charged to `your_project` using the `polaris-pbs.sh` script.
```console
qsub -A your_project -q your_queue -l select=2:system=polaris -l walltime=2:00:00 -d /path/to/your/workdir examples/ec/polaris-pbs-v2.sh /lus/eagle/projects/CVD-Mol-AI/braceal/src/genslm-esm/genslm-esm-training/examples/ec/training_configs/ec_aminoacid_8m.yaml
```

### Write loss curves to csv
```console
genslm-esm-training aggregate-loss-curves --training_runs_dir runs/ec_prod --output_dir examples/ec/loss_curves
```

## Contributing

For development, it is recommended to use a virtual environment. The following
commands will create a virtual environment, install the package in editable
mode, and install the pre-commit hooks.
```bash
python -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
pip install -e '.[dev,docs]'
pre-commit install
```
To test the code, run the following command:
```bash
pre-commit run --all-files
tox -e py310
```
