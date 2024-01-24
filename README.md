# genslm-esm

## Usage

We provide a command line interface to the package. To see the available commands, run:
```console
genslm-esm --help
```

### Write aminoacid fasta given a nucleotide fasta file
```console
genslm-esm write-aminoacid-fasta -f data/mdh/valid.fasta -o data/mdh/valid_aminoacid.fasta
```

### Make a train/validation split given a fasta file
```console
genslm-esm random-split-fasta -f data/mdh/mdh_natural_sequences.fnn -o data/mdh/
```

### Get the best checkpoint according to the validation loss
```console
genslm-esm best-checkpoint --train_output_dir <training_dir>
```

## Submit a training run to Polaris
This submits a 2 node job for 2 hours to the `your_queue` queue charged to `your_project` using the `polaris-pbs.sh` script.
```console
qsub -A your_project -q your_queue -l select=2:system=polaris -l walltime=2:00:00 -d /path/to/your/workdir examples/ec/polaris-pbs-v2.sh /lus/eagle/projects/CVD-Mol-AI/braceal/src/genslm-esm/examples/ec/training_configs/ec_aminoacid_8m.yaml
```