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