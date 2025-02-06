import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
import glob
import h5py
from torch.utils.data import DataLoader, Dataset, IterableDataset
from genslm.masking_scheduler import MaskingScheduler
import numpy as np
# In-memory dataset
class InMemorySequenceDataset(Dataset):
    def __init__(self, parquet_file):
        # Read the Parquet file into a PyArrow Table
        self.table = pq.read_table(parquet_file)
        self.num_rows = self.table.num_rows

        # Only need to read the 'id' and 'sequence' columns,
        # drop unnecessary columns. Columns present:
        # ['id', 'sequence', 'description', 'cluster_head', 'training_cluster'
        self.table = self.table.drop_columns(
            ["description", "cluster_head", "training_cluster"]
        )

    def __len__(self):
        return self.num_rows

    def __getitem__(self, idx):
        # Get the row at the specified index
        row = self.table.slice(idx, 1).to_pydict()

        # Dict is a list of values for each column (regardless of singular idx)
        sequence = row["sequence"][0]
        id_ = row["id"][0]

        # TODO: either need a collator or a tokenizer for the sequence
        yield [
                    id_,
                    sequence,
                ]


# Scanning dataset
class ScannerSequenceDataset(IterableDataset):
    def __init__(self, parquet_file, batch_size=1024, return_aminoacid=True, return_codon=False):
        # Initialize the dataset
        self.parquet_file = parquet_file
        self.batch_size = batch_size
        self.return_aminoacid = return_aminoacid
        self.return_codon = return_codon

    def __iter__(self):
        # Create a PyArrow dataset
        dataset = ds.dataset(self.parquet_file, format="parquet")

        # Only need to read the 'id' and 'sequence' columns
        # Columns present: ['id', 'sequence', 'description', 'cluster_head', 'training_cluster']
        columns = ["id", "nucleotide", "aminoacid"]

        # Create a scanner with the specified columns and batch size
        # Note this batch size is not the same as the DataLoader batch size,
        # but the number of rows to read at once
        scanner = dataset.scanner(columns=columns, batch_size=self.batch_size)

        # Iterate over RecordBatches
        for record_batch in scanner.to_batches():
            # Convert the RecordBatch to a dictionary of columns
            data = record_batch.to_pydict()
            num_rows = len(data["id"])

            for i in range(num_rows):
                # Extract sequence and other fields
                aa_sequence = None
                codon_sequence = None
                if self.return_aminoacid:
                    aa_sequence = ' '.join([c for c in data["aminoacid"][i]])
                if self.return_codon:
                    # have to split the codons by 3-mers
                    codon_sequence = data['nucleotide'][i]
                    codon_sequence = ' '.join([codon_sequence[i:i+3] for i in range(0, len(codon_sequence), 3)])
                id_ = data["id"][i]

                # print(f"Sequence: {sequence}")
                # TODO: either need a collator or a tokenizer for the sequence
                yield {
                    "id": id_,
                    "aminoacid": aa_sequence,
                    "codon": codon_sequence
                }
class MultiEpochScannerSequenceDataset(IterableDataset):
    # To work with MP in the dataloader we need to handle it explicitly
    # see https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
    def __init__(self, parquet_files: list[str], 
            batch_size: int = 1024, 
            return_aminoacid: bool = True, 
            return_codon: bool = False
            ):
        """
        parquet_files: List of Parquet file paths, one for each epoch.
        batch_size: Number of rows to read at once from the Parquet file.
        """
        self.parquet_files = parquet_files
        self.batch_size = batch_size
        self.return_aminoacid = return_aminoacid
        self.return_codon = return_codon

    def __iter__(self):
        # Iterate over each epoch's Parquet file
        for parquet_file in self.parquet_files:
            # Create a PyArrow dataset for the current file
            dataset = ds.dataset(parquet_file, format="parquet")

            # Only need to read the 'id' and 'sequence' columns
            columns = ["id", "nucleotide", "aminoacid"]

            # Create a scanner for the current file
            scanner = dataset.scanner(columns=columns, batch_size=self.batch_size)

            # Iterate over RecordBatches in the current file
            for record_batch in scanner.to_batches():
                data = record_batch.to_pydict()
                num_rows = len(data["id"])

                for i in range(num_rows):
                    aa_sequence = None
                    codon_sequence = None
                    if self.return_aminoacid:
                        aa_sequence = ' '.join([c for c in data["aminoacid"][i]])
                    if self.return_codon:
                        codon_sequence = data['nucleotide'][i]
                        codon_sequence = ' '.join([codon_sequence[i:i+3] for i in range(0, len(codon_sequence), 3)])
                    id_ = data["id"][i]

                    # Yield data from the current epoch
                    yield {
                        "id": id_,
                        "aminoacid": aa_sequence,
                        "codon": codon_sequence
                    }
class HDF5SequenceDataset(Dataset):
    def __init__(self, h5_filepath,
            return_aminoacid=True, 
            return_codon=True
            ): 
        self.return_aminoacid = return_aminoacid
        self.return_codon = return_codon
        self.h5_files = sorted(glob.glob(h5_filepath))
        samples_per_file = []
        for file in self.h5_files:
            with h5py.File(file, 'r') as data:
                samples_per_file.append(data['aminoacid'].shape[0])
        self.cum_samples_per_file = np.cumsum(samples_per_file)
        self.printed = False
    def __len__(self):
        return self.cum_samples_per_file[-1]
    
    def __getitem__(self, idx):
        file_idx = np.digitize(idx, self.cum_samples_per_file)
        if file_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cum_samples_per_file[file_idx-1]
        aa_sequence = None
        codon_sequence = None
        with h5py.File(self.h5_files[file_idx], 'r') as data:
            if self.return_aminoacid:
                # decode "bytes" to string
                aa_sequence = data['aminoacid'][sample_idx].decode('utf-8')
            if self.return_codon:
                codon_sequence = data['codon'][sample_idx].decode('utf-8')
    
            id_ = None #legacy

        if not self.printed:
            import torch.distributed as dist
            rank = dist.get_rank()
            if rank == 0:
                print(f"{aa_sequence=}")
                print(f"{codon_sequence}")
            self.printed = True
        
        return {
            "id": id_,
            "aminoacid": aa_sequence,
            "codon": codon_sequence
        }


class HDF5Dataset(Dataset):
    def __init__(self, h5_filepath):
        self.h5_files = sorted(glob.glob(h5_filepath))
        print(f"Found {len(self.h5_files)} files in {h5_filepath}", flush=True)
        # print(f"Found {len(self.h5_files)} files in {h5_filepath}")
        # print('\n'.join(self.h5_files))
        self.data = []
        samples_per_file = []
        for i, file in enumerate(self.h5_files):
            with h5py.File(file, 'r') as data:
                samples_per_file.append(data['input_ids'].shape[0])
        self.cum_samples_per_file = np.cumsum(samples_per_file)
        print(f"Samples per file: {samples_per_file}", flush=True)

    def __len__(self):
        return self.cum_samples_per_file[-1]

    def __getitem__(self, idx):
        file_idx = np.digitize(idx, self.cum_samples_per_file)
        if file_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cum_samples_per_file[file_idx-1]
        try:
            with h5py.File(self.h5_files[file_idx], 'r') as data:
                input_ids = torch.tensor(
                                data['input_ids'][sample_idx].astype(np.uint8), 
                                ).long()
        except IndexError as ie:
            with h5py.File(self.h5_files[file_idx], 'r') as data:
                print(f"Could not read from file {self.h5_files[file_idx]}")
                print(f"Index {idx} is out of bounds for dataset {data['input_ids'].shape[0]}")
            
                print(f"cum_samples_per_file = {self.cum_samples_per_file}")
                for i, file in enumerate(self.h5_files):
                    with h5py.File(file, 'r') as data:
                        print(f"[{i}] File {file} has {data['input_ids'].shape[0]} samples")
            
            raise ie
        return (None, input_ids)
    
class RefSeqCollator():
    def __init__(self, tokenizer, masked_lm=False, mlm_schedule=None, mlm_probability=0.15, require_tokenize=True):
        self.tokenizer = tokenizer
        self.require_tokenize = require_tokenize
        self.masked_lm = masked_lm
        self.max_length = 2048
        self.tokenizer.max_length = self.max_length
        self.mlm_scheduler = MaskingScheduler(schedule_type=mlm_schedule, 
                                manual_ratio=mlm_probability,
                                max_iters=10,
                                max_length=2048
                                ) if mlm_schedule else None

    def __call__(self, batch):
        # Tokenize the sequences
        # print(batch)
        # sequences = [" ".join([f"{aa}" for aa in seq[1]]) for seq in batch]
        sequences = [seq[1] for seq in batch]
        
        # make sure sequences are "max_length" long
        if self.require_tokenize:
            tokenized = self.tokenizer(sequences, 
                            padding='max_length', 
                            truncation=True, 
                            return_tensors="pt",
                            max_length=self.max_length)
        else: # we're recieving input ids only
            tokenized = {}
            tokenized['input_ids'] = torch.stack(sequences).squeeze(1)
            tokenized['attention_mask'] = tokenized['input_ids'] != self.tokenizer.pad_token_id
        # If we're using masked lm, need to 1) mask tokens and 2) create labels so we only take loss on the masked tokens
        if self.masked_lm:
            mask_rate = self.mlm_scheduler(np.random.randint(0, 10))
            # print(f"Mask rate: {mask_rate}")
            mask_indices = torch.rand(tokenized["input_ids"].shape) < mask_rate
            label_ids = tokenized["input_ids"].clone()
            pad_indices = tokenized["input_ids"] == self.tokenizer.pad_token_id

            tokenized["input_ids"][mask_indices & ~pad_indices] = self.tokenizer.mask_token_id

            # reset 10% of the masked tokens to random tokens
            random_indices = mask_indices & (torch.rand(tokenized["input_ids"].shape) < 0.1*mask_rate)
            tokenized["input_ids"][random_indices & ~pad_indices] = torch.randint(4,24, random_indices.size())[random_indices & ~pad_indices]  
            # for i, sample in enumerate(tokenized["input_ids"]):
            #     random_indices = mask_indices[i] & (torch.rand(sample.shape) < 0.1)
            #     for j in random_indices:
            #         if j:
            #             tokenized["input_ids"][i][j] = np.random.randint(4,27)
            # reset 10% of the masked tokens to the label
            label_indices = mask_indices & (torch.rand(tokenized["input_ids"].shape) < 0.1 * mask_rate)

            tokenized["input_ids"][label_indices] = label_ids[label_indices]

            label_ids[pad_indices] = -100
            label_ids[~mask_indices] = -100
        else:
            label_ids = tokenized["input_ids"].clone()


        # Convert the IDs to a tensor
        # ids = torch.tensor([seq[0] for seq in batch])
        # dont currently need ids, further crashes the standard HF trainer
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": label_ids,
            # "ids": ids,
        }

if __name__ == "__main__":
    from transformers import AutoTokenizer
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import os
    # dataset = ScannerSequenceDataset(
    #     "/lus/eagle/projects/FoundEpidem/hippekp/genslm-foundation/data/ncbi/refseq.parsed/parquet-training/epoch_0_linclust50.parquet"
    # )
    dataset = HDF5Dataset("/lus/eagle/projects/FoundEpidem/azton/genslm2/genslm/hdf5_dataset_stdtables_directpqt/")
    tokenizer = AutoTokenizer.from_pretrained("/lus/eagle/projects/FoundEpidem/azton/genslm2/genslm/tokenizer_genomic_llama")
    bad_tokens = [tokenizer.token_to_id(t) for t in ['X', 'U', 'O', 'B', 'Z']]
    print(f"{bad_tokens=}")
    print(tokenizer.get_vocab())
    print(f"{tokenizer.pad_token_id=}")
    print(f"{tokenizer.mask_token_id=}")
    collator = RefSeqCollator(tokenizer=tokenizer, masked_lm=True, mlm_schedule='fixed', mlm_probability=0.3, require_tokenize=isinstance(dataset, ScannerSequenceDataset))
    dataloader = DataLoader(dataset, batch_size=128, collate_fn=collator, num_workers=0, shuffle=False)

    for i, batch in tqdm(enumerate(dataloader)):
        for k in batch:
            print(f"{k=}")
            print(f"{batch[k].shape=}")
            print(f"{batch[k].shape=}")
            print(f"{batch[k][0].tolist()=}")
            print(f"{batch[k][0].min()=} ::: {batch[k][0].max()=}")
        exit()
    aminos = np.array([0 for _ in tokenizer.get_vocab()])
    bad_tok_seq_cnt = 0
    bad_tok_seq = []
    if not os.path.exists("aa_freq.txt"):
        for i, batch in tqdm(enumerate(dataloader), total=1000, desc="Counting Amino Acids"):
            if i > 1:
                break
            labels = batch["labels"]
            input_ids = batch["input_ids"]
            print(f"{input_ids.tolist()=}")
            for iid in input_ids:
                is_bad = False
                for bt in bad_tokens:            
                    if bt in iid:
                        if not is_bad:
                            is_bad = True
                            bad_tok_seq_cnt += 1
                            bad_tok_seq.append(iid.tolist())
            for s in input_ids:
                # iterate samples-
                for aa in s:
                    if aa != tokenizer.pad_token_id and aa != tokenizer.mask_token_id:
                        aminos[aa] += 1
        print(f"Found {bad_tok_seq_cnt} sequences with bad tokens of {(i+1)*128} samples ({bad_tok_seq_cnt/((i+1)*128)*100:0.2f}%)")
        with open('bad_tok_seq.txt', 'w') as f:
            for seq in bad_tok_seq:
                s = ', '.join([tokenizer.id_to_token(t) for t in seq])
                f.write(f"{s}\n")
        plt.figure(figsize=(10, 5))
        plt.bar(np.arange(len(aminos)), aminos)
        plt.xlabel("Amino Acid")
        plt.ylabel("Frequency")
        plt.yscale("log")
        plt.vlines(33, 0, 1e6, color='r', linestyle='--')
        plt.savefig('test_aa_freq.png')
        with open("aa_freq.txt", "w") as f:
            for i, aa in enumerate(aminos):
                f.write(f"{i}, {aa}\n")

    else:
        aminos = []
        with open("aa_freq.txt", "r") as f:
            for line in f:
                aminos.append(int(line.split(",")[1]))
        plt.figure(figsize=(10, 5))
        plt.bar(np.arange(len(aminos)), aminos)
        plt.xlabel("Amino Acid")
        plt.ylabel("Frequency")
        plt.yscale("log")
        plt.vlines(33, 0, 1e6, color='r', linestyle='--')
        plt.vlines(28, 0, 1e6, color='g', linestyle='--')
        plt.vlines(4, 0, 1e6, color='g', linestyle='--')
        plt.savefig('test_aa_freq.png')