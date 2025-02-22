import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

from genslm_esm.dataset import (
    FastaDataset,
    HDF5Dataset,
    SequenceHomologySampler,
    GenSLMColatorForLanguageModeling,
)
from genslm_esm.refseq_dataset import (
    MultiEpochScannerSequenceDataset,
    ScannerSequenceDataset,
    HDF5Dataset,
    RefSeqCollator,
) 
    
    
    
    
# /lus/eagle/projects/FoundEpidem/hippekp/genslm-foundation/data/ncbi/refseq.parsed/parquet-training/train_epoch_0_linclust50_faafna.parquet /lus/eagle/projects/FoundEpidem/hippekp/genslm-foundation/data/ncbi/refseq.parsed/parquet-training/train_epoch_1_linclust50_faafna.parquet /lus/eagle/projects/FoundEpidem/hippekp/genslm-foundation/data/ncbi/refseq.parsed/parquet-training/train_epoch_2_linclust50_faafna.parquet /lus/eagle/projects/FoundEpidem/hippekp/genslm-foundation/data/ncbi/refseq.parsed/parquet-training/train_epoch_3_linclust50_faafna.parquet /lus/eagle/projects/FoundEpidem/hippekp/genslm-foundation/data/ncbi/refseq.parsed/parquet-training/train_epoch_4_linclust50_faafna.parquet /lus/eagle/projects/FoundEpidem/hippekp/genslm-foundation/data/ncbi/refseq.parsed/parquet-training/train_epoch_5_linclust50_faafna.parquet
train_path = "/lus/eagle/projects/FoundEpidem/hippekp/genslm-foundation/data/ncbi/refseq.parsed/parquet-training/train_epoch_6_linclust50_faafna.parquet /lus/eagle/projects/FoundEpidem/hippekp/genslm-foundation/data/ncbi/refseq.parsed/parquet-training/train_epoch_7_linclust50_faafna.parquet /lus/eagle/projects/FoundEpidem/hippekp/genslm-foundation/data/ncbi/refseq.parsed/parquet-training/train_epoch_8_linclust50_faafna.parquet /lus/eagle/projects/FoundEpidem/hippekp/genslm-foundation/data/ncbi/refseq.parsed/parquet-training/train_epoch_9_linclust50_faafna.parquet".split()
eval_path="/lus/eagle/projects/FoundEpidem/hippekp/genslm-foundation/data/ncbi/refseq.parsed/parquet-training/refseq-validation-10k-faafna.parquet"

compute_aminoacid_loss = True
compute_codon_loss = True
tokenizer_path = "/lus/eagle/projects/FoundEpidem/azton/genslm-esm/tokenizer_esm_genslm"
train_dataset = ScannerSequenceDataset(
    train_path,
    return_aminoacid=compute_aminoacid_loss,
    return_codon=compute_codon_loss,
    
)
if eval_path:
    eval_dataset = ScannerSequenceDataset(
    eval_path,
    return_aminoacid=compute_aminoacid_loss,
    return_codon=compute_codon_loss,
    )
else:
    eval_dataset = None

str_dtype = h5py.special_dtype(vlen=str)
codon_strs = []
aminoacid_strs = []
ids = []
if False:
    for i, filename in tqdm(enumerate(train_path), total=len(train_path), position=0):
        # if i > 0: exit()
        train_dataset = ScannerSequenceDataset(
            filename,
            return_aminoacid=compute_aminoacid_loss,
            return_codon=compute_codon_loss,
            
        )

        for sample in tqdm(train_dataset, total=14035600, position=1):
            # ids.append(sample['id'])
            codon_strs.append(sample['codon'])
            aminoacid_strs.append(sample['aminoacid'])
            # if len(codon_strs) == 100000:
            #     break
        rootname = Path(filename).stem
        h5path="/lus/eagle/projects/FoundEpidem/azton/genomes/refseq_genslm_esm/test_{}.h5".format(rootname)
        print(f"saving to {h5path}")
        # print(f"ids: {len(ids)}")
        # print(f"codon_strs: {len(codon_strs)}")
        # print(f"aminoacid_strs: {len(aminoacid_strs)}")
        # print(f"sample: {sample}")
        # print(f"{codon_strs[-1]=}")
        # print(f"{aminoacid_strs[-1]=}")
        with h5py.File(h5path, 'w') as h5file:
            # h5file.create_dataset('id', data=ids, dtype=str_dtype)
            h5file.create_dataset('codon', data=codon_strs, dtype=str_dtype)
            h5file.create_dataset('aminoacid', data=aminoacid_strs, dtype=str_dtype)
    ids = []
    codon_strs = []
    aminoacid_strs = []

if True:
    codon_strs = []
    aminoacid_strs = []
    ids = []

    for sample in tqdm(eval_dataset, total=10000):
        # ids.append(sample['id'])
        codon_strs.append(sample['codon'])
        aminoacid_strs.append(sample['aminoacid'])

    rootname = Path(eval_path).stem
    h5path="/lus/eagle/projects/FoundEpidem/azton/genomes/refseq_genslm_esm/valid{}.h5".format(rootname)
    with h5py.File(h5path, 'w') as h5file:
        # h5file.create_dataset('id', data=ids, dtype=str_dtype)
        h5file.create_dataset('codon', data=codon_strs, dtype=str_dtype)
        h5file.create_dataset('aminoacid', data=aminoacid_strs, dtype=str_dtype)
        ids = []
        codon_strs = []
        aminoacid_strs = []


