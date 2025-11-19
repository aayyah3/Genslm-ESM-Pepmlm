from __future__ import annotations

import os
import subprocess
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from glob import glob

import pandas as pd
from tqdm import tqdm


def run_tm_align(kwargs: dict) -> dict[str, float]:
    tm_align_exe = kwargs.get('tm_align_exe')
    pdb1 = kwargs.get('pdb1')
    pdb2 = kwargs.get('pdb2')

    # Run TMalign subprocess
    command = f'{tm_align_exe} {pdb1} {pdb2} -outfmt 2'
    proc = subprocess.run(
        command,
        check=False,
        shell=True,
        capture_output=True,
    )

    # Parse stdout
    out = str(proc.stdout).split('\\')
    output = {
        'pdb1': pdb1,
        'pdb2': pdb2,
        'tm_score1': float(out[13][1:]),
        'tm_score2': float(out[14][1:]),
        'rmsd': float(out[15][1:]),
        'aligned_length': float(out[21][1:]),
    }
    len1, len2 = float(out[19][1:]), float(out[20][1:])
    max_len = max(len1, len2)
    output['coverage'] = output['aligned_length'] / max_len
    return output


def bulk_tm_align(
    tm_align_exe: str,
    pdbs: list[str],
    num_workers: int = 1,
) -> pd.DataFrame:
    """Run TM-align on all pairs of pdb files"""
    # Don't align seqs against themselves and don't repeat work
    kwargs = []
    for i in range(len(pdbs)):
        pdb1 = pdbs[i]
        for j in range(i, len(pdbs)):
            pdb2 = pdbs[j]
            if pdb1 == pdb2:
                continue
            kwargs.append(
                {
                    'tm_align_exe': tm_align_exe,
                    'pdb1': pdb1,
                    'pdb2': pdb2,
                },
            )

    print(f'Processing {len(kwargs)} TMAlign calls')

    outputs = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for output in tqdm(executor.map(run_tm_align, kwargs)):
            outputs.append(output)
    return pd.DataFrame(outputs)


def main():
    # Get list of PDB files
    pdbs = glob.glob('data/pdb/*/*.pdb')
    pdbs = [os.path.basename(pdb) for pdb in pdbs]

    # Run TMAlign
    df = bulk_tm_align(tm_align_exe='TMalign', pdbs=pdbs, num_workers=4)
    df.to_csv('data/tmalign.csv', index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--pdb_dir', type=str, required=True)
    parser.add_argument('--tm_align_exe', type=str, default='TMalign')
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()
    main()
