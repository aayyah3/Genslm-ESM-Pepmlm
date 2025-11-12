from __future__ import annotations

from esm.tokenization import EsmSequenceTokenizer

codons = [
    'GGC',
    'GCC',
    'ATC',
    'GAC',
    'GAA',
    'ATG',
    'GTG',
    'CTG',
    'GTC',
    'GCG',
    'GAT',
    'AAA',
    'GGT',
    'AAG',
    'GAG',
    'ACC',
    'AAC',
    'GTT',
    'ATT',
    'GCA',
    'CTC',
    'CGC',
    'GCT',
    'CAG',
    'CCG',
    'TTC',
    'GTA',
    'TCG',
    'GGA',
    'AAT',
    'TAC',
    'CTT',
    'TTG',
    'ACG',
    'TCC',
    'GGG',
    'AGC',
    'CCC',
    'ACA',
    'ACT',
    'TCT',
    'TTA',
    'CGT',
    'TAT',
    'CAA',
    'CGG',
    'TTT',
    'CAC',
    'CCT',
    'CCA',
    'TGG',
    'ATA',
    'TCA',
    'TGC',
    'AGT',
    'AGA',
    'CAT',
    'TGT',
    'CTA',
    'AGG',
    'TAA',
    'CGA',
    'TGA',
    'TAG',
]


def main():
    tokenizer = EsmSequenceTokenizer()
    # Get the current vocabulary
    vocabulary = tokenizer.get_vocab().keys()

    for codon in codons:
        # Check to see if codon is in the vocabulary or not (it won't be)
        # Keeping this line for generality
        if codon not in vocabulary:
            tokenizer.add_tokens(codon)

    # Save the new vocabulary
    tokenizer.save_pretrained('tokenizer_esmc_genslm')


if __name__ == '__main__':
    main()
