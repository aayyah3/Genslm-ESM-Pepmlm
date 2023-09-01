"""
This code was adapted from the EsmFold tutorial notebook, which can be found here:
https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/protein_folding.ipynb#scrollTo=1c9de19e
"""

import torch
from transformers import AutoTokenizer, EsmForProteinFolding


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1", low_cpu_mem_usage=True
    )

    model = model.cuda()

    # Performance optimization
    model.esm = model.esm.half()
    torch.backends.cuda.matmul.allow_tf32 = True

    # Uncomment this line if your GPU memory is 16GB or less, or if you're folding longer (over 600 or so) sequences
    # model.trunk.set_chunk_size(64)

    # Test protein sequence
    test_protein = [
        "MGAGASAEEKHSRELEKKLKEDAEKDARTVKLLLLGAGESGKSTIVKQMKIIHQDGYSLEECLEFIAIIYGNTLQSILAIVRAMTTLNIQYGDSARQDDARKLMHMADTIEEGTMPKEMSDIIQRLWKDSGIQACFERASEYQLNDSAGYYLSDLERLVTPGYVPTEQDVLRSRVKTTGIIETQFSFKDLNFRMFDVGGQRSERKKWIHCFEGVTCIIFIAALSAYDMVLVEDDEVNRMHESLHLFNSICNHRYFATTSIVLFLNKKDVFFEKIKKAHLSICFPDYDGPNTYEDAGNYIKVQFLELNMRRDVKEIYSHMTCATDTQNVKFVFDAVTDIIIKENLKDCGLF"
    ]

    # Tokenize protein sequence
    # tokenized_input = tokenizer(
    #     [test_protein], return_tensors="pt", add_special_tokens=False
    # )["input_ids"]

    # tokenized_input = tokenized_input.cuda()

    # with torch.no_grad():
    pdbs = model.infer_pdbs(test_protein)

    with open("result.pdb", "w") as f:
        f.write(pdbs[0])


if __name__ == "__main__":
    main()
