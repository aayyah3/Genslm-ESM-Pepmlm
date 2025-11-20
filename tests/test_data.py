"""Test cases for the data module."""

from __future__ import annotations

from genslm_esm.data import FastaDataset
from genslm_esm.data import group_codons


class TestGroupCodons:
    """Test cases for the group_codons function.

    Test cases for the group_codons function to ensure proper handling of
    ambiguous nucleotides.
    """

    def test_valid_codons_preserved(self) -> None:
        """Test that valid codons are preserved correctly."""
        # Test some standard valid codons
        valid_sequence = 'ATGGCTAGCTAA'  # ATG GCT AGC TAA
        result = group_codons(valid_sequence)
        expected = 'ATG GCT AGC TAA'
        assert result == expected

    def test_ambiguous_nucleotides_replaced_with_unk(self) -> None:
        """Test that all ambiguous nucleotide codes are replaced with <unk>."""
        # Test each ambiguous nucleotide code
        ambiguous_tests = [
            'Y',  # Pyrimidine (C or T)
            'R',  # Purine (A or G)
            'W',  # Weak (A or T)
            'S',  # Strong (G or C)
            'K',  # Keto (T or G)
            'M',  # Amino (C or A)
            'D',  # A, G, T (not C)
            'H',  # A, C, T (not G)
            'V',  # A, C, G (not T)
            'B',  # C, G, T (not A)
            'X',  # Any base
            'N',  # Any base
            '-',  # Gap
        ]

        for ambiguous_char in ambiguous_tests:
            # Test single ambiguous character in a codon
            test_seq = f'AT{ambiguous_char}'  # AT + ambiguous = invalid codon
            result = group_codons(test_seq)
            assert result == '<unk>', f'Failed for {ambiguous_char}'

            # Test ambiguous character at different positions
            test_seq2 = f'{ambiguous_char}TG'  # ambiguous + TG = invalid codon
            result2 = group_codons(test_seq2)
            assert result2 == '<unk>', f'Failed for {ambiguous_char} at start'

            test_seq3 = (
                f'A{ambiguous_char}G'  # A + ambiguous + G = invalid codon
            )
            result3 = group_codons(test_seq3)
            assert result3 == '<unk>', f'Failed for {ambiguous_char} in middle'

    def test_mixed_valid_invalid_codons(self) -> None:
        """Test sequences with both valid and invalid codons."""
        # Mix of valid and invalid codons
        mixed_seq = 'ATGYRWGCT'  # ATG (valid), YRW (invalid), GCT (valid)
        result = group_codons(mixed_seq)
        expected = 'ATG <unk> GCT'
        assert result == expected

    def test_case_insensitive(self) -> None:
        """Test that the function handles uppercase and lowercase input."""
        # Test lowercase ambiguous characters
        lowercase_seq = 'atgyrwgct'  # atg (valid), yrw (invalid), gct (valid)
        result = group_codons(lowercase_seq)
        expected = 'ATG <unk> GCT'
        assert result == expected

        # Test mixed case
        mixed_case_seq = 'AtGyRwGcT'  # AtG (valid), yRw (invalid), GcT (valid)
        result2 = group_codons(mixed_case_seq)
        expected2 = 'ATG <unk> GCT'
        assert result2 == expected2

    def test_empty_sequence(self) -> None:
        """Test empty sequence handling."""
        result = group_codons('')
        assert result == ''

    def test_short_sequences(self) -> None:
        """Test sequences shorter than 3 characters."""
        # Single character
        result1 = group_codons('A')
        assert result1 == '<unk>'

        # Two characters
        result2 = group_codons('AT')
        assert result2 == '<unk>'

    def test_sequences_not_divisible_by_3(self) -> None:
        """Test sequences where length is not divisible by 3."""
        # Length 4: ATGY -> ATG (valid) + Y (invalid, becomes <unk>)
        result1 = group_codons('ATGY')
        assert result1 == 'ATG <unk>'

        # Length 5: ATGYR -> ATG (valid) + YR (invalid, becomes <unk>)
        result2 = group_codons('ATGYR')
        assert result2 == 'ATG <unk>'

    def test_all_ambiguous_codons(self) -> None:
        """Test codons made entirely of ambiguous characters."""
        ambiguous_codons = [
            'YYY',  # All pyrimidines
            'RRR',  # All purines
            'WWW',  # All weak
            'SSS',  # All strong
            'KKK',  # All keto
            'MMM',  # All amino
            'DDD',  # All D
            'HHH',  # All H
            'VVV',  # All V
            'BBB',  # All B
            'XXX',  # All X
            'NNN',  # All N
            '---',  # All gaps
        ]

        for codon in ambiguous_codons:
            result = group_codons(codon)
            assert result == '<unk>', f'Failed for codon {codon}'

    def test_whitespace_handling(self) -> None:
        """Test that whitespace in input is handled correctly."""
        # Input with spaces should still work
        spaced_seq = 'A T G Y R W'
        result = group_codons(spaced_seq)
        # The function groups by 3 characters, so spaces are treated as
        # characters. This creates invalid codons: "A T", " G ", "Y R", " W "
        # (last one incomplete)
        expected = '<unk> <unk> <unk> <unk>'
        assert result == expected

    def test_known_valid_codons_from_table(self) -> None:
        """Test that all codons in the translation table are recognized."""
        # Test a few key codons from the translation table
        test_codons = ['TTT', 'TTC', 'TTA', 'TTG', 'ATG', 'TAA', 'TAG', 'TGA']

        for codon in test_codons:
            result = group_codons(codon)
            assert result == codon, f'Valid codon {codon} was not preserved'

    def test_comprehensive_ambiguous_sequence(self) -> None:
        """Test a sequence with multiple types of ambiguous characters."""
        # Create a sequence with various ambiguous characters
        complex_seq = 'ATGYRWSKMDHVBNX-ATGCGCTAA'
        result = group_codons(complex_seq)
        # Expected: ATG (valid), YRW (invalid), SKM (invalid), DHV (invalid),
        # BNX (invalid), -AT (invalid), GCG (valid), CTA (valid),
        # A (incomplete, invalid)
        expected = 'ATG <unk> <unk> <unk> <unk> <unk> GCG CTA <unk>'
        assert result == expected

    def test_whitespace_in_output(self) -> None:
        """Test that output format uses single spaces between codons."""
        seq = 'ATGGCTAGCTAA'
        result = group_codons(seq)
        # Should have single spaces between codons
        assert '  ' not in result  # No double spaces
        # Three spaces for four codons
        assert result.count(' ') == 3  # noqa: PLR2004


def test_dataset_amino_acid_space() -> None:
    """Test that the FastaDataset handles amino acid space correctly."""
    # Sequences that look like nucleotide, but we want amino acids
    sequences = ['ATGGCTAGCTAA', 'ATGCGT']

    # Create the dataset requesting amino acids
    dataset = FastaDataset(
        sequences=sequences,
        return_codon=False,
        return_aminoacid=True,  # Request amino acids in the getitem
        contains_nucleotide=False,  # Indicate sequences are amino acids
    )

    # Get all items in the dataset
    items = [dataset[i] for i in range(len(dataset))]

    # Check that amino acids are space-separated
    assert items[0]['aminoacid'] == 'A T G G C T A G C T A A'
    assert items[1]['aminoacid'] == 'A T G C G T'


def test_dataset_codon_translation() -> None:
    """Test that the FastaDataset handles amino acid space correctly."""
    # Nucleotide sequences to be translated
    sequences = ['ATGGCTAGCTAA', 'ATGCGT']

    # Create the dataset requesting amino acids
    dataset = FastaDataset(
        sequences=sequences,
        return_codon=False,
        return_aminoacid=True,  # Request amino acids in the getitem
        contains_nucleotide=True,  # Indicate sequences are nucleotide
    )

    # Get all items in the dataset
    items = [dataset[i] for i in range(len(dataset))]

    # Check that amino acids are space-separated
    assert items[0]['aminoacid'] == 'M A S'
    assert items[1]['aminoacid'] == 'M R'
