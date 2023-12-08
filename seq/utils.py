import pandas as pd
from Bio import SeqIO

def seqs_to_fasta(seqs, outfile):
    """ Write a collection of sequences (such as a list or pd.Series) to a file in FASTA format. 

    Args:
        seqs (iterable): A collection of sequences or (description, sequence) tuples.
        outfile (str): The path to the output FASTA file.
    """
    with open(outfile, 'w') as fasta_file:
        for seq_entry in seqs:
            if isinstance(seq_entry, tuple) and len(seq_entry) == 2:
                description, sequence = seq_entry
                fasta_file.write(f">{description}\n")
                fasta_file.write(f"{sequence}\n")
            elif isinstance(seq_entry, str):
                # If no description is provided, use a default description
                fasta_file.write(f">Sequence\n")
                fasta_file.write(f"{seq_entry}\n")
            else:
                raise ValueError("Each entry in seqs should be a sequence string or a (description, sequence) tuple.")
            
def fasta_to_csv(fasta_file, csv_file):
    """
    Convert a FASTA file to a CSV file using Biopython's SeqIO module.

    Args:
        fasta_file (str): Path to the input FASTA file.
        csv_file (str): Path to the output CSV file.
    """
    records = SeqIO.parse(fasta_file, "fasta")
    df = pd.DataFrame([(record.id, str(record.seq)) for record in records], columns=["description", "sequence"])
    df.to_csv(csv_file, index=False)

def csv_to_fasta(csv_file, fasta_file):
    """
    Convert a CSV file containing "description" and "sequence" columns to a FASTA file using Biopython's SeqIO module.

    Args:
        csv_file (str): Path to the input CSV file.
        fasta_file (str): Path to the output FASTA file.
    """
    df = pd.read_csv(csv_file)
    records = []

    for _, row in df.iterrows():
        record = f">{row['description']}\n{row['sequence']}"
        records.append(record)

    with open(fasta_file, 'w') as fasta_output:
        fasta_output.write("\n".join(records))
        