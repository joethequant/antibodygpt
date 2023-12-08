import subprocess
import pandas as pd
import os

from tempfile import NamedTemporaryFile
from .utils import seqs_to_fasta, fasta_to_csv

def number_seqs_as_df(seqs, scheme="imgt"):
    """Number a collection of sequences or (description, sqeuence) pairs with ANARCI,
    returning the numbered Abs as a dataframe. Returns a tuple of dataframes, the first
    corresponding to heavy chain annotations, and the second for light chain annotations. """

    with NamedTemporaryFile(delete=False) as tempf_i:
        seqs_to_fasta(seqs, tempf_i.name)
    with NamedTemporaryFile(delete=False) as tempf_o:
        subprocess.run(["ANARCI", "-i", tempf_i.name, "-o", tempf_o.name, "--csv"])

    if os.path.isfile(tempf_o.name + "_H.csv"):
        df_result_H = pd.read_csv(tempf_o.name + "_H.csv")
        os.remove(tempf_o.name + "_H.csv")
    else:
        df_result_H = None
    if os.path.isfile(tempf_o.name + "_KL.csv"):
        df_result_KL = pd.read_csv(tempf_o.name + "_KL.csv")
        os.remove(tempf_o.name + "_KL.csv")
    else:
        df_result_KL = None

    os.remove(tempf_o.name)
    os.remove(tempf_i.name)

    return df_result_H, df_result_KL
