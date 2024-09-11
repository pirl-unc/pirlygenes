#!/usr/bin/env python3

import pandas as pd
from glob import glob
from pyensembl import ensembl_grch38

from pirlygenes import load_all_dataframes


for csv_file, df in load_all_dataframes():
    gene_ids = df["Ensembl_Gene_ID"]
    names = df["Symbol"]
    for name, gene_id in zip(names, gene_ids):
        gene = ensembl_grch38.gene_by_id(gene_id)
        assert gene is not None, "No gene for %s" % gene_id
        print(name, gene_id, gene)
        if gene.name != name:
            print("Warning: %s != %s" % (name, gene.name))
