#! /usr/bin/env python
# coding:utf-8

import pathlib
import os
import sys

import numpy as np
import pandas as pd

p = "/home/icb/lambert.moyon/projects/pydea/pyDEA/"
if not p in sys.path:
    sys.path.insert(0, p)
    
import pyDEA


def load_DEA_result(path, p1, p2, thresh_LFC, thresh_pval):
    """Load DESeq2 DEA results into a pyDEA.DE_results instance

    Args:
        path (str): path to table containing DESeq2-formatted results
        p1 (str) : name of first condition
        p2 (str) : name of second condition
        thresh_LFC (float)
        thresh_pval (float)

    Returns:
        pyDEA.DE_results

    """
    table = pd.read_csv(path,
                        header=0,
                        index_col=0,
                        sep="\t"
                       ).rename(
            columns={"log2FoldChange":"logFC",
                     "pvalue":"pval"
                    }
            ).loc[:,["logFC","pval","padj"]
                 ].replace(np.nan,1)

    DEA_result = pyDEA.DE_results(table, p1, p2,
                                   thresh_LFC,
                                   thresh_pval,
                                   path=path
                                  )
    return DEA_result

