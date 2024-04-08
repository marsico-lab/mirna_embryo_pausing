# README - promoters_miRNAs_motifs

Command for scanning:

```
rsat matrix-scan \
    -v 1 \
    -quick \
    -i {input.fasta} \
    -seq_format fasta \
    -m {input.matrix_fp} \
    -matrix_format transfac \
    -ac_as_name \
    -o {output.fp} \
    -1str \
    -n "score" \
    -bginput \
    -pseudo 1 \
    -decimals 1 \
    -markov 1 \
    -bg_pseudo 0.01 \
    -origin "end" \
    -return limits \
    -return sites \
    -return pval \
    -uth pval 0.1 \
    ;
```

Where:

- `{input.fasta}` is the input fasta file of extracted promoters.
- `{input.matrix_fp}` is the matrix file from JASPAR for a given motif.
- `{output.fp}` is the output file of motif hits across sequences.

The output files are reprocessed to generate bed files of motif hits (dropping reported no-hits and low-score hits).

