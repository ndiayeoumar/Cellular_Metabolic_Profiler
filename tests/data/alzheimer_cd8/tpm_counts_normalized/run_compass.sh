#!/bin/bash
#SBATCH --job-name=compass_alzheimer_cd8
#SBATCH -c 20
#SBATCH --mem-per-cpu=4096MB

compass --data ../data/linear_gene_expression_matrix.tsv --num-processes 20 --species mus_musculus --lambda 0.25 --num-neighbors 10 --output-dir lamda_met_k10 --calc-metabolites
