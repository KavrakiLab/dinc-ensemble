#! /bin/bash

echo "Hello World!"

rm -r ./bench_out
mkdir ./bench_out

dinc-ensemble dock ./ligand/7ofv_ligand.mol2 ./ensemble/7ofv.pdb ./ensemble/2lw8_9.pdb ./ensemble/2lw8_10.pdb ./ensemble/2wo1.pdb ./ensemble/4w4z.pdb ./benchmark-ensemble-engens/ --dock-type INCREMENTAL > ./bench_out/bench_out.txt
