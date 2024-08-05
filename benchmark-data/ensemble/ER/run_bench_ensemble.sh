#! /bin/bash

echo "Hello World!"

rm -r ./bench_out
mkdir ./bench_out

while IFS="," read -r pdbc ligf 
do
   start=`date +%s`
   echo "PDB is : $pdbc"
   echo "Ligand is : $ligf"
   cmd="dinc-ensemble dock ${ligf} ./ensemble/ER/ensemble/1a52_renum_fixed.pdb ./ensemble/ER/ensemble/5dtv_renum_fixed.pdb ./ensemble/ER/ensemble/7msa_renum_fixed.pdb ../benchmark-ensemble-engens/${pdbc}/ --dock-type INCREMENTAL > ./bench_out/bench_out_${pdbc}.txt"
   #dinc-ensemble dock ${ligf} ./ensemble/ER/ensemble/1a52_renum_fixed.pdb ./ensemble/ER/ensemble/5dtv_renum_fixed.pdb ./ensemble/ER/ensemble/7msa_renum_fixed.pdb ../benchmark-ensemble-engens/${pdbc}/ --dock-type INCREMENTAL > ./bench_out/bench_out_${pdbc}.txt
   echo "Command is: $cmd"
   end=`date +%s`
   runtime=$((end-start))
   echo "Total time is: $runtime"

done < <(cut -d "," -f2,3 ./ligand_paths.csv | tail -n +2)

