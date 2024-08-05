#! /bin/bash

echo "Hello World!"

while IFS="," read -r pdbc ligf recf
do
   start=`date +%s`
   echo "PDB is : $pdbc"
   echo "Ligand is : $ligf"
   echo "Receptor is : $recf"
   cmd="dinc-ensemble dock ${ligf} ${recf} ../benchmark-ensemble-engens/${pdbc}/ --dock-type INCREMENTAL > bench_out_${pdbc}.txt"
   #dinc-ensemble dock ${ligf} ${recf} ../benchmark-ensemble-engens/${pdbc}/ --dock-type INCREMENTAL > bench_out_${pdbc}.txt
   echo "Command is: $cmd"
   end=`date +%s`
   runtime=$((end-start))
   echo "Total time is: $runtime"

done < <(cut -d "," -f2,3,4 ./full_cdk2_eng_ens_dataset.csv | tail -n +2)

