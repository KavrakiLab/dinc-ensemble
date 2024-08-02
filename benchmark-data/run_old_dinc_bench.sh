#! /bin/bash

echo "Hello World!"

while IFS="," read -r pdbc ligf recf
do
   start=`date +%s`
   echo "PDB is : $pdbc"
   echo "Ligand is : $ligf"
   echo "Receptor is : $recf"
   mkdir -p ./benchmark_res_dinc2/${pdbc}/
   cp ./benchmark-data/${ligf} ./benchmark_res_dinc2/${pdbc}/ligand.mol2
   cp ./benchmark-data/${recf} ./benchmark_res_dinc2/${pdbc}/receptor.pdb
   cp ./defaults.yml ./benchmark_res_dinc2/${pdbc}/defaults.yml
   cd ./benchmark_res_dinc2/${pdbc}/
   /dinc/dinc -l ./ligand.mol2 -r ./receptor.pdb -p ./defaults.yml
   cd ../..
   end=`date +%s`
   runtime=$((end-start))
   echo "Total time is: $runtime"

done < <(cut -d "," -f2,6,7 ./benchmark-data/full_dataset_paths.csv | tail -n +2)


: '
import os
import pandas as pd
import time

all_times = []
dataset = pd.read_csv("./benchmark-data/full_dataset_paths.csv")
all_pdbs = []

for i, row in dataset.iterrows():
    pdb_code = row["pdb_code"].lower()
    ligand_path = row["ligand_path"]
    rec_path = row["receptor_path"]
    cmd = "dinc-ensemble dock ./benchmark-data/{} ./benchmark-data/{} ./benchmark_res/{}_bench/ --dock-type INCREMENTAL > bench_out/{}.txt".format(ligand_path, rec_path, pdb_code, pdb_code)
    print(pdb_code, ligand_path, rec_path)
    start_t = time.time()
    os.system(cmd)
    end_t = time.time()
    total_t = end_t - start_t
    print(total_t)
    all_times.append(total_t)
    all_pdbs.append(pdb_code)
    #print(cmd)
times_df = pd.DataFrame({"pdb_id": all_pdbs, "time": all_times})
times_df.to_csv("benchmark_runtime.csv")
'
