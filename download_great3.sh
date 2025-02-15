#!/bin/bash
#SBATCH -n 1               # Number of cores (should also specify -N?)
#SBATCH -t 3-0          # Runtime in D-HH:MM, minimum of 10 minutes (3-0)
#SBATCH -p shared,dvorkin,serial_requeue  # Partition to submit to (shared)
#SBATCH --mem=2000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o cannon_out/myoutput_%A_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e cannon_out/myerrors_%A_%a.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=ALL
#SBATCH --mail-user=atsang@g.harvard.edu
#not SBATCH --array=0-39


# curl https://hsc-release.mtk.nao.ac.jp/archive/pdr1_incremental/parent_best_processed.tar.gz > parent_best_processed.tar.gz

# curl https://zenodo.org/record/3242143/files/COSMOS_23.5_training_sample.tar.gz > COSMOS_23.5_training_sample.tar.gz

curl https://zenodo.org/records/3242143/files/COSMOS_23.5_training_sample.tar.gz?download=1 > COSMOS_23.5_training_sample.tar.gz

echo "unzipping..."
tar -xvf COSMOS_23.5_training_sample.tar.gz
