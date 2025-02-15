#!/bin/bash
#SBATCH -n 1               # Number of cores (should also specify -N?)
#SBATCH -t 0-4          # Runtime in D-HH:MM, minimum of 10 minutes (3-0)
#SBATCH -p sapphire,shared,serial_requeue  # Partition to submit to (shared)
#SBATCH --mem=20000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o /n/home13/atsang/ml-interlopers/cannon_out/myoutput_%A.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e /n/home13/atsang/ml-interlopers/cannon_out/myerrors_%A.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=ALL
#SBATCH --mail-user=atsang@g.harvard.edu
#SBATCH --account=iaifi_lab

export XDG_RUNTIME_DIR=/n/scratchlfs/dvorkin_lab/atsang/tmp
echo "XDG_RUNTIME_DIR is $XDG_RUNTIME_DIR"

module load python/3.10.9-fasrc01
module load cuda/12.2.0-fasrc01
source activate torchenv
PATH=/n/sw/Mambaforge-22.11.1-4/bin/python:$PATH

cd /n/home13/atsang/ml-interlopers

python gen_multisub_test.py

echo "DONE!"

