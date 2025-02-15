#!/bin/bash
#SBATCH --gres gpu:4
#SBATCH -n 16               # Number of cores (should also specify -N?)
#SBATCH -t 3-0          # Runtime in D-HH:MM, minimum of 10 minutes (3-0)
#SBATCH -p gpu_requeue,gpu
#SBATCH --mem=12000           # Memory pool for all cores (see also --mem-per-cpu) (ran with 250k, but seff said we only used 12%)
#SBATCH -o cannon_out/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e cannon_out/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=ALL
#SBATCH --mail-user=atsang@g.harvard.edu
#SBATCH --account=iaifi_lab

export XDG_RUNTIME_DIR=/n/scratchlfs/dvorkin_lab/atsang/tmp
echo "XDG_RUNTIME_DIR is $XDG_RUNTIME_DIR"

module load python/3.10.9-fasrc01
module load cuda/12.2.0-fasrc01
module load gcc/12.2.0-fasrc01 # for pooling
module load openmpi/4.1.4-fasrc01 # for pooling

source activate torchenv
PATH=/n/sw/Mambaforge-22.11.1-4/bin/python:$PATH

batchsize=16

echo "python bl_bigkerunet.py _exp10_c45_br20_nsub1_hstelt3 500000 $batchsize $SLURM_NTASKS 0.10 --pixnum 640 --fac 32 --ker 3"
python bl_bigkerunet.py _exp10_c45_br20_nsub1_hstelt3 500000 $batchsize $SLURM_NTASKS 0.10 --pixnum 640 --fac 32 --ker 3


echo "DONE"
