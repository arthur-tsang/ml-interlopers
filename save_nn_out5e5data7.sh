#!/bin/bash
#SBATCH -n 1               # Number of cores (should also specify -N?)
#SBATCH -t 0-4          # Runtime in D-HH:MM, minimum of 10 minutes (3-0)
#SBATCH -p dvorkin,shared,serial_requeue  # Partition to submit to (shared)
#SBATCH --mem=20000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o cannon_out/myoutput_%A_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e cannon_out/myerrors_%A_%a.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=ALL
#SBATCH --mail-user=atsang@g.harvard.edu
#SBATCH --array=0-99
#SBATCH --account=iaifi_lab

#not SLURM_ARRAY_TASK_ID=0

export XDG_RUNTIME_DIR=/n/scratchlfs/dvorkin_lab/atsang/tmp
echo "XDG_RUNTIME_DIR is $XDG_RUNTIME_DIR"

module load python/3.10.9-fasrc01
module load cuda/12.2.0-fasrc01

source activate torchenv
PATH=/n/sw/Mambaforge-22.11.1-4/bin/python:$PATH

offset=1010000
delta=100
startidx=$((offset + delta * ${SLURM_ARRAY_TASK_ID}))
endidx=$((offset + delta * (${SLURM_ARRAY_TASK_ID} + 1)))

mname='bigkerUNet_bl_cat_sub_m8m11ninety_noise10_c60_hstelt3_5e5_drop10_b16_f32_k3'
folder_postfix='_catval_sub_m8m11ninety_noise10_c60_hstelt3'
in_size=640
out_prefix='out5e5data7'

echo "python save_nn_out.py $mname $folder_postfix $in_size $out_prefix $startidx $endidx"
python save_nn_out.py $mname $folder_postfix $in_size $out_prefix $startidx $endidx

echo "DONE"
