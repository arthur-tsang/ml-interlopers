#!/bin/bash
#SBATCH -n 1               # Number of cores (should also specify -N?)
#SBATCH -t 0-8          # Runtime in D-HH:MM, minimum of 10 minutes (3-0)
#SBATCH -p test,shared,serial_requeue  # Partition to submit to (shared)
#SBATCH --mem=2000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o cannon_out/myoutput_%A_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e cannon_out/myerrors_%A_%a.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=ALL
#SBATCH --mail-user=atsang@g.harvard.edu

#not SBATCH --array=0-3

SLURM_ARRAY_TASK_ID=0

export XDG_RUNTIME_DIR=/n/scratchlfs/dvorkin_lab/atsang/tmp
echo "XDG_RUNTIME_DIR is $XDG_RUNTIME_DIR"

module load python/3.8.5-fasrc01
module load cuda/11.1.0-fasrc01
# source activate va100
source activate ltest
which python
# PATH=/n/home13/atsang/.conda/envs/va100/bin:$PATH
# which python
date

offset=1010000
delta=10000
startidx=$((offset + delta * ${SLURM_ARRAY_TASK_ID}))
endidx=$((offset + delta * (${SLURM_ARRAY_TASK_ID} + 1)))


echo "Running problem_finder from $startidx to $endidx"
python -E problem_finder.py $startidx $endidx
date

echo "DONE"
