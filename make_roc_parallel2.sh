#!/bin/bash
#SBATCH -n 1               # Number of cores (should also specify -N?)
#SBATCH -t 0-2          # Runtime in D-HH:MM, minimum of 10 minutes (3-0)
#SBATCH -p sapphire,shared  # Partition to submit to (shared)
#SBATCH --mem-per-cpu=10000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o cannon_out/myoutput_%A_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e cannon_out/myerrors_%A_%a.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=ALL
#SBATCH --mail-user=atsang@g.harvard.edu
#SBATCH --account=iaifi_lab
#SBATCH --array=0-11

################################################################################
export XDG_RUNTIME_DIR=/n/scratchlfs/dvorkin_lab/atsang/tmp
echo "XDG_RUNTIME_DIR is $XDG_RUNTIME_DIR"

module load python/3.10.9-fasrc01
module load cuda/12.2.0-fasrc01

source activate torchenv
PATH=/n/sw/Mambaforge-22.11.1-4/bin/python:$PATH
################################################################################

joblist=()

## multisub test sets
for res in hst hstelt hstelt3
do
    postfix="_catval_exp10_c60_br20_nsub-1_${res}"
    joblist+=("python make_roc.py $postfix")
done

## lower intermediate concentrations
for conc in 20 25
do
    for res in hst hstelt hstelt3
    do
        postfix="_catval_exp10_c${conc}_br20_nsub1_${res}"
        joblist+=("python make_roc.py $postfix")
    done
done

## lower ellipticity (range)
for res in hst hstelt hstelt3
do
    postfix="_lecatval_exp10_c60_br20_nsub1_${res}"
    joblist+=("python make_roc.py $postfix")
done


echo "My job (${SLURM_ARRAY_TASK_ID}) is: ${joblist[${SLURM_ARRAY_TASK_ID}]}"
eval "${joblist[${SLURM_ARRAY_TASK_ID}]}"
