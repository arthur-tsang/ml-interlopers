#!/bin/bash
#SBATCH -n 1               # Number of cores (should also specify -N?)
#SBATCH -t 0-4          # Runtime in D-HH:MM, minimum of 10 minutes (3-0)
#SBATCH -p sapphire,shared,test      # Partition to submit to (shared)
#SBATCH --mem=5000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o cannon_out/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e cannon_out/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=ALL
#SBATCH --mail-user=atsang@g.harvard.edu
#SBATCH --account=iaifi_lab

# export XDG_RUNTIME_DIR=/n/scratchlfs/dvorkin_lab/atsang/tmp
# echo "XDG_RUNTIME_DIR is $XDG_RUNTIME_DIR"

export myport=11497
echo "ssh -NL $myport:$(hostname):$myport $USER@login.rc.fas.harvard.edu"

# module load python/3.8.5-fasrc01
# # module load Anaconda3/2020.11
# module load cuda/11.1.0-fasrc01


# source activate ltest
# echo "activated ltest environment"
# conda activate iaifi
# echo "activated iaifi environment"

module load python/3.10.9-fasrc01
module load cuda/12.2.0-fasrc01
conda activate torchenv
date

export PATH=/n/home13/atsang/.local/bin:$PATH

echo "PATH is $PATH"

jupyter lab --no-browser --port=$myport --ip='0.0.0.0'
