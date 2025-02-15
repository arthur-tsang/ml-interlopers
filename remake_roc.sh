#!/bin/bash
#SBATCH -n 1               # Number of cores (should also specify -N?)
#SBATCH -t 0-2          # Runtime in D-HH:MM, minimum of 10 minutes (3-0)
#SBATCH -p dvorkin,shared  # Partition to submit to (shared)
#SBATCH --mem-per-cpu=10000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o cannon_out/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e cannon_out/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=ALL
#SBATCH --mail-user=atsang@g.harvard.edu
#SBATCH --account=iaifi_lab

################################################################################
export XDG_RUNTIME_DIR=/n/scratchlfs/dvorkin_lab/atsang/tmp
echo "XDG_RUNTIME_DIR is $XDG_RUNTIME_DIR"

module load python/3.10.9-fasrc01
module load cuda/12.2.0-fasrc01

source activate torchenv
PATH=/n/sw/Mambaforge-22.11.1-4/bin/python:$PATH
################################################################################

# python remake_roc.py _blhst 'Resolution 80 mas, Concentration 60' --threshold

# echo "Concentration 60:"
# echo "Resolution 80 mas"
# python remake_roc.py _blhst 'Resolution 80 mas, Concentration 60' --threshold
# echo "Resolution 20"
# python remake_roc.py _blhstelt 'Resolution 20 mas, Concentration 60' --threshold
# echo "Resolution 10"
# python remake_roc.py _blhstelt3 'Resolution 10 mas, Concentration 60' --threshold
# echo "Concentration 15:"
# echo "Resolution 10"
# python remake_roc.py _blhstelt3c15 'Resolution 10 mas, Concentration 15' --threshold
#python remake_roc.py _noise1 'Trained and tested on higher noise'
#python remake_roc.py _noise1bis 'Trained on fid. noise, tested on higher noise'
#python remake_roc.py _sublowm 'Tested with low-mass subhalo background'

## And then the multiple-subhalo robustness tests
# python remake_roc.py _catval_exp10_c60_br20_nsub2_hstelt3 'Two subhalos in each system'
# title=$'Multiple subhalos in each system\n(Power-law mass function)'
# python remake_roc.py _catval_exp10_c60_br20_nsub-1_hstelt3 "${title}"

## And then the systems with different concentrations

# echo "Different concentrations, resolution 10"

# python remake_roc.py _catval_exp10_c20_br20_nsub1_hstelt3 'Resolution 10 mas, Concentration 20 (trained on c=60)'
# python remake_roc.py _catval_exp10_c25_br20_nsub1_hstelt3 'Resolution 10 mas, Concentration 25 (trained on c=60)'
python remake_roc.py _catval_exp10_c25_br20_nsub1_hstelt3_retrained 'Resolution 10 mas, Concentration 25'
# python remake_roc.py _catval_exp10_c30_br20_nsub1_hstelt3 'Resolution 10 mas, Concentration 30 (trained on c=60)'
# python remake_roc.py _catval_exp10_c45_br20_nsub1_hstelt3 'Resolution 10 mas, Concentration 45 (trained on c=60)'
python remake_roc.py _catval_exp10_c45_br20_nsub1_hstelt3_retrained 'Resolution 10 mas, Concentration 45'


# echo "Different concentrations, resolution 20"

# python remake_roc.py _catval_exp10_c20_br20_nsub1_hstelt 'Resolution 20 mas, Concentration 20'
# python remake_roc.py _catval_exp10_c30_br20_nsub1_hstelt 'Resolution 20 mas, Concentration 30'
# python remake_roc.py _catval_exp10_c45_br20_nsub1_hstelt 'Resolution 20 mas, Concentration 45'

# echo "Different concentrations, resolution 80"

# python remake_roc.py _catval_exp10_c20_br20_nsub1_hst 'Resolution 80 mas, Concentration 20'
# python remake_roc.py _catval_exp10_c30_br20_nsub1_hst 'Resolution 80 mas, Concentration 30'
# python remake_roc.py _catval_exp10_c45_br20_nsub1_hst 'Resolution 80 mas, Concentration 45'

### Finally, exp1 runs

# echo "hst"
# python remake_roc.py _catval_exp1_c60_br20_nsub1_hst 'Resolution 80 mas, 1 exposure' --threshold
# echo "hstelt"
# python remake_roc.py _catval_exp1_c60_br20_nsub1_hstelt 'Resolution 20 mas, 1 exposure' --threshold
