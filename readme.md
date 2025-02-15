# UNets for subhalo detection

Code to go along with the paper 2401.16624.

## How to train

Use `bl_bigkerunet.py`. See `bl_5e5data7.sh` for an example running on the cluster.

## How to generate data (training or test)

Use `gen_together.py`. See `gen_together.sh` for an example running on the cluster.

Use the keyword `cattrain` to generate from the train part of the paltas catalogue and `catval` to generate from the validation/test part.

## How to evaluate

Use `save_nn_out.py`. See `save_nn_out5e5data7.sh` for an example.