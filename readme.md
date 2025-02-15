# UNets for subhalo detection

Code to go along with the paper 2401.16624.

The `paltas` directory was copied from a particular version of Sebastian Wagner-Carena's code (https://github.com/swagnercarena/paltas).

Some of the code uses code from the SmaAt-UNet paper under `ml-interlopers/SmaAtUNet`. You can find my fork at https://github.com/arthur-tsang/SmaAtUNet and clone it inside your local copy of this repo.

Apologies for the messy code, but at least I hope this can still be of some use to anyone working on reproducing my results.

## How to train

Use `bl_bigkerunet.py`. See `bl_5e5data7.sh` for an example running on the cluster.

## How to generate data (training or test)

Use `gen_together.py`. See `gen_together.sh` for an example running on the cluster.

Use the keyword `cattrain` to generate from the train part of the paltas catalogue and `catval` to generate from the validation/test part.

## How to evaluate

Use `save_nn_out.py`. See `save_nn_out5e5data7.sh` for an example.