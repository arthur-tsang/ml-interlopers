import numpy as np
import matplotlib.pyplot as plt

import os
import sys

# from gen_kpert import main as gen_kpert_main

def path_test(in_path):
    """Tests kpert files to make sure that they are not full of nans or all zero."""

    # in_path looks like stuff/0.npy

    # def remake_file(should_remove=True):
    #     """Remake the file at `in_path`"""
    #     file_number = int(os.path.split(in_path)[-1][:-4])
    #     if should_remove:
    #         os.remove(in_path)
    #     gen_kpert_main(param_keyword, file_number, file_number + 1)

    try:
        in_arr = np.load(in_path)

        # if np.sum(np.isnan(in_arr)) > 0 or np.sum(in_arr**2) < 1e-10:
        if np.sum(np.isnan(in_arr)) > 0:
            print(in_path, 'has nan')
            os.remove(in_path)

    except ValueError:
        print('Could not load {} (ValueError)'.format(in_path))
        os.remove(in_path)

    except FileNotFoundError:
        pass
    #     print('File not found:', in_path)



if __name__ == '__main__':
    root_dir = '/n/holyscratch01/dvorkin_lab/Users/atsang/mif'

    # in_dir_list = [os.path.join(root_dir, 'in_catval_subc_m8m10_noise10_c60_fakeelt'),
    #                os.path.join(root_dir, 'coord_catval_subc_m8m10_noise10_c60_fakeelt'),
    #                os.path.join(root_dir, 'params_catval_subc_m8m10_noise10_c60_fakeelt')]

    folders = ['coord_catval_exp10_c60_br5_nsub1_hst',
               'coord_catval_exp10_c60_br5_nsub1_hstelt',
               'coord_catval_exp10_c60_br5_nsub1_hstelt3',
               'in_catval_exp10_c60_br5_nsub1_hst',
               'in_catval_exp10_c60_br5_nsub1_hstelt',
               'in_catval_exp10_c60_br5_nsub1_hstelt3',
               'params_catval_exp10_c60_br5_nsub1_hst',
               'params_catval_exp10_c60_br5_nsub1_hstelt',
               'params_catval_exp10_c60_br5_nsub1_hstelt3']

    in_dir_list = [os.path.join(root_dir, folder) for folder in folders]
    
    assert(len(sys.argv) == 3)
    startidx = int(sys.argv[1])
    endidx = int(sys.argv[2])

    for i in range(startidx, endidx):
        if i % 200 == 0:
            print(i, flush=True)

        for in_dir in in_dir_list:
            in_path = os.path.join(in_dir, '{}.npy'.format(i))
            path_test(in_path)

    print('Tested {} to {}'.format(startidx, endidx))
