## A simple python script to help us copy-paste a single 2d numpy array as a png string

import numpy as np
import matplotlib.pyplot as plt
import base64
import sys
import os

tmp_dir = '/n/home13/atsang/ml-interlopers/tmp'

if __name__ == '__main__':
    assert(len(sys.argv) == 2)

    filedir = sys.argv[1]

    myarr = np.load(filedir)

    plt.imshow(myarr, 'turbo')
    plt.colorbar()

    tmp_file = os.path.join(tmp_dir, 'pictostr.png')

    plt.savefig(tmp_file)

    with open(tmp_file, 'rb') as f:
        binary_str = base64.b64encode(f.read())

    regular_str = str(binary_str)[2:-1] # get rid of the b' and ' in b'...'

    full_str = 'data:image/png;base64,' + regular_str

    print(full_str)
    print()
    print("Click thrice above, copy, and paste into a browser. You should see a plot of the data.")
    

    ## The following code would in principle allow us to print a hyperlink, but
    ## I think our full_str is just too long for it to work.

    # print( '\x1b]8;;%s\x1b\\%s\x1b]8;;\x1b\\' %
    #        ( 'example.com' , 'This is a link' ) )

    # print( '\x1b]8;;%s\x1b\\%s\x1b]8;;\x1b\\' %
    #        ( full_str , 'This is a link' ) )
