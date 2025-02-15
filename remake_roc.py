import argparse
import json

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def main_remake(postfix, title, print_threshold=False):
    classes = ['Background',
               '$10^6$ - $10^{6.5}$', '$10^{6.5}$ - $10^7$',
               '$10^7$ - $10^{7.5}$', '$10^{7.5}$ - $10^8$',
               '$10^8$ - $10^{8.5}$', '$10^{8.5}$ - $10^9$',
               '$10^9$ - $10^{9.5}$', '$10^{9.5}$ - $10^{10}$',
               '$10^{10}$ - $10^{10.5}$', '$10^{10.5}$ - $10^{11}$']

    matplotlib.rcParams.update({'font.size': 15})

    with open(f'tmp/rocdata{postfix}.json', 'r') as f:
        datalist = json.load(f)

    data_arrs = [np.array(l) for l in datalist]

    print('At 10% false positive:')

    for mass_bin in [1,2,3,4]:
        confidences = np.linspace(0, 1, 1000)

        true_hist = np.histogram(data_arrs[mass_bin], bins=confidences)[0]
        false_hist = np.histogram(data_arrs[0], bins=confidences)[0]

        true_raw = np.cumsum(true_hist[::-1])[::-1]
        false_raw = np.cumsum(false_hist[::-1])[::-1]

        tps = true_raw / np.sum(true_hist)
        fps = false_raw / np.sum(false_hist)

        try:
            if not print_threshold:
                print('{:.3f}'.format(tps[fps < 0.10][0]), end=', ')
            else:
                # print('[DEBUG] lengths:', len(confidences), len(fps), len(true_hist))
                # print('confidences', confidences)
                # print('fps', fps)
                print(f'{tps[fps < 0.10][0]:.3f} ({confidences[:-1][fps < 0.10][0]:.3f})', end=', ')
        except IndexError:
            print('(Index error)', end=', ')

        plt.plot(fps, tps, label=classes[mass_bin+4]+r' $M_\odot$')

    print()

    plt.legend(fontsize=14, loc='lower right')
    # plt.title('UNet')

    plt.plot([0,1], [0,1], 'grey', linestyle='--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(title)
    
    plt.tight_layout()

    plt.savefig(f'tmp/roccurve{postfix}.pdf')
    plt.savefig(f'tmp/roccurve{postfix}.png')

    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('postfix', type=str)
    parser.add_argument('title', type=str)
    parser.add_argument('--threshold', action='store_true')
    
    args = parser.parse_args()

    main_remake(args.postfix, args.title, print_threshold=args.threshold)
