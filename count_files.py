import numpy as np
import os
import sys
import time
import datetime

def count_files(path, verbose=True):
    done_nums = sorted([int(name[:-4]) for name in os.listdir(path)
                        if name[-5] != 'b'])

    if len(done_nums) == 0:
        print('Folder is empty!')
        return []

    starts_and_ends = [done_nums[0]]
    for num, next_num in zip(done_nums[:-1], done_nums[1:]):
        if next_num - num > 1:
            starts_and_ends.append(num)
            starts_and_ends.append(next_num)

    starts_and_ends.append(done_nums[-1])

    if verbose:
        print('Remember these ranges are inclusive - inclusive:')
        for i in range(0, len(starts_and_ends), 2):
            print(f'{starts_and_ends[i]} - {starts_and_ends[i+1]}')

    return starts_and_ends

def time_comparison(path, desired_count):
    now1 = datetime.datetime.now()
    num1 = len(os.listdir(path))
    print(f'Now, at {now1}, there are {num1} out of {desired_count} files.', flush=True)

    try:
        while True:
            time.sleep(30)

            now2 = datetime.datetime.now()
            num2 = len(os.listdir(path))

            if num2 == num1:
                continue

            rate = (num2 - num1) / (now2 - now1).total_seconds()
            time_left = (desired_count - num2) / rate
            final_time = now2 + datetime.timedelta(seconds=time_left)

            print(f'{now2}. There are {num2} files.')
            print(f'  At our average rate of {rate:.3f}/sec, we will finish by {final_time}, i.e. in {time_left} sec.', flush=True)
    except KeyboardInterrupt:
        print()
        

if __name__ == '__main__':
    assert(len(sys.argv) == 2 or len(sys.argv) == 3)
    # '/n/holyscratch01/dvorkin_lab/Users/atsang/mif/in_narrowcat_subc_m6m10_noise10_c60'
    path = sys.argv[1]
    if len(sys.argv) == 3:
        desired_count = int(sys.argv[2])
    else:
        desired_count = 100000

    count_files(path)
    time_comparison(path, desired_count)
