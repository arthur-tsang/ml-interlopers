import os, sys
import shutil

if __name__ == '__main__':
    for item in sorted(os.listdir()):
        if os.path.isfile(item):
            continue
        elif item in ['oldmif', 'empty_folders', 'cannon_out']:
            continue
        else:
            folder_contents_len = len(os.listdir(item))

            if folder_contents_len == 0:
                shutil.move(item, 'empty_folders') # oldfile, destination
                print('moved {} to empty_folders'.format(item))
            else:
                print('folder {} contains {} elements'.format(item, folder_contents_len))
