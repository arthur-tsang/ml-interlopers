import shutil
from pathlib import Path


if __name__ == '__main__':

    with open('files.txt') as f:
        for line in f.readlines():
            line = line.strip()
            print(f'copying {line}')
            src = Path('/n/home13/atsang/legacy-ml-interlopers') / line
            dst = Path('./') / line
            shutil.copy2(src, dst) # preserves modification dates
