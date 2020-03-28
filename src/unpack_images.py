import os
import shutil

if __name__ == '__main__':
    input = './images/test'
    out = './processed/test'

    if not os.path.exists(out):
        os.makedirs(out)

    for folder in [x for x in os.listdir(input) if x[0] != '.']:
        file_names = [x for x in os.listdir(os.path.join(input, folder, 'images')) if x[0] != '.']
        for file in file_names:
            shutil.copy(os.path.join(input, folder, 'images', file), os.path.join(out, file))
