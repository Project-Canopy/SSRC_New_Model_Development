import csv
import os
from PIL import Image

rootdir = 'project-canopy_data_RGB'

if __name__ == '__main__':
    f = open('pure_black_tifs.csv', 'w', encoding='UTF8', newline='')
    writer = csv.writer(f)
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            file_name = os.path.join(subdir, file)
            img = Image.open(file_name)
            if not img.getbbox():
                file_name = file_name.split('\\')
                file_name = file_name[1] + '/' + '100' + '/' + file_name[2].split('_')[0] + '/' + file_name[2][:-4] + '.tif'
                writer.writerow([file_name])
    f.close()
