# Author: Jiabin

import os
import numpy as np
from PIL import Image

def pil_open(path):
    with open(path, 'rb') as file:
        img = Image.open(file).convert('L')
        return img


def generate_masks(markers_dir, masks_dir, delta=0, start_value=1):
    for filename in os.listdir(markers_dir):
        if filename.endswith('.tif'):
            path = os.path.join(markers_dir, filename)
            marker = pil_open(path)
            marker = marker.resize((224, 224))
            mask1, mask2 = marker_to_distance_mask(marker, delta, start_value)
            np.save(os.path.join(masks_dir, f'{filename[:-4]}_inner.npy'), mask1)
            np.save(os.path.join(masks_dir, f'{filename[:-4]}_outer.npy'), mask2)
            print(f'{filename}: Generate mask.')



def marker_to_distance_mask(marker, delta=1, start_value=0):
    marker = np.array(marker)
    x_index, y_index = np.where(marker == 0)
    black_num = len(x_index)

    mask1 = np.zeros(marker.shape)
    if black_num != 0:
        for (x, y) in zip(x_index, y_index):
            mask1[x, y] = 1 / black_num

    mask2 = np.zeros(marker.shape)
    xmin, xmax = x_index[0], x_index[-1]
    ymin, ymax = y_index[0], y_index[-1]
    print(xmin, xmax, ymin, ymax)
    rows, cols = mask2.shape
    sum = 0
    for i in range(rows):
        for j in range(cols):
            x0, x1 = i - xmin, i - xmax
            y0, y1 = j - ymin, j - ymax
            dis = 0
            if x0 * x1 > 0:
                dis += min(abs(x0), abs(x1))
            if y0 * y1 > 0:
                dis += min(abs(y0), abs(y1))
            if dis != 0:
                mask2[i, j] = start_value + delta * dis
                sum += mask2[i, j]
    mask2 /= sum
    return mask1, mask2



if __name__ == '__main__':
    generate_masks('../../data/markers/6_mask', '../../data/masks/6_mask')





