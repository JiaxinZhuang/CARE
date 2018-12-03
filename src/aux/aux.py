import pandas as pd
import os
from PIL import Image
import math
import numpy as np
import sys


# use Welford's algorithms to compute mean and std
def update(existingAggregate, newValue):
    count, mean, M2 = existingAggregate
    count += 1
    delta = newValue - mean
    mean += delta/count
    delta2 = newValue - mean
    M2 = M2 + delta * delta2

    return (count, mean, M2)

def finalize(existingAggregate):
    count, mean, M2 = existingAggregate
    mean, variance, sampleVavriance = (mean, M2/count, M2/(count-1))
    if count < 2:
        return float('nan')
    else:
        return mean, variance, sampleVavriance


def pil_loaders(path):
    img = Image.open(path)
    return img.convert('RGB')

    #data_dir = '../data/split_data/split_data_{}_fold_train.csv'

def compute_mean_std(filename):
    csvfile = pd.read_csv(filename, index_col=0)
    images = csvfile.values[:,0]

    data_dir = '../../data/ISIC2018_Task3_Training_Input'
    #existingAggregate_r = (0,0,0)
    #existingAggregate_g = (0,0,0)
    #existingAggregate_b = (0,0,0)
    existingAggregate = {'R':(0,0,0),
                         'G':(0,0,0),
                         'B':(0,0,0)}

    total = len(images)
    for index, x in enumerate(images):
        img_path = os.path.join(data_dir, str(x))
        img = pil_loaders(img_path)
        print(' => Process [{}/{}] {}'.format(index, total, img_path))
        for x in img.mode:
            channel = np.array(img.getchannel(x)).flatten()
            channel = channel / 255.0
            for pixel in channel:
                existingAggregate[x] = update(existingAggregate[x], pixel)


    _, mr, vr = finalize(existingAggregate['R'])
    _, mg, vg = finalize(existingAggregate['G'])
    _, mb, vb = finalize(existingAggregate['B'])
    return (mr, mg, mb), (vr, vg, vb)

if __name__=='__main__':
    output = 'mean_std_{}.csv'
    the_dir = '../../data/split_data'
    filename_train = 'split_data_{}_fold_train.csv'
    mean_red = []
    mean_green = []
    mean_blue = []
    std_red = []
    std_green = []
    std_blue = []

    print(sys.argv[1])
    term = int(sys.argv[1])

    for i in range(term, term+1):
        output = output.format(i)
        filepath = os.path.join(the_dir, filename_train)
        filepath = filepath.format(i)
        # train
        print('=> Processing from {}'.format(filepath))
        (mr,mg,mb), (vr,vg,vb) = compute_mean_std(filepath)
        vr = math.sqrt(vr)
        vg = math.sqrt(vg)
        vb = math.sqrt(vb)
        mean_red.append(mr)
        mean_green.append(mg)
        mean_blue.append(mb)
        std_red.append(vr)
        std_green.append(vg)
        std_blue.append(vb)
        print('{} {} {} {} {} {}'.format(mr, mg, mb, vr, vg, vb))

    raw_data = {'mean_red':mean_red, 'mean_green':mean_green, 'mean_blue': mean_blue,
            'std_red':std_red, 'std_green': std_green, 'std_blue': std_blue}

    df = pd.DataFrame(raw_data, columns=['mean_red', 'mean_green', 'mean_blue',\
         'std_red', 'std_green', 'std_blue'])
    df.to_csv(output)


