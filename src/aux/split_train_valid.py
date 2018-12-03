import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

def convert_index2name(indexes, image_data, label_data):
    result_image = []
    result_label = []
    for x in indexes:
        result_image.append(image_data[x])
        result_label.append(label_data[x])
    return result_image, result_label

data_dir='../../data/ISIC2018_Task3_Training_GroundTruth.csv'

csvfile = pd.read_csv(data_dir)
original_data = csvfile.values

images_name = []
labels = []
for image_name, *a_list in original_data:
    image_name = image_name + '.jpg'
    label = np.argmax(a_list)
    images_name.append(image_name)
    labels.append(label)

skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)

total_split_index = []
for X, y in skf.split(images_name, labels):
    total_split_index.append([X,y])

for index, sets in enumerate(total_split_index):
    x, y = sets
    file_name_train = "split_data_{}_fold_train.csv".format(index+1)
    file_name_test = "split_data_{}_fold_test.csv".format(index+1)
    #print(file_name_train)
    raw_image_train, raw_labels_train = convert_index2name(x, images_name, labels)
    raw_image_test, raw_labels_test = convert_index2name(y, images_name, labels)
    raw_data_train = {'image_train': raw_image_train, 'label_train': raw_labels_train}
    raw_data_test =  {'image_test': raw_image_test, 'label_test': raw_labels_test}
    df_train = pd.DataFrame(raw_data_train, columns = ['image_train', 'label_train'])
    df_test = pd.DataFrame(raw_data_test, columns= ['image_test', 'label_test'])
#     print(df_train)
#     print(df_test)
    df_train.to_csv(file_name_train)
    df_test.to_csv(file_name_test)
    print('=> generate {} {}'.format(file_name_train, file_name_test))
