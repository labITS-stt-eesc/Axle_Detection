# importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
train = pd.read_csv("train.csv")
for col in train[["xmin", "ymin", "xmax", "ymax"]]:
    train[col] = (train[col]).apply(lambda x: round(x))
data = pd.DataFrame()
data['format'] = train['image']

# as the images are in train_images folder, add train_images before the image name
for i in range(data.shape[0]):
    data['format'][i] = 'train_images/' + data['format'][i]

# add xmin, ymin, xmax, ymax and class as per the format required
for i in range(data.shape[0]):
    data['format'][i] = data['format'][i] + ',' + str(train['xmin'][i]) + ',' + str(train['ymin'][i]) + ',' + str(train['xmax'][i]) + ',' + str(train['ymax'][i]) + ',' + train['label'][i]

data.to_csv('annotate.txt', header=None, index=None, sep=' ')
