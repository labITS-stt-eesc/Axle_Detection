from PIL import Image
from os import path, makedirs
import os
import re
import pandas as pd
import sys


def get_parent_dir(n=1):
    """ Retorna o caminho para o diretório de trabalho """
    current_path = os.path.dirname(os.path.abspath(__file__))
    for k in range(n):
        current_path = os.path.dirname(current_path)
    return current_path


#Caminhos para as pastas

Data_Path = os.path.join(get_parent_dir(1), "Data")
train_path = os.path.join(
    Data_Path, "Source_Images", "Training_Images"
)
csv = os.path.join(train_path, "TCC-export.csv")
txt = os.path.join(train_path, "data_train.txt")

Model_Path = os.path.join(Data_Path, "Model_Weights")
classes = os.path.join(Model_Path, "data_classes.txt")



vott_df = pd.read_csv(csv)
labels = vott_df["label"].unique()
labeldict = dict(zip(labels, range(len(labels))))
vott_df.drop_duplicates(subset=None, keep="first", inplace=True)




# Arredonda floats
for col in vott_df[["xmin", "ymin", "xmax", "ymax"]]:
    vott_df[col] = (vott_df[col]).apply(lambda x: round(x))

# Cria arquivo txt
last_image = ""
txt_file = ""
vott_df["code"] = vott_df["label"].apply(lambda x: labeldict[x])

for index, row in vott_df.iterrows():
    if not last_image == row["image"]:
        txt_file += "\n" + os.path.join(train_path, row["image"]) + " "
        txt_file += ",".join(
            [
                str(x)
                for x in (row[["xmin", "ymin", "xmax", "ymax", "code"]].tolist())
            ]
        )
    else:
        txt_file += " "
        txt_file += ",".join(
            [
                str(x)
                for x in (row[["xmin", "ymin", "xmax", "ymax", "code"]].tolist())
            ]
        )
    last_image = row["image"]
file = open(txt, "w")
file.write(txt_file[1:])
file.close()


# Abre um arquivo de classes
file = open(classes, "w")

# Sort o dicionário para pegar as classes
SortedLabelDict = sorted(labeldict.items(), key=lambda x: x[1])
for elem in SortedLabelDict:
    file.write(elem[0] + "\n")
file.close()
