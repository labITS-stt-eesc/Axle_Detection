from csv import DictReader
import os

INPUT_FILE = 'Detection_Results.csv'

with open(INPUT_FILE, 'r') as csvfile:
    reader = DictReader(csvfile)
    for row in reader:
        txt = os.path.splitext(row["image"])[0] 
        print("txt" + txt)
        file_name = "{}.txt".format(txt)
        print("filename" + file_name)
        line = "Eixo " + row["confidence"] + " " + str(round(float(row["xmin"]))) + " " + str(round(float(row["ymin"]))) + " " + str(round(float(row["xmax"]))) + " " + str(round(float(row["ymax"]))) +'\n'
        with open(file_name, 'a') as output:
            output.write(line)

