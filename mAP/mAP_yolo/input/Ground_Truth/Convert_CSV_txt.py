from csv import DictReader
import os

INPUT_FILE = 'Test_csv.csv'

with open(INPUT_FILE, 'r') as csvfile:
    reader = DictReader(csvfile)
    for row in reader:
        txt = os.path.splitext(row["image_name"])[0] 
        file_name = "{}.txt".format(txt)
        line = "Eixo " + " " + str(round(float(row["xmin"]))) + " " + str(round(float(row["ymin"]))) + " " + str(round(float(row["xmax"]))) + " " + str(round(float(row["ymax"]))) +'\n'
        with open(file_name, 'a') as output:
            output.write(line)


