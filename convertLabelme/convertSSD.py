"""
Converts labelme annotations into SSD-ready training annotations


(xmin, ymin)----------------
-                          -
-                          -
-                          -
-                          -
-                          -
-                          - 
-                          -
------------------(xmax,ymax)


FRCNN Annotation Structure for Training Data 
"image1_path" "object1",class_number
"image1_path" "object2",class_number
"image1_path" "object3",class_number

image1_path xmin1,ymin1,xmax1,ymax1,class_number
image1_path xmin2,ymin2,xmax2,ymax2,class_number
image1_path xmin3,ymin3,xmax3,ymax3,class_number

E:\Projects\Python\Data\Images\worldoftrucks\Source_Images\5FD8F3.jpg,592,475,651,475,1
E:\Projects\Python\Data\Images\worldoftrucks\Source_Images\5FD8F3.jpg,393,459,438,459,1
E:\Projects\Python\Data\Images\worldoftrucks\Source_Images\5FD8F3.jpg,344,457,390,457,1

"""

import json
import glob
import os
import csv

# All files ending with .jpg on data folder
filelist = glob.glob('Source_Images/*.json')

print("Starting SSD conversion...")
with open('Train_csv.csv', 'w', newline='') as output:
    csv_writer = csv.writer(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    for image in filelist:
        
        print(image)
        image_path = os.path.abspath(image)

        with open(image) as f:
            readContent = f.read()

        parsedJSON = json.loads(readContent)

        for shape in parsedJSON['shapes']:
            if shape['label'] == 'axle':
                xmin = int(shape['points'][0][0])
                ymin = int(shape['points'][0][1])
                xmax = int(shape['points'][1][0])
                ymax = int(shape['points'][1][1])
            csv_writer.writerow([image_path[:-4]+"jpg", xmin, ymin, xmax, ymax, 1])
print("Finished")