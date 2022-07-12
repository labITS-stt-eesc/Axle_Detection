"""
Converts labelme annotations into YOLO-ready training annotations


(xmin, ymin)----------------
-                          -
-                          -
-                          -
-                          -
-                          -
-                          - 
-                          -
------------------(xmax,ymax)


YOLO Annotation Structure Training Data 
"image1_path" "object1" "object2" "object3"

image_path xmin1,ymin1,xmax1,ymax1,class_number1 xmin2,ymin2,xmax2,ymax2,class_number2 xmin3,ymin3,xmax3,ymax3,class_number3

E:\\20160927110318-550_color-_5BROI-1_5D-12.jpg 991,628,1110,742,0 1600,623,1721,738,0 1296,490,1405,585,0 955,489,1055,581,0

"""

import json
import glob
import os

# All files ending with .jpg on data folder
filelist = glob.glob('Source_Images/*.json')


print("Starting YOLO conversion...")
with open('data_train.txt', 'w') as output:

    for image in filelist:
        
        print(image)
        image_path = os.path.abspath(image)

        with open(image) as f:
            readContent = f.read()

        parsedJSON = json.loads(readContent)

        print(image_path[:-4]+"jpg", end=" ", file=output)
        for shape in parsedJSON['shapes']:
            if shape['label'] == 'axle':
                xmin = int(shape['points'][0][0])
                ymin = int(shape['points'][0][1])
                xmax = int(shape['points'][1][0])
                ymax = int(shape['points'][0][1])
            print("{},{},{},{},0".format(xmin, ymin, xmax, ymax), end=" ", file=output)
        print(file=output)
print("Finished!")


