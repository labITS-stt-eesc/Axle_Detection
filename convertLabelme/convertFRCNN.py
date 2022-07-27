"""
Converts labelme annotations into FRCNN-ready training annotations


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
"image1_path" "object1",class_name
"image1_path" "object2",class_name
"image1_path" "object3",class_name

image1_path xmin1,ymin1,xmax1,ymax1,class_name
image1_path xmin2,ymin2,xmax2,ymax2,class_name
image1_path xmin3,ymin3,xmax3,ymax3,class_name

E:\Projects\Python\Data\Images\worldoftrucks\Source_Images\5FD8F3.jpg,592,475,651,475,axle
E:\Projects\Python\Data\Images\worldoftrucks\Source_Images\5FD8F3.jpg,393,459,438,459,axle
E:\Projects\Python\Data\Images\worldoftrucks\Source_Images\5FD8F3.jpg,344,457,390,457,axle

"""

import json
import glob
import os

# All files ending with .json on source_images folder
filelist = glob.glob('Source_Images/*.json')

print("Starting FRCNN conversion...")
with open('annotate.txt', 'w') as output:

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
            print("{},{},{},{},{},axle".format(image_path[:-4]+"jpg", xmin, ymin, xmax, ymax), file=output)
print("Finished")