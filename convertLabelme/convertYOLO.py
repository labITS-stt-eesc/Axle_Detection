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


YOLO Annotation Structure for Training Data 
"image1_path" "object1",class_number "object2",class_number "object3",class_number

image_path xmin1,ymin1,xmax1,ymax1,class_number xmin2,ymin2,xmax2,ymax2,class_number xmin3,ymin3,xmax3,ymax3,class_number

E:\\20160927110318-550_color-_5BROI-1_5D-12.jpg 991,628,1110,742,0 1600,623,1721,738,0 1296,490,1405,585,0 955,489,1055,581,0

"""

import json
import glob
import os

# All files ending with .json on source_images folder
filelist = glob.glob('Source_Images/*.json')


print('Starting YOLO conversion...')
with open('data_train.txt', 'w') as output:

    for json_file in filelist:
        
        print('Converting data from', json_file)
        image_path = os.path.abspath(json_file)

        with open(json_file) as f:
            readContent = f.read()

        parsedJSON = json.loads(readContent)

        # Converting json into mAP format
        with open(image_path[:-4]+'txt', 'w') as txt:
            for shape in parsedJSON['shapes']:
                xmin = int(shape['points'][0][0])
                ymin = int(shape['points'][0][1])
                xmax = int(shape['points'][1][0])
                ymax = int(shape['points'][1][1]) 
                print("{} {} {} {} {}".format(shape['label'], xmin, ymin, xmax, ymax), file=txt)               

        # Converting json into YOLOv3 NN format
        print(image_path[:-4]+'jpg', end=' ', file=output)
        for shape in parsedJSON['shapes']:
            if shape['label'] == 'axle':
                xmin = int(shape['points'][0][0])
                ymin = int(shape['points'][0][1])
                xmax = int(shape['points'][1][0])
                ymax = int(shape['points'][1][1])
            print('{},{},{},{},0'.format(xmin, ymin, xmax, ymax), end=' ', file=output)
        print(file=output)
print('Finished!')

print('Cleaning up...')
with open('data_train.txt', 'r') as strip_input:
    strip_data = strip_input.read()
    strip_data = strip_data.replace(' \n', '\n')

with open('data_train.txt', 'w') as strip_output:
    strip_output.write(strip_data)
print('Done!')
