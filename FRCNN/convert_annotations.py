import json
import opencv

with open('20160927100754-550_color-[ROI-1]-8.json') as f:
    readContent = f.read()

parsedJSON = json.loads(readContent)

for shape in parsedJSON["shapes"]:
    if shape["label"] == "Axle":
        print(shape)



#print(parsedJSON["shapes"][0]["label"])
