import os



def detect_object(yolo, img, save_img=True, save_img_path="output"):
    try:
        image = Image.open(img)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_array = np.array(image)	
    except:
        print("File Open Error! Try again!")
        return None, None

    prediction, r_image = yolo.detect_image(image)

    if save_img:
        r_image.save(os.path.join(save_img_path, os.path.basename(img)))

    return prediction, image_array

def get_parent_dir(n=1):
    #retorna o caminho para o diretório de trabalho
    current_path = os.path.dirname(os.path.abspath(__file__))
    for k in range(n):
        current_path = os.path.dirname(current_path)
    return current_path


def GetFileList(dirName, endings=[".jpg", ".jpeg", ".png", ".mp4"]):
    # cria uma lista de todos os arquivos no diretório
    listOfFile = os.listdir(dirName)
    allFiles = list()
    
    
    # garante que todos os finais começam com .

    for i, ending in enumerate(endings):
        if ending[0] != ".":
            endings[i] = "." + ending
            
            
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + GetFileList(fullPath, endings)
        else:
            for ending in endings:
                if entry.endswith(ending):
                    allFiles.append(fullPath)
    return allFiles

from utils.yolo import YOLO, detect_video
from PIL import Image
from timeit import default_timer as timer

import pandas as pd
import numpy as np


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Diretórios de trabalho
data_folder = os.path.join(get_parent_dir(0), "Data")

image_folder = os.path.join(data_folder, "Source_Images")

input_path = os.path.join(image_folder, "Test_Images")

output_path = os.path.join(image_folder, "Test_Image_Detection_Results")
detection_results_file = os.path.join(output_path, "Detection_Results.csv")

model_folder = os.path.join(data_folder, "Model_Weights")

model_weights = os.path.join(model_folder, "trained_weights_final.h5")
model_classes = os.path.join(model_folder, "data_classes.txt")

anchors_path = os.path.join( "utils", "model_data", "yolo_anchors.txt")



save_img = True

#grau de confiança
score = 0.5

input_paths = GetFileList(input_path)



# separa imagens de videos
img_endings = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG")
vid_endings = (".mp4", ".mpeg", ".mpg", ".avi")

input_image_paths = []
input_video_paths = []
for item in input_paths:
	if item.endswith(img_endings):
		input_image_paths.append(item)
	elif item.endswith(vid_endings):
		input_video_paths.append(item)

if not os.path.exists(output_path):
	os.makedirs(output_path)


yolo = YOLO(
	**{
		"model_path": model_weights,
		"anchors_path": anchors_path,
		"classes_path": model_classes,
		"score": score,
		"gpu_num": 1,
		"model_image_size": (416, 416),
	}
)

# Cria um dataframe para os resultados
out_df = pd.DataFrame(
	columns=[
		"image",
		"image_path",
		"xmin",
		"ymin",
		"xmax",
		"ymax",
		"label",
		"confidence",
		"x_size",
		"y_size",
	]
)


class_file = open(model_classes, "r")
input_labels = [line.rstrip("\n") for line in class_file.readlines()]
print("Found {} input labels: {} ...".format(len(input_labels), input_labels))

if input_image_paths:
	print(
		"Found {} input images: {} ...".format(
			len(input_image_paths),
			[os.path.basename(f) for f in input_image_paths[:5]],
		)
	)
	start = timer()
	text_out = ""


	for i, img_path in enumerate(input_image_paths):
		print(img_path)
		prediction, image = detect_object(
			yolo,
			img_path,
			save_img=save_img,
			save_img_path=output_path,
		)
		y_size, x_size, _ = np.array(image).shape
		for single_prediction in prediction:
			out_df = out_df.append(
				pd.DataFrame(
					[
						[
							os.path.basename(img_path.rstrip("\n")),
							img_path.rstrip("\n"),
						]
						+ single_prediction
						+ [x_size, y_size]
					],
					columns=[
						"image",
						"image_path",
						"xmin",
						"ymin",
						"xmax",
						"ymax",
						"label",
						"confidence",
						"x_size",
						"y_size",
					],
				)
			)
	end = timer()
	print(
		"Processed {} images in {:.1f}sec - {:.1f}FPS".format(
			len(input_image_paths),
			end - start,
			len(input_image_paths) / (end - start),
		)
	)
	out_df.to_csv(detection_results_file, index=False)


if input_video_paths:
	print(
		"Found {} input videos: {} ...".format(
			len(input_video_paths),
			[os.path.basename(f) for f in input_video_paths[:5]],
		)
	)
	start = timer()
	for i, vid_path in enumerate(input_video_paths):
		detect_video(yolo, vid_path, output_path=output_path)
	end = timer()
	print(
		"Processed {} videos in {:.1f}sec".format(
			len(input_video_paths), end - start
		)
	)

yolo.close_session()
