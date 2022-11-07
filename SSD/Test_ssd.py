import os
from PIL import Image, ImageFont, ImageDraw
from timeit import default_timer as timer



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
    
    
from keras import backend as K

from keras.preprocessing import image
from keras.optimizers import Adam
from imageio import imread
import numpy as np


from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss


img_height = 300
img_width = 300
confidence_threshold = 0.70


K.clear_session()

model = ssd_300(image_size=(img_height, img_width, 3),
                n_classes=1,
                mode='inference',
                l2_regularization=0.0005,
                scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], 
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 100, 300],
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=[123, 117, 104],
                swap_channels=[2, 1, 0],
                confidence_thresh=confidence_threshold,
                iou_threshold=0.75,
                top_k=10,
                nms_max_output_size=400)


#weights_path = 'E:\\Projects\\Python\\Axle_Detection\\SSD\\ssd300_pascal_07+12_epoch-29_loss-1.7706_val_loss-2.0828.h5'
weights_path = 'E:\\Projects\\Python\\Axle_Detection\\SSD - Copy\\ssd300_epoch-80_loss-0.4275_val_loss-0.4224.h5'

model.load_weights(weights_path, by_name=True)


adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)


model.compile(optimizer=adam, loss=ssd_loss.compute_loss)


input_images = [] 
orig_images = []


data_folder = os.path.join(get_parent_dir(0), "Data")

input_path = os.path.join(data_folder, "Test_Images")
save_path = os.path.join(data_folder, "Result_Images")
txt_path = os.path.join(data_folder, "TXT")


input_paths = GetFileList(input_path)


start_t = timer()

for item in input_paths:
    start = timer()
    
    input_images = [] 
    orig_images = []
    
    orig_images.append(imread(item))
    img = image.load_img(item, target_size=(img_height, img_width))
    img = image.img_to_array(img) 
    input_images.append(img)
    input_images = np.array(input_images)



    

    y_pred = model.predict(input_images)

    

    y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]
    
    y_pred_txt = y_pred_thresh[0]
    
    
    image2 = Image.open(item) 
    width, height = image2.size
    
    ratio_y = height/img_height
    ratio_x = width/img_width
    a=0
    
    txt = os.path.join(txt_path, os.path.splitext(os.path.basename(item))[0])     
    file_name = "{}.txt".format(txt)
    for row in y_pred_txt:
        a = a+1
        line = "Eixo " + " " + str(row[1]) + " " + str(int(round(row[2]*ratio_x))) + " " + str(int(round(row[3]*ratio_y))) + " " + str(int(round(row[4]*ratio_x))) + " " + str(int(round(row[5]*ratio_y))) + '\n'
        with open(file_name, 'a') as output:
            output.write(line)
                    
    font_path = os.path.join(os.path.dirname(__file__), "font/FiraMono-Medium.otf")
    font = ImageFont.truetype(
            font=font_path, size=np.floor(3e-2 * image2.size[1] + 0.0).astype("int32")
        )
    thickness = (image2.size[0] + image2.size[1]) // 300        

    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    print("Predicted boxes:\n")
    print('   class   conf xmin   ymin   xmax   ymax')
    print(y_pred_thresh[0])

    total = "Caminhão com {} eixo(s)".format(a)

    for box in y_pred_thresh[0]:
            draw = ImageDraw.Draw(image2)
            total_size = draw.textsize(total, font)
            label = "Eixo: {:.2f}".format(box[1])
            label_size = draw.textsize(label, font)
            
            if box[3]*ratio_y - label_size[1] >= 0:
                text_origin = np.array([box[2]*ratio_x, box[3]*ratio_y - label_size[1]])
            else:
                text_origin = np.array([box[2]*ratio_x, box[5]*ratio_y])            
            
            
            for i in range(thickness):
                draw.rectangle(
                    [box[2]*ratio_x + i, box[3]*ratio_y + i, box[4]*ratio_x - i, box[5]*ratio_y - i], outline="#800080"      
                )
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill="#800080",
            )
            
            draw.rectangle(
                [(0,0), tuple(total_size)],
                fill="#800080",
            ) 
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            draw.text((0,0), total, fill=(0, 0, 0), font=font)
            del draw
            
            
    end = timer()
    print("Time spent: {:.3f}sec".format(end - start))

        
    image2.save(os.path.join(save_path, os.path.basename(item)))

end_t = timer()


print(
		"Processed {} images in {:.1f}sec - {:.1f}FPS".format(
			len(input_paths),
			end_t - start_t,
			len(input_paths) / (end_t - start_t),
        )
    )
        
        
        
    