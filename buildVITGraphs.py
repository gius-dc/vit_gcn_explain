import cv2
import torch
from utility import utility, gcn, features
from skimage import segmentation
from skimage import color
from skimage import morphology
import numpy as np
import argparse

import json
import os
from tqdm import tqdm

from transformers import pipeline
from models.models import ImageSimilarityModel
from PIL import Image

def draw_roi_grid(image, grid_size=(14, 14)):
    h, w = image.shape[:2]
    
    # Dimensioni del singolo ROI
    roi_width = w // grid_size[1]
    roi_height = h // grid_size[0]
    
    patch_id = 1
    
    # Disegna la griglia di ROI
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            x1 = j * roi_width
            y1 = i * roi_height
            x2 = (j + 1) * roi_width
            y2 = (i + 1) * roi_height
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = str(patch_id)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = x1 + (roi_width - text_size[0]) // 2
            text_y = y1 + (roi_height + text_size[1]) // 2
            cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
            patch_id += 1
    
    return image

# Funzione per caricare il modello BriaAI e ottenere le maschere di foreground e background
def get_foreground_background_mask(image, model):
    """
    Ottieni la maschera di foreground e background da BriaAI.
    """
    # Usa BriaAI per ottenere la maschera binaria
    pillow_mask = model(image, return_mask=True)
    mask = np.array(pillow_mask)
    
    # Crea maschere per foreground e background
    mask_foreground = np.zeros_like(mask, dtype=np.uint8)
    mask_foreground[mask > 0] = 255
    
    mask_background = np.zeros_like(mask, dtype=np.uint8)
    mask_background[mask == 0] = 255
    
    return mask_foreground, mask_background

# Funzione per ottenere le informazioni dal file di annotazione .txt
def getInformations(file_path):
    objects = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                values = line.strip().split()
                
                if len(values) >= 5:
                    class_id = values[0]
                    x_min, y_min, x_max, y_max = map(float, values[1:5])
                    objects.append({'id': class_id, 'b_box': [x_min, y_min, x_max, y_max]})
                else:
                    print(f'Invalid line: {line}')
    except FileNotFoundError:
        print(f'File not found: {file_path}')
    
    return objects

def run(args, isTrain=False):
    input_path = args.dataset + ("/train/" if isTrain else "/val/")
    maxImages = args.maxImages if isTrain else int(0.3 * args.maxImages)

    files = sorted([f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) and f.lower().endswith('.jpg')])

    utility.check_and_create_folders(args.output)

    # Carica il modello di segmentazione BriaAI
    briaai_model = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True, device=0)
    
    # Inizializza il modello per il Vision Transformer (ViT)
    modelTransformer = ImageSimilarityModel(device='cuda')
    
    id_x = 0
    classCounter = np.zeros(args.classes)
    
    for file in tqdm(files_sorted, desc='Extracting ROIs and Features', unit='img'):
        if np.all(classCounter == maxImages):
            break
        
        path = os.path.join(input_path, file)
        info_file = path.replace('.jpg', '.txt')
        objects = getInformations(info_file)
        
        for obj_i in objects:
            
            if int(obj_i['id']) <= args.classes and classCounter[int(obj_i['id'])-1] < maxImages:
                id_x += 1
                classCounter[int(obj_i['id'])-1] += 1
                print(classCounter)
                frame = cv2.imread(path)
                cropped_region = utility.resizeImage(frame, 800, 600)
                cropped_region_pil = Image.fromarray(cv2.cvtColor(cropped_region, cv2.COLOR_BGR2RGB))

                if not args.fullImage:
                    b_box = obj_i['b_box']
                    x, y, w, h = int(b_box[0]), int(b_box[1]), int(b_box[2]), int(b_box[3])
                    cropped_region = frame[y:y + h, x:x + w]
                    
                mask_foreground, mask_background = get_foreground_background_mask(cropped_region_pil, briaai_model)
                
                mask_output_dir_foreground = os.path.join(args.output, "masks", "foreground")
                mask_output_dir_background = os.path.join(args.output, "masks", "background")
                os.makedirs(mask_output_dir_foreground, exist_ok=True)
                os.makedirs(mask_output_dir_background, exist_ok=True)
                
                cv2.imwrite(os.path.join(mask_output_dir_foreground, f"{id_x}_mask_foreground.png"), mask_foreground)
                cv2.imwrite(os.path.join(mask_output_dir_background, f"{id_x}_mask_background.png"), mask_background)
                # cv2.imshow("Mask_foreground", mask_foreground)
                # cv2.waitKey()
                background_visualization = cv2.cvtColor(mask_background, cv2.COLOR_GRAY2BGR)
                background_with_grid = draw_roi_grid(background_visualization, grid_size=(14, 14))
                # cv2.imshow("Background with grid", background_with_grid)
                # cv2.waitKey()

                obj_id = int(obj_i['id']) - 1
                
                if args.useBackground:
                    obj_id += 1


                rois = []
                extractedFeaturesList = modelTransformer.extractEmbeddings(cropped_region, mainToken=True)
                
                grid_size = (14, 14)  # griglia 14x14

                img_height, img_width = cropped_region.shape[:2]  # altezza e larghezza dell'immagine originale
                roi_width = img_width // grid_size[1]
                roi_height = img_height // grid_size[0]

                for roiIdx, extractedFeature in enumerate(extractedFeaturesList):
                    roi_dict = {"roi_number": roiIdx, "labels": obj_id, "feature_vec": []}
                    roi_dict["feature_vec"] = extractedFeature
                    
                    if roiIdx == 0:
                        roi_dict["labels"] = obj_id + 1  # cls token
                        print(f"CLS Token - ROI Index: {roiIdx}")
                    else:
                        # Calcola l'indice di riga e colonna per il ROI, iniziando da roiIdx == 1
                        row_idx = (roiIdx - 1) // grid_size[1]  # indice riga
                        col_idx = (roiIdx - 1) % grid_size[1]   # indice colonna

                        # Calcola le coordinate e dimensioni del ROI
                        roi_x = col_idx * roi_width
                        roi_y = row_idx * roi_height
                        roi_w = roi_width
                        roi_h = roi_height

                        # print(f"roi_x: {roi_x}, roi_y: {roi_y}, roi_w: {roi_w}, roi_h: {roi_h}")
                        
                        # Estrai la regione interessata dalla maschera
                        roi_mask = mask_foreground[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                        
                        # Calcola la percentuale di pixel con valore 255 nella regione
                        foreground_pixels = np.sum(roi_mask == 255)
                        total_pixels = roi_mask.size
                        foreground_percentage = (foreground_pixels / total_pixels) * 100

                        # Controlla se il ROI è nel foreground o nel background
                        threshold_percentage = 30
                        if foreground_percentage >= threshold_percentage:
                            roi_dict["labels"] = obj_id + 1  # Classe foreground
                        else:
                            roi_dict["labels"] = 0  # Classe background

                        # print(f"ROI Index: {roiIdx}")
                        # print(f"ROI Bounding Box: ({roi_x}, {roi_y}, {roi_w}, {roi_h})")
                        # print(f"Class Label: {roi_dict['labels']}")
                        # print(f"Feature Vector Length: {len(roi_dict['feature_vec'])}")
                        # print(f"Feature Vector (first 10 values): {roi_dict['feature_vec'][:10]}")
                        # input("Press ENTER to continue")
                    rois.append(roi_dict)

                # Esporta il grafo per ciascun ROI
                gcn.exportGraph(rois, args.output, isTrain, knn_param=args.knnParams, img_number=id_x)

    return classCounter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='../coco-dataset/', 
                    help='Main Dataset Folder')
    parser.add_argument('--output', type=str, default='graphs', 
                    help='Output folder where build GCN')
    parser.add_argument('--maxImages', type=int, default=200, 
                    help='Max number of images to extract per each class')
    parser.add_argument('--knnParams', type=int, default=3, 
                    help='K-NN parameter')
    parser.add_argument('--histSize', type=int, default=10, 
                    help='Histogram Size')
    parser.add_argument('--classes', type=int, default=10, 
                    help='Number of Classes')
    parser.add_argument('--clusters', type=int, default=10, 
                    help='Number of Clusters')
    parser.add_argument('--features', type=str, choices=['vit', 'standard'], 
                    help='Select Features to be extracted. "vit"(vision transformer) or "standard"(color,intensity,texture,gradient)')
    
    parser.add_argument('--fullImage', action='store_true', default=False,
                    help='Use Full Image or Bounding Boxes')
    parser.add_argument('--uniform', action='store_true', default=False,
                    help='Uniform Batch size for all ROIs')
    parser.add_argument('--useBackground', action='store_true', default=False,
                    help='Use background (Not working!)')
    args = parser.parse_args()
    print(args)
    
    print("Working on Train set")
    trainCount = run(args, isTrain=True)

    print("Working on Test set")
    testCount = run(args, isTrain=False)

    with open(args.output + "info.txt", 'w') as log:
        log.write(f'Main informations: {str(args)}\n Train images per class counter: {str(trainCount)}\n Test images per class counter: {str(testCount)}')