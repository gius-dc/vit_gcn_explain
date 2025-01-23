import cv2
import numpy as np
import os
import shutil
import random
import glob 
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

litter_classes = ['bottle','can','fish_net','paper','ropes', 'trash', 'vehicle_tyre']
litter_classes_background = ['background','bottle','can','fish_net','paper','ropes', 'trash', 'vehicle_tyre']
classes_background = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]
classes = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]
bird_classes = ['Black Footed Albatross', 'Bobolink', 'Brewer Blackbird', 'Cardinal', 'Crested Auklet', 'Gray Catbird', 'Groove Billed Ani', 'Indigo Bunting', 
                'Laysan Albatross', 'Lazuli Bunting', 'Least Auklet', 'Painted Bunting', 'Parakeet Auklet', 'Red Winged Blackbird' 'Rhinoceros Auklet', 'Rusty Blackbird', 'Sooty Albatross', 'Spotted Catbird', 'Yellow Breasted Chat', 'Yellow Headed Blackbird']
bird_species_background = [
    "Background",
    "Black-footed Albatross",
    "Laysan Albatross",
    "Sooty Albatross",
    "Groove-billed Ani",
    "Crested Auklet",
    "Least Auklet",
    "Parakeet Auklet",
    "Rhinoceros Auklet",
    "Brewer Blackbird",
    "Red-winged Blackbird",
    "Rusty Blackbird",
    "Yellow-headed Blackbird",
    "Bobolink",
    "Indigo Bunting",
    "Lazuli Bunting",
    "Painted Bunting",
    "Cardinal",
    "Spotted Catbird",
    "Gray Catbird",
    "Yellow-breasted Chat",
    "Eastern Towhee",
    "Chuck-will's-Widow",
    "Brandt Cormorant",
    "Red-faced Cormorant",
    "Pelagic Cormorant"
]
bird_species = [
    "Black-footed Albatross",
    "Laysan Albatross",
    "Sooty Albatross",
    "Groove-billed Ani",
    "Crested Auklet",
    "Least Auklet",
    "Parakeet Auklet",
    "Rhinoceros Auklet",
    "Brewer Blackbird",
    "Red-winged Blackbird",
    "Rusty Blackbird",
    "Yellow-headed Blackbird",
    "Bobolink",
    "Indigo Bunting",
    "Lazuli Bunting",
    "Painted Bunting",
    "Cardinal",
    "Spotted Catbird",
    "Gray Catbird",
    "Yellow-breasted Chat",
    "Eastern Towhee",
    "Chuck-will's-Widow",
    "Brandt Cormorant",
    "Red-faced Cormorant",
    "Pelagic Cormorant"
]
float_formatter = "{:.2f}".format
fourFloat_formatter = "{:.4f}".format

def check_and_create_folders(main_folder):

    # Subfolders to check/create
    subfolders = ["adjacency_train", "adjacency_test", "graphs_train","graphs_test"]

    # Check if the main folder exists
    if not os.path.exists(main_folder):
        print(f"The folder '{main_folder}' does not exist. Creating now...")
        os.makedirs(main_folder)
    else:
        print(f"The folder '{main_folder}' already exists.")

    # Check and create subfolders
    for subfolder in subfolders:
        subfolder_path = os.path.join(main_folder, subfolder)
        if not os.path.exists(subfolder_path):
            print(f"The subfolder '{subfolder}' does not exist. Creating now...")
            os.makedirs(subfolder_path)
        else:
            print(f"The subfolder '{subfolder}' already exists.")

def split_data(input_folder, output_folder, split_ratio=0.3):
    # Trova tutti i file con estensione .pkl nella cartella di input
    pkl_files = glob.glob(os.path.join(input_folder, '*.pkl'))

    # Calcola il numero di file da spostare nella cartella di test
    num_files_test = int(len(pkl_files) * split_ratio)
    num_files_train = len(pkl_files) - num_files_test

    folders = [output_folder + 'graphs_train', output_folder + 'graphs_test', output_folder + 'adjacency_train',  output_folder + 'adjacency_test']

    # Creazione delle cartelle se non esistono
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Mescola casualmente l'elenco dei file
    random.shuffle(pkl_files)

    # Copia i file nella cartella Train e Test secondo il rapporto specificato
    for i, file_path in enumerate(pkl_files):
        path = file_path.split(".")[0]
        if i < num_files_train:
            print(os.path.join(folders[0], os.path.basename(path + '.pkl')))
            print(os.path.join(folders[2], os.path.basename(path + '.npz')))
            shutil.copy(path + '.pkl', os.path.join(folders[0], os.path.basename(path + '.pkl')))
            shutil.copy(path + '.npz', os.path.join(folders[2], os.path.basename(path + '.npz')))
        else:
            print(os.path.join(folders[1], os.path.basename(path + '.pkl')))
            print(os.path.join(folders[3], os.path.basename(path + '.npz')))
            shutil.copy(path + '.pkl', os.path.join(folders[1], os.path.basename(path + '.pkl')))
            shutil.copy(path + '.npz', os.path.join(folders[3], os.path.basename(path + '.npz')))

def extractFrames(pred):
    boxes = pred[:, :4]  # Estrai le coordinate x, y, w, h
    confidences = pred[:, 4]
    obj = []
    for box, confidence, class_id in zip(boxes, confidences, pred[:, 5]):
        x, y, w, h = box.tolist()
        x, y, w, h = int(x), int(y), int(w), int(h)
        score = float_formatter(confidence.item())
        
        if float(score) >= 0.6:
            obj.append({
                'class_id': int(class_id),
                'class': classes[int(class_id)],
                'score': score,
                'x': x,
                'y': y,
                'w': w,
                'h': h
            })
    return obj

def experimentFolder(path):
    # Cartella di base in cui controllare le sottocartelle
    cartella_base = path

    # Ottieni elenco delle cartelle nella cartella di base
    sottocartelle = [nome for nome in os.listdir(cartella_base) if os.path.isdir(os.path.join(cartella_base, nome))]

    # Trova il numero massimo nella lista delle sottocartelle
    if sottocartelle:
        numeri_cartelle = [int(nome[4:]) for nome in sottocartelle if nome.startswith('exp')]
        nuovo_numero_cartella = max(numeri_cartelle) + 1 if numeri_cartelle else 1
    else:
        nuovo_numero_cartella = 1

    # Crea la nuova cartella
    nuova_cartella = os.path.join(cartella_base, f"exp{nuovo_numero_cartella:02d}")
    os.makedirs(nuova_cartella)
    print("Created folder", nuova_cartella)
    return(nuova_cartella)

from skimage.segmentation import slic, mark_boundaries
def drawROIs(image, segments, output_path):
    # Applica l'algoritmo SLIC per ottenere le regioni

    # Inizializza l'immagine con i numeri delle regioni
    segmentation_with_numbers = image.copy()

    # Numerazione delle regioni
    for i in np.unique(segments):
        segment_mask = (segments == i).astype(np.uint8)
        contours, _ = cv2.findContours(segment_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = contours[0]
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(segmentation_with_numbers, str(i), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Evidenzia i bordi delle regioni dopo aver aggiunto i numeri
    segmentation_with_boundaries = mark_boundaries(segmentation_with_numbers, segments, color=(1, 0, 0), mode='thick')
    cv2.imwrite(output_path, segmentation_with_boundaries)


def exportROIimage(cropped_region, segments, output_path):
    
    roi_number = 0
    contours_image = np.copy(cropped_region)
    for segVal in np.unique(segments):
        print(segVal)
        if segVal == 0:
            continue
        # Crea una maschera per il segmento corrente
        mask = np.zeros(cropped_region.shape[:2], dtype="uint8")
        mask[segments == segVal] = 255
        
        # Trova i contorni della ROI
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Disegna i contorni sulla matrice dell'immagine
        cv2.drawContours(contours_image, contours, -1, (0, 255, 0), 1)  # Cambia il colore e lo spessore a tuo piacimento
        
        # Calcola il centroide della ROI per posizionare il numero
        M = cv2.moments(contours[0])
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # Aggiungi il numero al centroide della ROI
            cv2.putText(contours_image, str(roi_number), (cX - 10, cY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            roi_number += 1
    plt.figure(figsize=(8, 8))
    plt.title("ROIs: " + str(len(np.unique(segments))))
    plt.imshow(cv2.cvtColor(contours_image, cv2.COLOR_BGR2RGB))
    plt.savefig(output_path, bbox_inches='tight')
    #print("Saved")

def heatmapROIimage(image, segments, roi_array, idx, title):
    intensity_values = roi_array

    # Normalizza i valori tra 0 e 1
    normalized_values = (intensity_values - np.min(intensity_values)) / (np.max(intensity_values) - np.min(intensity_values))

    # Inizializza una matrice per disegnare la heatmap sulle ROI
    heatmap = np.zeros(image.shape[:2])

    # Assegna i valori normalizzati alla posizione della ROI
    for i, segVal in enumerate(np.unique(segments)):
        mask = np.zeros(image.shape[:2], dtype="uint8")
        mask[segments == segVal] = normalized_values[i] * 255  # Moltiplica per 255 per ottenere valori tra 0 e 255
        heatmap += mask

    # Applica la mappa dei colori alla heatmap
    heatmap = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_COOL)

    # Sovrapponi la heatmap sull'immagine originale
    overlay = cv2.addWeighted(image, 0.5, heatmap, 0.7, 0)

    # Visualizza l'immagine con la heatmap
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), cmap='cool')
    plt.title(title + " feature heatmaps")
    cbar = plt.colorbar(label='Score (Blue - Red)', ticks=np.linspace(0, 1, num=6))

    plt.savefig('40features/' + str(idx) + '_'+title+'.png', bbox_inches='tight')
    plt.axis('off')

import networkx as nx
import scipy.sparse as sp

def drawFeaturesMatrix(features_matrix, name):
    feature_0 = features_matrix[:, 0]
    feature_1 = features_matrix[:, 1]

    plt.scatter(feature_0, feature_1, c='skyblue', alpha=0.5)
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    plt.title('Scatter plot tra Feature 0 e Feature 1')
    plt.savefig(name + '.png')
    plt.close() 

from sklearn.manifold import TSNE

def drawData(data, name):
    # Creazione dell'oggetto NetworkX Graph
    G = nx.Graph()
    edge_index = data.edge_index.cpu().numpy()  # Converti edge_index in un array NumPy

    # Aggiungi nodi e archi al grafo
    G.add_nodes_from(range(data.num_nodes))
    G.add_edges_from(edge_index.T.tolist())  # Transponi edge_index e converte in una lista di tuple

    # Disegna il grafo utilizzando le features come colori dei nodi
    node_colors = data.x[:, 0].tolist()  # Usa la prima feature come colore dei nodi

    pos = nx.spring_layout(G)  # Layout del grafo
    nx.draw(G, pos, with_labels=True, node_color=node_colors, cmap='viridis', node_size=800, font_weight='bold')
    plt.title("Grafo con features associate ai nodi")
    plt.savefig(name + '.png')
    plt.close() 

def drawGraph(A, name):
    adj_matrix = sp.coo_matrix((A.data, (A.row, A.col)), shape=A.shape).toarray()

    # Ottenere il numero di nodi nel grafo
    num_nodes = adj_matrix.shape[0]

    # Creare un oggetto NetworkX Graph
    G = nx.Graph()

    # Aggiungere nodi al grafo
    G.add_nodes_from(range(num_nodes))

    # Trovare gli archi basati sulla matrice di adiacenza
    edges = list(zip(A.row, A.col))  # Ottenere gli indici dei valori non nulli

    # Aggiungere gli archi al grafo
    G.add_edges_from(edges)

    # Disegnare il grafo
    pos = nx.spring_layout(G)  # Layout del grafo
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=800, font_weight='bold', font_color='black')
    plt.title("Grafo k-nearest neighbors da matrice COO")
    plt.savefig(name + '.png')
    plt.close() 

import torch
import pandas as pd

def getFeaturesInformations(node_mask, feature_len):
    
    if node_mask is None:
        raise ValueError(f"The attribute 'node_mask' is not available ")
    if node_mask.dim() != 2 or node_mask.size(1) <= 1:
        raise ValueError(f"Cannot compute feature importance for "
                            f"object-level 'node_mask' "
                            f"(got shape {node_mask.size()})")


    mean_node = torch.mean(node_mask, dim=1)
    dev_std = torch.std(node_mask, dim=1)
    var_node = torch.var(node_mask, dim=1)
    #print("Mean:",mean_node)
    #print("DevStd:", dev_std)
    #print("Variance", var_node)
    return mean_node, dev_std, var_node

    score = node_mask.sum(dim=0)
    labels = feat_labels
    if len(labels) != score.numel():
        raise ValueError(f"The number of labels (got {len(labels)}) must "
                         f"match the number of scores (got {score.numel()})")

    score = score.cpu().numpy()
    idx = 0
    features_mean = []
    for i in range(0, len(score), feature_len):
        mean_10 = sum(score[i:i+feature_len]) / float(feature_len)
        features_mean.append(mean_10)
    if feature_len != 768:
        return features_mean[0], features_mean[1], features_mean[2], features_mean[3]
    else:
        return features_mean[0]
    
def getFeatures40(node_mask):
    step = 10
    if node_mask is None:
        raise ValueError(f"The attribute 'node_mask' is not available ")
    if node_mask.dim() != 2 or node_mask.size(1) <= 1:
        raise ValueError(f"Cannot compute feature importance for "
                            f"object-level 'node_mask' "
                            f"(got shape {node_mask.size()})")

    gradient = []
    intensity = []
    color = []
    texture = []
    
    for node in node_mask:

        gradient_mean = sum(node[0:9]) / float(step)
        intensity_mean = sum(node[10:19]) / float(step)
        color_mean = sum(node[20:29]) / float(step)
        texture_mean = sum(node[30:39]) / float(step)

        gradient.append(round(gradient_mean.item(), 4))
        intensity.append(round(intensity_mean.item(), 4))
        color.append(round(color_mean.item(), 4))
        texture.append(round(texture_mean.item(), 4))

    return gradient, intensity, color, texture

    
def getFeaturesNode(node_mask, feature_len):
    step = 10
    if node_mask is None:
        raise ValueError(f"The attribute 'node_mask' is not available ")
    if node_mask.dim() != 2 or node_mask.size(1) <= 1:
        raise ValueError(f"Cannot compute feature importance for "
                            f"object-level 'node_mask' "
                            f"(got shape {node_mask.size()})")

    feat_labels = range(node_mask.size(1))
    print("node_mask", node_mask.shape)
    score = node_mask.sum(dim=0)
    print("sum", score.shape)
    labels = feat_labels
    if len(labels) != score.numel():
        raise ValueError(f"The number of labels (got {len(labels)}) must "
                         f"match the number of scores (got {score.numel()})")

    score = score.cpu().numpy()

    idx = 0
    features_mean = []
    print("LEN SOCRE", len(score))
    for i in range(0, len(score), step):
        mean_10 = sum(score[i:i+step]) / float(step)
        print(mean_10)
        features_mean.append(mean_10)

    if feature_len != 768:
        return features_mean[0], features_mean[1], features_mean[2], features_mean[3]
    else:
        return features_mean[0], score

def coco2yolo(b_box, image_w, image_h):

    x1 = int(b_box[0])
    y1 = int(b_box[1])
    w = int(b_box[2])
    h = int(b_box[3])
    print(x1,y1,w,h)

    # Check for borders
    if x1 > image_w:
        x1 = image_w
    if y1 > image_h:
        y1 = image_h
    if w > image_w:
        w = image_w
    if h > image_h:
        h = image_h 

    return [int((2*x1 + w)/(2*image_w)) , int((2*y1 + h)/(2*image_h)), int(w/image_w), int(h/image_h)]

def yolo2coco(b_box, image_width, image_height):
    x_center, y_center, width, height = b_box[0],b_box[1],b_box[2],b_box[3]
    """
    Converts bounding box coordinates from YOLO format to COCO format.
    
    Arguments:
    - x_center: x-coordinate of the bounding box center in YOLO format (normalized value between 0 and 1).
    - y_center: y-coordinate of the bounding box center in YOLO format (normalized value between 0 and 1).
    - width: Width of the bounding box in YOLO format (normalized value between 0 and 1).
    - height: Height of the bounding box in YOLO format (normalized value between 0 and 1).
    - image_width: Width of the original image.
    - image_height: Height of the original image.
    
    Returns a tuple containing the bounding box coordinates in COCO format (x, y, width, height).
    """
    # Normalize coordinates > 1.0
    if x_center > 1.0:
        x_center = 1.0
    if y_center > 1.0:
        y_center = 1.0
    if width > 1.0:
        width = 1.0
    if height > 1.0:
        height = 1.0

    # Calculate COCO format bounding box coordinates
    x = int((x_center - width / 2) * image_width)
    y = int((y_center - height / 2) * image_height)
    w = int(width * image_width)
    h = int(height * image_height)

    return x, y, w, h

def extractBackground(segments, image, output_path, threshold_size=5000):
    # Calcola le dimensioni delle ROI
    roi_sizes = {i: np.sum(segments == i) for i in np.unique(segments)}

    # Trova le ROI che superano la soglia di dimensione
    background_rois = [i for i, size in roi_sizes.items() if size > threshold_size]

    # Crea una maschera per le ROI da rimuovere (sfondo)
    background_mask = np.isin(segments, background_rois)

    # Applica la maschera per rimuovere le ROI (sfondo)
    image_without_background = np.where(background_mask[:, :, np.newaxis], 255, image)

    cv2.imwrite(output_path, image_without_background)

def fillEmptySlots(rois, n_clusters):
    while len(rois) < n_clusters:
        ultimo_elemento_clonato = rois[-1].copy()  # Clona l'ultimo elemento
        rois.append(ultimo_elemento_clonato)
    
    if len(rois) > n_clusters:
        rois = rois[:n_clusters]
    
    return rois

def resizeImage(frame, larghezza_massima=800, altezza_massima=600):

    # Controlla le dimensioni dell'immagine
    altezza, larghezza, _ = frame.shape

    # Verifica se le dimensioni superano il massimo consentito
    if larghezza > larghezza_massima or altezza > altezza_massima:
        # Calcola i nuovi valori di dimensione rispettando il rapporto d'aspetto
        rapporto_aspetto = larghezza / altezza
        nuova_larghezza = larghezza_massima
        nuova_altezza = int(larghezza_massima / rapporto_aspetto)

        # Rimpicciolisci l'immagine
        frame = cv2.resize(frame, (nuova_larghezza, nuova_altezza))

    return frame

def calculatePatchDimensions(image, original_size=224, patch_size=16):
    # Ottieni le dimensioni dell'immagine
    height, width, _ = image.shape
    # Calcola il fattore di scala tra la dimensione originale e quella di destinazione
    scale_x = width / original_size
    scale_y = height / original_size
    # Calcola il numero di patch per lato nell'immagine originale
    num_patches = original_size // patch_size
    patch_coordinates = []

    for i in range(num_patches):
        for j in range(num_patches):
            # Coordinate top-left e bottom-right della patch nell'immagine originale
            top_left_x_orig = j * patch_size
            top_left_y_orig = i * patch_size
            bottom_right_x_orig = (j + 1) * patch_size
            bottom_right_y_orig = (i + 1) * patch_size

            # Scala le coordinate per l'immagine di destinazione
            top_left_x = int(top_left_x_orig * scale_x)
            top_left_y = int(top_left_y_orig * scale_y)
            bottom_right_x = int(bottom_right_x_orig * scale_x)
            bottom_right_y = int(bottom_right_y_orig * scale_y)

            patch_coordinates.append(((top_left_x, top_left_y), (bottom_right_x, bottom_right_y)))

    return patch_coordinates

def drawPatchGrid(image, patch_coordinates, imagePath=None):
    def get_contrasting_color(color):
        # Calcola la luminanza del colore (formula per percezione umana)
        luminance = (0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]) / 255
        # Ritorna nero o bianco a seconda della luminanza
        return (0, 0, 0) if luminance > 0.5 else (255, 255, 255)

    # Calcola il colore medio globale dell'immagine
    avg_color = cv2.mean(image)[:3]
    # Ottieni il colore contrastante per griglia e testo
    contrasting_color = get_contrasting_color(avg_color)

    # Copia dell'immagine per sovrapporre la griglia
    overlay = image.copy()

    # Disegna i rettangoli della griglia
    for (top_left, bottom_right) in patch_coordinates:
        cv2.rectangle(overlay, top_left, bottom_right, contrasting_color, 1)

    # Aggiungi i numeri alle patch
    patch_num = 1
    for (top_left, _) in patch_coordinates:
        text_position = (top_left[0] + 2, top_left[1] + 12)
        cv2.putText(overlay, str(patch_num), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, contrasting_color, 1, cv2.LINE_AA)
        patch_num += 1

    # Salva l'immagine se specificato
    if imagePath:
        cv2.imwrite(imagePath, overlay)

    return overlay

def drawPatches(image, imagePath, original_size = 224, patch_size = 16):
    # Ottieni le dimensioni dell'immagine
    height, width, _ = image.shape

    # Calcola il fattore di scala tra la dimensione originale e quella di destinazione
    scale_x = width / original_size
    scale_y = height / original_size

    # Calcola il numero di patch per lato nell'immagine originale
    num_patches = original_size // patch_size
    patch_num = 0

    for i in range(num_patches):
        for j in range(num_patches):
            # Coordinate top-left e bottom-right della patch nell'immagine originale
            top_left_x_orig = j * patch_size
            top_left_y_orig = i * patch_size
            bottom_right_x_orig = (j + 1) * patch_size
            bottom_right_y_orig = (i + 1) * patch_size
            
            # Scala le coordinate per l'immagine di destinazione
            top_left_x = int(top_left_x_orig * scale_x)
            top_left_y = int(top_left_y_orig * scale_y)
            bottom_right_x = int(bottom_right_x_orig * scale_x)
            bottom_right_y = int(bottom_right_y_orig * scale_y)

            # Disegna il rettangolo della patch
            cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 0, 255), 1)

            # Numerazione della patch
            text_position = (top_left_x + 2, top_left_y + 12)
            cv2.putText(image, str(patch_num), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

            patch_num += 1

    cv2.imwrite(imagePath, image)
    return image

def draw_colored_patch_grid(image, patch_coordinates, predictions, class_colors, class_labels, imagePath=None, alpha=0.5):
    # Skip the first patch/prediction (cls token)
    patch_coordinates = patch_coordinates[1:]
    predictions = predictions[1:]

    overlay = image.copy()
    for (top_left, bottom_right), pred in zip(patch_coordinates, predictions):
        color = class_colors.get(pred, (0, 0, 0))
        color_bgr = (int(color[2]), int(color[1]), int(color[0]))
        cv2.rectangle(overlay, top_left, bottom_right, color_bgr, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Draw patch borders
    for (top_left, bottom_right) in patch_coordinates:
        cv2.rectangle(image, top_left, bottom_right, (255, 255, 255), 1)

    # Number the patches
    patch_num = 1
    for (top_left, _) in patch_coordinates:
        text_position = (top_left[0] + 2, top_left[1] + 12)
        cv2.putText(image, str(patch_num), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        patch_num += 1

    # Legend
    legend_height = 20
    legend_width = 150
    legend_margin = 10
    legend_image = np.ones((image.shape[0], legend_width, 3), dtype=np.uint8) * 255

    for i, (label, color) in enumerate(class_colors.items()):
        color_bgr = (int(color[2]), int(color[1]), int(color[0]))
        rect_top_left = (10, 10 + i * (legend_height + legend_margin))
        rect_bottom_right = (30, 30 + i * (legend_height + legend_margin))
        cv2.rectangle(legend_image, rect_top_left, rect_bottom_right, color_bgr, -1)
        cv2.rectangle(legend_image, rect_top_left, rect_bottom_right, (0, 0, 0), 1)
        cv2.putText(legend_image, class_labels[i], (35, 25 + i * (legend_height + legend_margin)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    combined_image = np.hstack((image, legend_image))

    if imagePath:
        cv2.imwrite(imagePath, combined_image)
    return combined_image
