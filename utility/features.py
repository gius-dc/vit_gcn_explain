import cv2
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from skimage.feature import graycomatrix, graycoprops


def removeBlackPixels(image):
    def isGray(img):
        # Ottieni il numero di canali dell'immagine
        channels = len(img.shape)
        
        # Verifica se l'immagine è in scala di grigi
        return channels == 2
    if not isGray(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mask = (image > 0).astype(np.uint8) * 255

    # Applica la maschera all'immagine originale
    img_grey_noBlack= cv2.bitwise_and(image, image, mask=mask)
    return img_grey_noBlack

def histogramGradientNormStd(image, hist_size):
    # Remove Black Pixels
    image = removeBlackPixels(image)

    hist = cv2.calcHist([image], [0], None, [hist_size], [0, 256])
    hist = hist.flatten()
    gradient = np.gradient(hist)
    #print(gradient)

    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_gradient = scaler.fit_transform(np.array(gradient).reshape(-1, 1)).flatten()

    # Standardizza il gradiente
    standard_scaler = StandardScaler()
    standardized_gradient = standard_scaler.fit_transform(np.array(normalized_gradient).reshape(-1, 1)).flatten()
    #print("Gradient",standardized_gradient)
    return standardized_gradient.tolist()

def histogramGradient(image, hist_size):
    hist = cv2.calcHist([image], [0], None, [hist_size], [0, 256])
    hist = hist.flatten()
    gradient = np.gradient(hist)

    return gradient.tolist()

def histogramIntensityNormStd(image, hist_size):
    # Remove Black Pixels
    image = removeBlackPixels(image)

    # Calcola l'istogramma dell'immagine
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist.flatten()

    # Calcola la media dei livelli di intensità
    intensity_mean = np.sum(hist * np.arange(256)) / np.sum(hist)

    # Calcola la deviazione standard dei livelli di intensità
    intensity_stddev = np.sqrt(np.sum(hist * (np.arange(256) - intensity_mean) ** 2) / np.sum(hist))

    # Trova il valore massimo e il valore minimo dei livelli di intensità
    intensity_max = np.max(np.arange(256)[hist > 0])
    intensity_min = np.min(np.arange(256)[hist > 0])

    # Trova la mediana dei livelli di intensità
    intensity_median = np.median(np.arange(256)[hist > 0])

    # Trova la moda dei livelli di intensità
    intensity_mode = np.argmax(hist)

    # Calcola alcuni percentili
    intensity_percentiles = np.percentile(np.arange(256)[hist > 0], [25, 75])

    arr = np.zeros(hist_size)
    arr[0] = intensity_min
    arr[1] = intensity_max
    arr[2] = intensity_mean
    arr[3] = intensity_stddev
    arr[4] = intensity_median
    arr[5] = intensity_mode
    arr[6:8] = intensity_percentiles
    # Aggiungi altre statistiche o valori legati all'intensità se necessario

    # Normalizza l'array
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_arr = scaler.fit_transform(arr.reshape(-1, 1)).flatten()

    # Standardizza l'array
    standard_scaler = StandardScaler()
    standardized_arr = standard_scaler.fit_transform(normalized_arr.reshape(-1, 1)).flatten()
    #print("Intensity",standardized_arr)
    return standardized_arr.tolist()

def histogramIntensity(image, hist_size):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist.flatten()

    # Calcola la media dei livelli di intensità
    intensity_mean = np.sum(hist * np.arange(256)) / np.sum(hist)

    # Calcola la deviazione standard dei livelli di intensità
    intensity_stddev = np.sqrt(np.sum(hist * (np.arange(256) - intensity_mean) ** 2) / np.sum(hist))

    # Trova il valore massimo e il valore minimo dei livelli di intensità
    intensity_max = np.max(np.arange(256)[hist > 0])
    intensity_min = np.min(np.arange(256)[hist > 0])

    arr = np.zeros(hist_size)
    arr[0] = intensity_min
    arr[1] = intensity_max
    arr[2] = intensity_mean
    arr[3] = intensity_stddev

    return arr.tolist()

def histogramColor(image, hist_size):

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Calcola l'istogramma dei colori (ad esempio, per il canale rosso)
    hist = cv2.calcHist([image_rgb], [0], None, [hist_size], [0, 256])
    hist = hist.flatten()

    return hist.tolist()

def histogramColorNormStd(image, hist_size):
    # Trova i pixel non neri
    non_black_mask = (image > 0).all(axis=2)

    # Converti l'immagine in RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Calcola l'istogramma dei colori solo per i pixel non neri
    hist = cv2.calcHist([image_rgb[non_black_mask]], [0], None, [hist_size], [0, 256])
    hist = hist.flatten()

    # Normalizza l'istogramma
    hist_normalized = hist / np.sum(hist)

    # Normalizza l'istogramma e standardizza i valori
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_hist = scaler.fit_transform(hist_normalized.reshape(-1, 1)).flatten()

    #print("Color", normalized_hist)

    return normalized_hist.tolist()

import time 

def histogramCSLTP(image, hist_size):

    def sigmoid(x):
        if x > 3:
            return 2
        elif x < -3:
            return 1
        else:
            return 0 
    
    
    image_height = image.shape[0]
    image_width = image.shape[1]

    img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    zeroHorizontal = np.zeros(image_width + 2).reshape(1, image_width + 2)
    zeroVertical = np.zeros(image_height).reshape(image_height, 1)

    img_grey = np.concatenate((img_grey, zeroVertical), axis = 1)
    img_grey = np.concatenate((zeroVertical, img_grey), axis = 1)
    img_grey = np.concatenate((zeroHorizontal, img_grey), axis = 0)
    img_grey = np.concatenate((img_grey, zeroHorizontal), axis = 0)

    pattern_img = np.zeros((image_height + 1, image_width + 1))
    
    
    for x in range(1, image_height -2):
        for y in range(1, image_width -2):
            
            s1 = sigmoid(img_grey[x-2, y-2] - img_grey[x+2, y+2])
            s3 = sigmoid(img_grey[x-2, y+2] - img_grey[x+2, y-2])*3
    
            s = s1 + s3
        
            pattern_img[x, y] = s
    start = time.time()
    pattern_img = pattern_img[1:(image_height+1), 1:(image_width+1)].astype(int)
    
    histogram = np.histogram(pattern_img, bins = np.arange(hist_size +1))[0]
    histogram = histogram.reshape(1, -1)
    
    #print("Time elapsed:", time.time() - start)

    return histogram[0].tolist()
    
def histogramGLCM(image, hist_size):
    #props: {‘contrast’, ‘dissimilarity’, ‘homogeneity’, ‘energy’, ‘correlation’, ‘ASM’}
    img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    glcm = graycomatrix(img_grey, distances=[5], angles=[0], levels=256,
                        symmetric=True, normed=True)
    
    arr = np.zeros(hist_size)
    arr[0] = graycoprops(glcm, 'contrast')[0, 0]
    arr[1] = graycoprops(glcm, 'dissimilarity')[0, 0]
    arr[2] = graycoprops(glcm, 'homogeneity')[0, 0]
    arr[3] = graycoprops(glcm, 'energy')[0, 0]
    arr[4] = graycoprops(glcm, 'correlation')[0, 0]
    arr[5] = graycoprops(glcm, 'ASM')[0, 0]
    
    #print(arr.tolist())
    return arr.tolist()

def histogramGLCMNormStd(image, hist_size):
    
    # Converti l'immagine in scala di grigi
    img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = removeBlackPixels(image)

    # Calcola la matrice GLCM solo per i pixel non neri
    glcm = graycomatrix(image, distances=[5], angles=[0], levels=256,
                        symmetric=True, normed=True)
    
    arr = np.zeros(hist_size)
    arr[0] = graycoprops(glcm, 'contrast')[0, 0]
    arr[1] = graycoprops(glcm, 'dissimilarity')[0, 0]
    arr[2] = graycoprops(glcm, 'homogeneity')[0, 0]
    arr[3] = graycoprops(glcm, 'energy')[0, 0]
    arr[4] = graycoprops(glcm, 'correlation')[0, 0]
    arr[5] = graycoprops(glcm, 'ASM')[0, 0]

    # Normalizza l'array
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_arr = scaler.fit_transform(arr.reshape(-1, 1)).flatten()

    # Standardizza l'array
    standard_scaler = StandardScaler()
    standardized_arr = standard_scaler.fit_transform(normalized_arr.reshape(-1, 1)).flatten()
    #print(standardized_arr)
    return standardized_arr.tolist()

def SIFTNormStd(image, num_keypoints=10):
    # Converti l'immagine in scala di grigi
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Crea un oggetto SIFT
    sift = cv2.SIFT_create()

    # Trova i keypoints e i descrittori con SIFT
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)

    # Seleziona i primi num_keypoints keypoints e i rispettivi descrittori
    selected_keypoints = keypoints[:num_keypoints]
    selected_descriptors = descriptors[:num_keypoints, :]

    # Unifica i descrittori in un array piatto
    sift_features = selected_descriptors.flatten()

    # Normalizza l'array
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_features = scaler.fit_transform(sift_features.reshape(-1, 1)).flatten()

    # Standardizza l'array
    standard_scaler = StandardScaler()
    standardized_features = standard_scaler.fit_transform(normalized_features.reshape(-1, 1)).flatten()
    print(len(standardized_features))
    return standardized_features.tolist()