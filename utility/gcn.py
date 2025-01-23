import os
import pickle
import numpy as np
import graphviz
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import MinMaxScaler

from utility import utility


'''
	Graph construction, for tracker at CVPR Lab
	Code by Wieke Prummel 
'''

def flatten_features(frame_dict):
    features = []
    for roi_dict in frame_dict:
        features.append(roi_dict["feature_vec"])
    return np.array(features)

def flatten_labels(frame_dict):
    labels = []
    for roi_dict in frame_dict:
        labels.append(roi_dict["labels"])
    
    # Uncomment if graph classification
    #np.unique(labels)
    #labels = max(labels)
    #print(max_labels)
    
    return np.array(labels)

def flatten_rois(frame_dict):
    rois = []
    for roi_dict in frame_dict:
        rois.append(roi_dict["roi_number"])
    return np.array(rois)

def calculateGraph(rois, knn_param):
    """
    Calcola la matrice di adiacenza, indici e pesi del grafo k-NN.
    
    Args:
        rois: Lista di dizionari che contengono gli ROIs e i relativi vettori di caratteristiche.
        knn_param: Numero di vicini da considerare.
    
    Returns:
        features_matrix: Matrice delle caratteristiche.
        labels: Etichette corrispondenti ai ROIs.
        rois: ROIs originali appiattiti.
        A: Matrice di adiacenza sparsa (k-NN graph).
        row, col: Indici delle connessioni nella matrice di adiacenza.
    """
    features_matrix = flatten_features(rois)
    if len(features_matrix) < knn_param:
        raise ValueError("KNN parameter is too high for the number of ROIs.")
    
    labels = flatten_labels(rois)
    rois = flatten_rois(rois)
    
    if np.isnan(features_matrix).any():
        raise ValueError("Features matrix contains NaN values!")
    
    # Costruzione del grafo k-NN
    A = kneighbors_graph(features_matrix, knn_param, mode='distance', include_self=True)
    row, col = A.nonzero()
    
    return features_matrix, labels, rois, A, row, col

def exportGraph(rois, path_to_construction, isTrain, knn_param, img_number):
    """
    Esporta il grafo k-NN in formato .npz e .pkl per addestramento o test.
    """
    # Determina le cartelle per train/test
    subf = "train" if isTrain else "test"
    npz_path = os.path.join(path_to_construction, f"adjacency_{subf}/")
    pkl_path = os.path.join(path_to_construction, f"graphs_{subf}/")
    os.makedirs(npz_path, exist_ok=True)
    os.makedirs(pkl_path, exist_ok=True)
    
    try:
        # Calcolo del grafo
        features_matrix, labels, rois, _, row, col = calculateGraph(rois, knn_param)
        
        # Salva il grafo in formato .npz
        np.savez(os.path.join(npz_path, f'frame_{img_number}'), row=row, col=col)
        
        # Salva i dati in formato .pkl
        filename_graph = os.path.join(pkl_path, f'frame_{img_number}.pkl')
        with open(filename_graph, 'wb') as f:
            pickle.dump([rois, features_matrix, labels], f)
        print(f"Graph exported to {filename_graph}")
    
    except ValueError as e:
        print(f"Error while processing frame {img_number}: {e}")


def createGraph(rois, knn_param, output_folder='.', output_filename='knn_graph', create_image=True):
    """
    Crea un grafo k-NN e opzionalmente salva un'immagine del grafo.
    
    Args:
        rois: Lista di dizionari che contengono gli ROIs e i relativi vettori di caratteristiche.
        knn_param: Numero di vicini da considerare.
        output_folder: Cartella di salvataggio dell'immagine PNG del grafo.
        output_filename: Nome del file di output senza estensione.
        create_image: Booleano per determinare se creare o meno l'immagine del grafo.
    
    Returns:
        features: Una lista contenente i ROIs, la matrice delle caratteristiche e le etichette.
        row, col: Gli indici delle righe e delle colonne della matrice di adiacenza.
    """
    try:
        # Calcolo del grafo
        features_matrix, labels, rois, A, row, col = calculateGraph(rois, knn_param)
        
        if create_image:
            # Normalizza i pesi per la visualizzazione
            weights = np.asarray(A[row, col])  # Conversione a numpy array
            scaler = MinMaxScaler(feature_range=(0, 1))
            normalized_weights = scaler.fit_transform(weights.reshape(-1, 1)).flatten()
            
            # Crea il grafo usando Graphviz
            dot = graphviz.Digraph(format='png', engine='dot')
            dot.attr(dpi='300')  # Aumenta la risoluzione
            
            # Aggiungi i nodi
            for idx in range(len(rois)):
                dot.node(str(idx), label=str(rois[idx]))
            
            # Aggiungi gli archi, colorandoli in base al peso
            for r, c, weight in zip(row, col, normalized_weights):
                gray_value = int(255 * (1 - weight))  # Più alto il peso, più scuro il colore
                color_hex = f'#{gray_value:02x}{gray_value:02x}{gray_value:02x}'
                dot.edge(str(r), str(c), label=f"{weight:.2f}", color=color_hex)
            
            # Salva l'immagine del grafo
            output_path = os.path.join(output_folder, output_filename)
            dot.render(output_path, cleanup=True)
            print(f"Graph image saved to {output_path}.png")
        
        return [rois, features_matrix, labels], row, col
    
    except ValueError as e:
        print(f"Error while creating graph: {e}")
        return None, None, None
