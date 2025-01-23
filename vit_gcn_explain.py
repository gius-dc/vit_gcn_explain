import os
import torch
import argparse
import numpy as np
import shutil
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from transformers import ViTModel, ViTFeatureExtractor
from models.models import GCN_graph3, GCN_graph_nodeClassification, GCNClassifier
from models.models import ImageSimilarityModel
from utility import utility, gcn
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer, ThresholdConfig
import networkx as nx
from torch_geometric.utils import to_networkx
from graphviz import Digraph
from torch_geometric.explain import unfaithfulness
from torchviz import make_dot
import imageio
import ffmpeg
import colorsys
import random

# Parsing degli argomenti da linea di comando
parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disable CUDA usage.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--classes', type=int, default=10, help='Number of classes.')
parser.add_argument('--ft', type=int, default=768, help='Number of features.')
parser.add_argument('--features', type=str, default='vit', help='Name of the features.')
parser.add_argument('--clusters', type=int, default=10, help='Number of clusters for segmentation.')
parser.add_argument('--histSize', type=int, default=10, help='Histogram size.')
parser.add_argument('--fullImage', action='store_true', default=False, help='Use full image or bounding boxes.')
parser.add_argument('--id_range_start', type=int, default=None, help='Start of the id range to process.')
parser.add_argument('--id_range_end', type=int, default=None, help='End of the id range to process.')
parser.add_argument('--id', type=int, default=None, help='Single id to process.')
parser.add_argument('--input_path', type=str, default='../datasets/GLitter/val', help='Path to the input dataset.')

args = parser.parse_args()
print(args)

# Configura CUDA e semi per riproducibilità
args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

device = torch.device('cuda' if args.cuda else 'cpu')
if args.features == "vit":
    modelTransformer = ImageSimilarityModel(device=device)
    
# Carica il modello ViT pre-addestrato e il processore
model_name = "google/vit-base-patch16-224"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
vit_model = ViTModel.from_pretrained(model_name)
vit_model.eval()  # Imposta il modello in modalità valutazione

def extractEmbeddings(image, mainToken=False):
    """
    Estrae gli embeddings da un'immagine usando un modello ViT pre-addestrato.
    """
    # Prepara l'immagine
    inputs = feature_extractor(images=image, return_tensors="pt")
    
    # Esegui l'inferenza
    with torch.no_grad():
        outputs = vit_model(**inputs, output_attentions=True)
        attentions = outputs.attentions
    
    # Estrai gli embeddings
    if mainToken:
        embeddings = outputs.last_hidden_state[:, 0, :].squeeze(0).numpy()
        return [embeddings.tolist()]  # Embedding del token [CLS]
    
    embeddings = outputs.last_hidden_state.squeeze(0).numpy()
    return embeddings.tolist(), attentions

def prepare_output_folder():
    output_folder = "output"
    
    # Rimuovi la cartella se esiste
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    
    # Crea la cartella
    os.makedirs(output_folder)
    
def create_output_folders(id_x):
    """
    Crea la struttura delle cartelle per ogni id_x all'interno della cartella 'output'.
    """
    output_folder = os.path.join("output", str(id_x))
    
    # Crea la cartella principale per id_x
    os.makedirs(output_folder, exist_ok=True)
    
    # Crea le sottocartelle
    attentions_with_grid_folder = os.path.join(output_folder, "attentions_with_grid")
    attentions_without_grid_folder = os.path.join(output_folder, "attentions_without_grid")
    explainer_graphs_folder = os.path.join(output_folder, "explainer_graphs")
    
    os.makedirs(attentions_with_grid_folder, exist_ok=True)
    os.makedirs(attentions_without_grid_folder, exist_ok=True)
    os.makedirs(explainer_graphs_folder, exist_ok=True)
    
    return output_folder, attentions_with_grid_folder, attentions_without_grid_folder, explainer_graphs_folder # Restituisci tutte le cartelle
    
def save_cropped_frame(frame, id_x, obj_i, full_image=True):
    cv2.imwrite(f"vit_output/input_{id_x}.png", frame)
    b_box = obj_i['b_box']
    x, y, w, h = map(int, b_box)
    return frame if full_image else frame[y:y + h, x:x + w]

def getInformations(file_path):
    """
    Legge un file e estrae informazioni su ciascun oggetto descritto in ogni riga.
    Ogni riga del file dovrebbe contenere:
    - Un ID di classe (primo valore), che identifica il tipo di oggetto.
    - Quattro coordinate (x_min, y_min, x_max, y_max) che definiscono i bordi del riquadro (bounding box) dell'oggetto.
    """
    
    objects = []  # Lista per memorizzare le informazioni sugli oggetti trovati
    
    try:
        # Apertura del file per la lettura delle informazioni
        with open(file_path, 'r') as file:
            for line in file:
                # Rimuove spazi vuoti e divide la riga in valori
                values = line.strip().split()
                
                # Controlla che ci siano almeno 5 valori nella riga
                if len(values) >= 5:
                    # Il primo valore è l'ID della classe, che rappresenta il tipo di oggetto
                    class_id = values[0]
                    
                    # I successivi quattro valori rappresentano le coordinate del riquadro
                    # x_min e y_min definiscono il vertice in alto a sinistra del riquadro
                    # x_max e y_max definiscono il vertice in basso a destra del riquadro
                    x_min, y_min, x_max, y_max = map(float, values[1:5])
                    
                    # Aggiunge il dizionario con l'ID della classe e il riquadro alla lista
                    objects.append({'id': class_id, 'b_box': [x_min, y_min, x_max, y_max]})
                else:
                    # Stampa un messaggio se la riga non contiene almeno 5 valori
                    print(f'Riga non valida: {line}')
    
    # Gestisce il caso in cui il file specificato non esiste
    except FileNotFoundError:
        print(f'File non trovato: {file_path}')
    
    return objects

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def analyze_single_embedding(attentions, embedding_index, discard_ratio, head_fusion, device='cpu'):
    result = torch.eye(attentions[0].size(-1)).to(device)
    counter = 0  # Introduci un indice contatore prima del ciclo
    with torch.no_grad():
        for attention in attentions:
            counter += 1  # Incrementa il contatore ad ogni iterazione
            attention = attention.to(device)
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"
            B = torch.zeros_like(attention_heads_fused)
            B[:, 0, embedding_index] = 1
            
            attention_to_embedding = attention_heads_fused * B
            I = torch.eye(attention_to_embedding.size(-1)).to(device)
            a = (attention_to_embedding + I) / 2
            a = a / a.sum(dim=-1, keepdim=True)

            flat = a.view(a.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * (1 - discard_ratio)), -1, False)
            flat[:, indices] = 0
            a = flat.view_as(a)

            result = torch.matmul(result, a)

    mask = result[0, 0, 1:]
    width = int(np.sqrt(mask.size(0)))  
    mask = mask.reshape(width, width).cpu().numpy()
    max_val = np.max(mask)
    if max_val != 0:
        mask = mask / max_val

    #print(f"Numero di iterazioni: {counter}")  # Stampa il valore del contatore dopo il ciclo
    #input("Press Enter to continue...")  # Aggiungi una pausa per visualizzare il messaggio
    return mask

def create_heatmap_indices(image, attentions, indices, discard_ratio=0.9, device='cpu', draw_grid=False):
    mask = analyze_single_embedding(attentions, indices, discard_ratio, "max", device)
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    heatmap = show_mask_on_image(np.array(image)[:, :, ::-1], mask)
    
    if draw_grid:
        patch_coordinates = utility.calculatePatchDimensions(image)
        heatmap = utility.drawPatchGrid(heatmap, patch_coordinates)
    
    return heatmap

def drawHistClasses(outputs, output_folder, image_name="histogram_predictions_classes.png", litter_classes_background=None):
    """
    Disegna e salva un istogramma delle predizioni per classe.
    """
    predictions_per_class = []

    # Raccoglie le predizioni per classe
    for i, node_output in enumerate(outputs):
        predicted_class_node = torch.argmax(node_output).item()
        predictions_per_class.append(predicted_class_node)
        print(f"Nodo {i}: Predizione - Classe {predicted_class_node}")
        print(f"Vettore delle predizioni: {node_output.tolist()}")

    # Crea l'istogramma
    plt.figure(figsize=(10, 6))
    counts, bins, patches = plt.hist(predictions_per_class, bins=np.arange(len(litter_classes_background) + 1) - 0.5, edgecolor='black', align='mid')
    plt.title("Distribuzione delle Predizioni per Classe")
    plt.xlabel("Classe")
    plt.ylabel("Numero di Predizioni")
    plt.xticks(range(len(litter_classes_background)), litter_classes_background, rotation=45, ha='right')
    plt.grid(True)

    # Aggiungi il numero di predizioni sopra ogni barra
    for count, patch in zip(counts, patches):
        plt.text(patch.get_x() + patch.get_width() / 2, patch.get_height(), int(count), ha='center', va='bottom')

    # Crea la cartella di output se non esiste
    os.makedirs(output_folder, exist_ok=True)

    # Salva l'immagine
    output_image_path = os.path.join(output_folder, image_name)
    plt.savefig(output_image_path)
    print(f"Immagine salvata in: {output_image_path}")
    plt.close()

def drawHistClassesNoZero(outputs, output_folder, image_name="histogram_predictions_classes.png", litter_classes=None):
    """
    Disegna e salva un istogramma delle predizioni per classe, ignorando la prima classe.
    """
    predictions_per_class = []

    # Raccoglie le predizioni per classe
    for i, node_output in enumerate(outputs):
        node_output_copy = node_output.clone()
        node_output_copy[0] = float('-inf')  # Ignora la prima classe
        predicted_class_node = torch.argmax(node_output_copy).item()
        predictions_per_class.append(predicted_class_node - 1)  # Allinea l'indice
        print(f"Nodo {i}: Predizione - Classe {predicted_class_node - 1}")
        print(f"Vettore delle predizioni (prima classe ignorata): {node_output_copy.tolist()}")

    # Crea l'istogramma
    plt.figure(figsize=(10, 6))
    counts, bins, patches = plt.hist(predictions_per_class, bins=np.arange(len(litter_classes) + 1) - 0.5, edgecolor='black', align='mid')
    plt.title("Distribuzione delle Predizioni per Classe")
    plt.xlabel("Classe")
    plt.ylabel("Numero di Predizioni")
    plt.xticks(range(len(litter_classes)), litter_classes, rotation=45, ha='right')
    plt.grid(True)

    # Aggiungi il numero di predizioni sopra ogni barra
    for count, patch in zip(counts, patches):
        plt.text(patch.get_x() + patch.get_width() / 2, patch.get_height(), int(count), ha='center', va='bottom')

    # Crea la cartella di output se non esiste
    os.makedirs(output_folder, exist_ok=True)

    # Salva l'immagine
    output_image_path = os.path.join(output_folder, image_name)
    plt.savefig(output_image_path)
    print(f"Immagine salvata in: {output_image_path}")
    plt.close()


def drawConnectionHistToZero(all_node_indices, output_folder, image_name="histogram_connection_to_zero.png"):
    """
    Disegna un istogramma con due barre: una per i nodi relazionati al nodo 0 e l'altra per gli altri nodi.
    """
    connected_to_node_0 = 0  # Nodi connessi al nodo 0
    not_connected_to_node_0 = 0  # Nodi non connessi al nodo 0

    # Conta i nodi relazionati e non relazionati al nodo 0
    for node_indices in all_node_indices:
        if 0 in node_indices:
            connected_to_node_0 += 1
        else:
            not_connected_to_node_0 += 1

    total_nodes = connected_to_node_0 + not_connected_to_node_0
    connected_percentage = (connected_to_node_0 / total_nodes) * 100
    not_connected_percentage = (not_connected_to_node_0 / total_nodes) * 100

    # Crea l'istogramma
    plt.figure(figsize=(8, 5))
    categories = ['Relazionati al nodo 0', 'Non relazionati al nodo 0']
    counts = [connected_to_node_0, not_connected_to_node_0]
    
    plt.bar(categories, counts, color=['blue', 'red'], edgecolor='black')
    plt.title("Distribuzione dei Nodi Relazionati al Nodo 0")
    plt.xlabel("Categoria di Nodi")
    plt.ylabel("Numero di Nodi")
    plt.grid(True, axis='y')

    # Aggiungi i numeri e le percentuali sopra le barre
    for i, count in enumerate(counts):
        percentage = connected_percentage if i == 0 else not_connected_percentage
        plt.text(i, count + 0.1, f"{count} ({percentage:.2f}%)", ha='center', va='bottom', fontsize=12)

    # Crea la cartella di output se non esiste
    os.makedirs(output_folder, exist_ok=True)

    # Salva l'immagine
    output_image_path = os.path.join(output_folder, image_name)
    plt.savefig(output_image_path)
    plt.close()

def drawHistNodeConnections(sorted_indices, all_node_indices, outputs, predicted_class_global, output_folder, image_name="node_connection_histogram.png"):
    """
    Disegna un istogramma che mostra, per ciascun nodo classificato come predicted_class_global,
    quanti nodi collegati NON sono stati classificati come predicted_class_global.
    """
    sorted_indices_list = sorted_indices.tolist()
    non_predicted_class_count = []

    for idx, node_index in enumerate(sorted_indices_list):
        connected_nodes = all_node_indices[idx]
        count_non_predicted = sum(1 for connected_node in connected_nodes if connected_node not in sorted_indices_list)
        non_predicted_class_count.append(count_non_predicted)
        print(f"Node {node_index}: {len(connected_nodes)} connected nodes. {count_non_predicted} not classified as {predicted_class_global}.")

    # Crea un istogramma
    plt.figure(figsize=(len(sorted_indices_list) * 0.3, 8), dpi=300)
    plt.bar(sorted_indices_list, non_predicted_class_count, color='skyblue', edgecolor='black')

    # Aggiungi titolo e etichette
    plt.title(f"Distribuzione dei Nodi Non Classificati come Classe {predicted_class_global}", fontsize=16)
    plt.xlabel("Nodi classificati come Predicted Class", fontsize=14)
    plt.ylabel(f"Nodi collegati NON classificati come Classe {predicted_class_global}", fontsize=14)
    plt.xticks(sorted_indices_list, labels=[f"Node {i}" for i in sorted_indices_list], rotation=90, fontsize=12)

    # Mostra valori sopra le barre
    for idx, count in zip(sorted_indices_list, non_predicted_class_count):
        plt.text(idx, count, str(count), ha='center', va='bottom', fontsize=10)

    # Crea la cartella di output se non esiste
    os.makedirs(output_folder, exist_ok=True)

    # Salva il grafico
    output_image_path = os.path.join(output_folder, image_name)
    plt.tight_layout()
    plt.savefig(output_image_path)
    print(f"Grafico salvato in: {output_image_path}")

    plt.close()

def save_video(frames, output_path):
    frames_float = [frame.astype(np.float32) / 255.0 for frame in frames]
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', 
               s='{}x{}'.format(frames[0].shape[1], frames[0].shape[0]))
        .output(output_path, vcodec='libx264', crf=18, preset='slow')
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    for frame in frames_float:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    process.wait()

def save_video_with_opencv(frames, output_path, fps=10):
    if not frames:
        return
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.shape[2] == 3 else frame
        out.write(frame_bgr)
    out.release()

def generate_colors(num_classes, seed=456):
    random.seed(seed)
    colors = {}
    for i in range(num_classes):
        hue = random.random()
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors[i] = (int(r*255), int(g*255), int(b*255))
    return colors

def main():
    prepare_output_folder()  # Prepara la struttura della cartella "output", cancellandola se esiste già

    files = [f for f in os.listdir(args.input_path) if os.path.isfile(os.path.join(args.input_path, f)) and f.lower().endswith('.jpg')]
    files_sorted = sorted(files)
    id_x = 0
    
    for file in tqdm(files_sorted, desc='Extracting ROIs', unit='img'):
        path = os.path.join(args.input_path, file)
        info_file = path.replace('.jpg', '.txt')
        objects = getInformations(info_file)

        rois_data = []  # Lista di ROIs e delle loro informazioni

        for obj_i in objects:
            if int(obj_i['id']) <= args.classes:
                id_x += 1
                
                # Controlla se stiamo elaborando un singolo id o un range
                if args.id is not None:  # Se è specificato un singolo id
                    if id_x != args.id:
                        continue  # Salta se id_x non corrisponde all'id specificato
                else:  # Se non è specificato un singolo id
                    if (args.id_range_start is not None and id_x < args.id_range_start) or \
                       (args.id_range_end is not None and id_x > args.id_range_end):
                        continue  # Salta se id_x non è nel range
                
                output_folder, attentions_with_grid_folder, attentions_without_grid_folder, explainer_graphs_folder = create_output_folders(id_x)  # Crea la struttura delle cartelle per id_x
                frame = cv2.imread(path)
                frame = utility.resizeImage(frame, 800, 600)

                # Salva l'immagine input
                cv2.imwrite(os.path.join(output_folder, f"{id_x}_input.png"), frame)


                # Sezione di estrazione delle ROI
                cropped_region = save_cropped_frame(frame, id_x, obj_i, args.fullImage)
        
                # Salvataggio delle ROI
                roi_filename = os.path.join(output_folder, f"{id_x}_rois.png")
                img_rois = utility.drawPatchGrid(cropped_region.copy(), utility.calculatePatchDimensions(cropped_region), roi_filename)
                cv2.imwrite(f"{id_x}_rois.png", img_rois)

                # Estrai gli embeddings per tutte le patch
                extracted_features = modelTransformer.extractEmbeddings(cropped_region, mainToken=True)
        
                # Costruisci la struttura di rois per ogni embedding di patch
                rois = [{"roi_number": i, "labels": int(obj_i["id"]), "feature_vec": f} for i, f in enumerate(extracted_features)]
        
                # Crea il grafofrom torch_geometric.nn import GNNExplainer
                graph_features, row, col = gcn.createGraph(rois, knn_param=3, output_folder=output_folder, create_image=True)
                
                # Preparazione dati da dare all'explainer
                features_matrix = torch.tensor(graph_features[1], dtype=torch.float)

                # Converte row e col in tensori e crea l'edge_index
                edge_index = torch.tensor([row, col], dtype=torch.long)

                # Creazione del grafo (per l'explainer)
                data = Data(x=features_matrix, edge_index=edge_index)
                
                # GNNExplainer
                model = GCNClassifier(args.ft, args.hidden, args.classes).to(device)
                load_model = torch.load('modelTrainGCN_tesi/exp01/5CUB200.pth')
                model.load_state_dict(load_model['model_state_dict'])
                
                data = data.to(device)
                outputs = model(data.x, data.edge_index)
                make_dot(outputs, params=dict(model.named_parameters())).render("model_graph", format="png")
                print(model.predict_image_class(outputs))
                # mean_class_scores = torch.mean(outputs, dim=0)
                # Otteniamo la predizione globale
                # Trova la seconda classe con il punteggio medio più alto
                # highest_class_scores = mean_class_scores.clone()
                # highest_class_scores[0] = float('-inf')  # Ignora l'indice 0
                predicted_class_global = model.predict_image_class(outputs)
                
                if predicted_class_global != 0:
                    predicted_class = predicted_class_global
                else:
                    predicted_class = model.predict_image_class_no_0(outputs)
                
                local_predictions = torch.argmax(outputs, dim=1)
                matching_indices = torch.where(local_predictions == predicted_class)[0]
                matching_scores = outputs[matching_indices, predicted_class]
                sorted_indices = matching_indices[torch.argsort(matching_scores, descending=True)]
                
                
                # Salva la predizione globale in un file txt
                prediction_file_path = os.path.join(output_folder, "global_prediction.txt")
                with open(prediction_file_path, 'w') as f:
                    if predicted_class_global != 0:
                        f.write(f"Predizione globale (classe basata sul nodo CLS token): {predicted_class_global}\n")
                        f.write(f"Classe: {utility.litter_classes_background[predicted_class_global]}\n")
                    else:
                        f.write(f"Predizione globale (classe basata sul nodo CLS token): 0\n")
                        f.write(f"Classe: {utility.litter_classes_background[0]}\n")
                        f.write(f"Predizione escludendo la classe 0: {predicted_class}\n")
                        f.write(f"Classe: {utility.litter_classes_background[predicted_class]}\n")
                predicted_class_global = predicted_class
                
                #print(outputs)
                #print(outputs.shape)
                # print(outputs[0])
                # print("Indici dei nodi con le predizioni più forti per la classe globale:", sorted_indices.tolist())
                # Chiamata alla funzione per disegnare e salvare l'istogramma
                drawHistClasses(outputs, output_folder, "histogram_predictions_classes.png", utility.litter_classes_background)
                drawHistClassesNoZero(outputs, output_folder, "histogram_predictions_classes_no_0.png", utility.litter_classes)

                #print(sorted_indices.tolist())


                attentions = modelTransformer.extract_features_and_attentions(cropped_region)
                #print(len(attentions))
                #print(attentions[0].shape)
                #input("Press Enter to continue...")

                # Genera attention rollout per tutti gli indici in sorted_indices
                heatmap_with_grid = create_heatmap_indices(cropped_region, attentions, sorted_indices.tolist(), draw_grid=True, device=device)
                cv2.imwrite(os.path.join(attentions_with_grid_folder, f"attention_rollout_whole_gcn_{id_x}.png"), heatmap_with_grid)
                
                heatmap_without_grid = create_heatmap_indices(cropped_region, attentions, sorted_indices.tolist(), draw_grid=False, device=device)
                cv2.imwrite(os.path.join(attentions_without_grid_folder, f"attention_rollout_whole_gcn_{id_x}.png"), heatmap_without_grid)
                
                # Genera attention rollout per tutti gli indici da 0 a 197
                all_indices = list(range(197))

                heatmap_all_with_grid = create_heatmap_indices(cropped_region, attentions, all_indices, draw_grid=True, device=device)
                cv2.imwrite(os.path.join(attentions_with_grid_folder, f"attention_rollout_whole_vit_{id_x}.png"), heatmap_all_with_grid)

                heatmap_all_without_grid = create_heatmap_indices(cropped_region, attentions, all_indices, draw_grid=False, device=device)
                cv2.imwrite(os.path.join(attentions_without_grid_folder, f"attention_rollout_whole_vit_{id_x}.png"), heatmap_all_without_grid)

                # Liste per i frame
                frames_with_grid = []
                frames_without_grid = []

                # Ciclo da k = 0 a k = 10
                for k in range(0, 11):
                    # Calcola le attenzioni con il nuovo valore di k
                    attentions_weighted = modelTransformer.extract_attentions_with_weights(cropped_region, sorted_indices, k=k)
                    
                    # Crea la heatmap con la griglia
                    heatmap_all_with_grid = create_heatmap_indices(cropped_region, attentions_weighted, all_indices, draw_grid=True, device=device)
                    cv2.imwrite(os.path.join(attentions_with_grid_folder, f"attention_rollout_whole_vit_weighted_k{k}.png"), heatmap_all_with_grid)
                    
                    # Crea la heatmap senza la griglia
                    heatmap_all_without_grid = create_heatmap_indices(cropped_region, attentions_weighted, all_indices, draw_grid=False, device=device)
                    cv2.imwrite(os.path.join(attentions_without_grid_folder, f"attention_rollout_whole_vit_weighted_k{k}.png"), heatmap_all_without_grid)
                    
                    # Converti da RGB a BGR a RGB per risolvere l'inversione dei canali
                    heatmap_with_grid_rgb = cv2.cvtColor(heatmap_all_with_grid, cv2.COLOR_BGR2RGB)
                    heatmap_without_grid_rgb = cv2.cvtColor(heatmap_all_without_grid, cv2.COLOR_BGR2RGB)
                    
                    # Aggiungi i frame 
                    frames_with_grid.append(heatmap_with_grid_rgb)
                    frames_without_grid.append(heatmap_without_grid_rgb)

                # Salva GIF
                imageio.mimsave(
                    os.path.join(attentions_with_grid_folder, "attention_rollout_with_grid.gif"), 
                    frames_with_grid, 
                    duration=1,
                    loop=0,
                    quality=10
                )
                imageio.mimsave(
                    os.path.join(attentions_without_grid_folder, "attention_rollout_without_grid.gif"), 
                    frames_without_grid, 
                    duration=1,
                    loop=0,
                    quality=10
                )
                # Salva i video
                save_video_with_opencv(frames_with_grid, os.path.join(attentions_with_grid_folder, "video_with_grid.mp4"))
                save_video_with_opencv(frames_without_grid, os.path.join(attentions_without_grid_folder, "video_without_grid.mp4"))
                
                coeffs = {
                    'edge_size': 0.8,
                    'node_feat_size': 0.7,
                    'edge_reduction': 'sum',
                    'node_feat_reduction': 'mean',
                    'edge_ent': 0.3,
                    'node_feat_ent': 0.3,
                    'EPS': 1e-10,
                }

                threshold_config = ThresholdConfig(
                    threshold_type="topk",
                    value=10
                )

                model_config = dict(
                    mode='multiclass_classification',  # La rete classifica su più classi
                    task_level='node',                  # Predizione a livello di nodo
                    return_type='probs'
                )

                explainer = Explainer(
                    model=model,
                    algorithm=GNNExplainer(epochs=200, coeffs=coeffs),
                    explanation_type='model',
                    node_mask_type='attributes',
                    edge_mask_type='object',
                    model_config=model_config,
                    #threshold_config=threshold_config
                )
            
                all_node_indices = []

                for index in sorted_indices.tolist():
                    # Ottenere la spiegazione per ogni indice
                    explanation = explainer(data.x, data.edge_index, index=index)
                    explanation.visualize_graph(os.path.join(explainer_graphs_folder, f"graph_{index}.png"), backend="graphviz")
                    node_mask = explanation.get('node_mask')
                    node_indices = torch.nonzero(node_mask).squeeze().tolist()
                    node_indices = list(set(index[0] if isinstance(index, list) else index for index in node_indices))
                    print(f"Node indices for index {index}:", node_indices)
                    all_node_indices.append(node_indices)  # Aggiungi node_indices alla lista

                    explanation.visualize_feature_importance(
                        path=os.path.join(explainer_graphs_folder, f"top_5_{index}.png"),  # Salva il plot
                        top_k=5  # Seleziona solo le top 5 feature
                    )

                drawConnectionHistToZero(all_node_indices=all_node_indices, output_folder=output_folder, image_name="histogram_connection_to_zero.png")
                drawHistNodeConnections(sorted_indices, all_node_indices, outputs, predicted_class_global, output_folder, image_name="histogram_node_connections_no_predicted_class.png")

                num_classes = len(utility.litter_classes_background)
                # class_colors = generate_colors(num_classes)

                class_colors = {
                    0: (255, 255, 255),  # bianco
                    1: (255, 0, 0),      # rosso
                    2: (0, 255, 0),      # verde
                    3: (0, 0, 255),      # blu
                    4: (255, 255, 0),    # giallo
                    5: (255, 165, 0),    # arancione
                    6: (128, 0, 128),    # viola
                    7: (0, 0, 0)         # nero
                }

                class_labels = {i: utility.litter_classes_background[i] for i in range(num_classes)}  # Mappa classi a etichette

                # Salva l'immagine con la griglia colorata
                img_colored_rois = utility.draw_colored_patch_grid(
                    cropped_region.copy(),
                    utility.calculatePatchDimensions(cropped_region),
                    local_predictions.tolist(),
                    class_colors,
                    class_labels,
                    os.path.join(output_folder, f"{id_x}_colored_rois.png"),
                    alpha=0.5
                )
                cv2.imwrite(f"{id_x}_colored_rois.png", img_colored_rois)

                for node_indices in all_node_indices:
                    heatmap_with_grid = create_heatmap_indices(cropped_region, attentions, node_indices, draw_grid=True, device=device)
                    cv2.imwrite(os.path.join(attentions_with_grid_folder, f"attention_rollout_{id_x}_{node_indices}.png"), heatmap_with_grid)
                    
                    heatmap_without_grid = create_heatmap_indices(cropped_region, attentions, node_indices, draw_grid=False, device=device)
                    cv2.imwrite(os.path.join(attentions_without_grid_folder, f"attention_rollout_{id_x}_{node_indices}.png"), heatmap_without_grid)
                
                
if __name__ == "__main__":
    main()