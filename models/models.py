from torch_geometric.nn import GCNConv, Set2Set, Linear, global_mean_pool, LayerNorm
from torch_geometric.explain.algorithm import GNNExplainer
import torch_geometric.transforms as T

import torch
import torch.nn.functional as F
from torch.nn import Dropout
import torchvision.transforms.functional as TF
import torch.nn as nn
import numpy as np
from transformers import AutoFeatureExtractor, AutoModel, AutoImageProcessor
from numpy.typing import NDArray
from typing import List, Tuple, Dict, Union, Sequence
from PIL import Image

class ImageSimilarityModel(object):
    def __init__(self, ckp_path: str = "google/vit-base-patch16-224", # "google/vit-base-patch16-224"
                                                                      # "google/vit-base-patch16-224-in21k"
                                                                      # "facebook/deit-base-patch16-224"
                                                                      # "philschmid/vit-base-patch16-224-in21k-euroSat" (richiede from_tf=True)
                                                                      # "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10"
                                                                      # "edwinpalegre/vit-base-trashnet-demo" (produce grafi molto buoni, ma le predizioni sono tendenzialmente errate rispetto al modello di google)
                                                                      # "edwinpalegre/ee8225-group4-vit-trashnet-enhanced" (produce grafi molto buoni, ma le predizioni sono tendenzialmente errate rispetto al modello di google)
                 pad_size: int = 224,
                 top_k: int = 5, save_examples: bool = False, use_thr: bool = False, score_thr: float = 20.0,
                 score_perc_thr: float = 0.35, device: str = 'cpu', features_size = 512):
        self.device = device
        self.extractor = AutoFeatureExtractor.from_pretrained(ckp_path)
        self.model = AutoModel.from_pretrained(ckp_path, output_attentions=True, from_tf=False).to(self.device)
        self.top_k = top_k
        self.use_thr = use_thr
        self.score_thr = score_thr
        self.score_perc_thr = score_perc_thr
        self.save_examples = save_examples
        #self.linear_layer = nn.Linear(in_features=768, out_features=features_size).to(self.device)
        #self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.pad_size = pad_size

    def extract_embeddings_old(self, x: Union[NDArray, Image.Image]) -> NDArray:
        
        if isinstance(x, Image.Image):
            x = (x,)

        x = _pad_data(x, self.pad_size)
        
        image_pp = self.extractor(x, return_tensors="pt").to(self.device)
        features = self.model(**image_pp).last_hidden_state[:, 0].detach().cpu().numpy()
        return features.squeeze()
    
    def extract_embeddings(self, x: NDArray) -> NDArray:
        x = _pad_data([x], self.pad_size)
        image_pp = self.extractor(x, return_tensors="pt").to(self.device)
        features = self.model(**image_pp)

        return features.last_hidden_state[:, 0].squeeze().tolist()

    def extractEmbeddings(self, x: NDArray, mainToken: bool = True) -> List[List[float]]:
        x = _pad_data([x], self.pad_size)
        image_pp = self.extractor(x, return_tensors="pt").to(self.device)
        features = self.model(**image_pp)
        if mainToken:
            embeddings = features.last_hidden_state.squeeze().cpu()
        else:
            embeddings = features.last_hidden_state[:, 0].squeeze()
  
        patch_embeddings = embeddings.tolist()
        
        return patch_embeddings
        
    def extract_features_and_attentions(self, x: NDArray) -> NDArray:
        x = self.extractor(images=x, return_tensors="pt").to(self.device)
        outputs = self.model(**x, output_attentions=True)
        attentions = outputs.attentions
        f = outputs.last_hidden_state[:, 0]
        #print(attentions)
        #print("Attentions", attentions[0].shape)
        #print("Features", f.shape)

        return attentions
    

    def extract_attentions_with_weights(self, x: np.ndarray, indices_foreground: list, num_tot_patches=197, max_penalty=1.0, k=5) -> list:
        x = self.extractor(images=x, return_tensors="pt").to(self.device)
        outputs = self.model(**x, output_attentions=True)
        attentions = outputs.attentions
        
        # Numero di patch di primo piano e di sfondo
        num_foreground_patches = len(indices_foreground)
        num_background_patches = num_tot_patches - num_foreground_patches
        
        # Calcola la percentuale di foreground
        percentage_foreground = num_foreground_patches / num_tot_patches

        # Calcola il peso delle patch di sfondo
        if num_foreground_patches == 0:  # Se non ci sono patch di foreground
            background_weight = max_penalty
        elif num_foreground_patches == num_tot_patches:  # Se tutte le patch sono foreground
            background_weight = 0
        else:
            # Penalizzazione dinamica basata su percentage_foreground
            # Più è alta la percentuale di foreground, minore sarà la penalità per il background
            background_weight = (1 - percentage_foreground) ** k * max_penalty

        # Crea una lista di pesi per tutte le patch, inizializzando a background_weight
        patch_weights = torch.full((num_tot_patches,), background_weight)

        # Imposta il peso delle patch di foreground a 1.0
        patch_weights[indices_foreground] = 1.0

        # Debugging: stampa i dati calcolati
        # print(f"Numero di patch foreground: {num_foreground_patches}")
        # print(f"Numero di patch background: {num_background_patches}")
        # print(f"Percentuale foreground: {percentage_foreground * 100:.2f}%")
        # print(f"Peso assegnato alle patch di background: {background_weight:.4f}")

        # Modifica le matrici di attenzione
        adjusted_attentions = []
        for idx, attention_matrix in enumerate(attentions):
            adjusted_attention = attention_matrix.clone()

            # Modifica le matrici di attenzione in base ai pesi calcolati
            for i in range(num_tot_patches):
                adjusted_attention[0, :, i, :] *= patch_weights[i]  # Penalizza il peso come query
                adjusted_attention[0, :, :, i] *= patch_weights[i]  # Penalizza il peso come key

            # Debugging: verifica che la matrice sia modificata correttamente
            # print(f"Attenzione modificata al livello {idx}: shape {adjusted_attention.shape}")
            adjusted_attentions.append(adjusted_attention)

        return adjusted_attentions

def _pad_data(imgs: Sequence[NDArray], pad_size: int) -> NDArray:
    new_images = []
    for img in imgs:
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        w, h = img.size
        if h > pad_size or w > pad_size:
            img = np.array(TF.resize(img, size=pad_size - 1, max_size=pad_size))
        else:
            img = np.array(img)

        h, w, _ = img.shape

        if h < pad_size:
            size_to_pad = (pad_size - h)
            pad_hl = size_to_pad // 2
            pad_hr = size_to_pad - pad_hl
            img = np.pad(img, ((pad_hl, pad_hr), (0, 0), (0, 0)))

        if w < pad_size:
            size_to_pad = (pad_size - w)
            pad_wl = size_to_pad // 2
            pad_wr = size_to_pad - pad_wl
            img = np.pad(img, ((0, 0), (pad_wl, pad_wr), (0, 0)))

        new_images.append(img)

    return np.array(new_images)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

class GCNModelWithFocalLoss(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GCNModelWithFocalLoss, self).__init__()
        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, output_size)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
    
class GCNModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, output_size)

    def forward(self, x, edge_index):
        # Prima convoluzione GCN
        x = F.relu(self.conv1(x, edge_index))

        # Seconda convoluzione GCN
        x = self.conv2(x, edge_index)

        # Log-softmax per la classificazione multiclasse
        x = F.log_softmax(x, dim=1)
        return x
    
class GCN_noLayers(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim=16, output_dim=1):
        super().__init__()
        self.in_channels = in_channels

        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
class GCN_hidden(nn.Module):
    def __init__(self, in_channels, hidden_dim, output_dim, num_layers):
        super(GCN_hidden, self).__init__()
        self.in_channels = in_channels
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers - 1)])
        self.conv_final = GCNConv(hidden_dim, output_dim)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        
        x = self.conv_final(x, edge_index)
        return F.log_softmax(x, dim=1)


class GCN_graph2(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim=16, output_dim=25):
        super().__init__()
        self.in_channels = in_channels

        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.lin1 = Linear(hidden_dim, output_dim)
        self.batch_norm1 = LayerNorm(hidden_dim, mode='node')
        self.batch_norm2 = LayerNorm(hidden_dim, mode='node')
        self.batch_norm3 = LayerNorm(hidden_dim, mode='node')

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        #print(x.shape)
        x = F.relu(self.batch_norm1(self.conv1(x, edge_index)))
        #print(x.shape)

        #x = F.dropout(x, p=0.5)
        x = F.relu(self.batch_norm2(self.conv2(x, edge_index)))
        #print(x.shape)

        #x = F.dropout(x, p=0.5)
        x = F.relu(self.batch_norm3(self.conv3(x, edge_index)))
        #print(x.shape)

        x = torch.mean(x, dim=0)
        #print(x.shape)

        #x = F.dropout(x, p=0.5)
        x = self.lin1(x)
        #print(x.shape)
        #exit()
        return x
    
class GCN_graph(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim=16, output_dim=25):
        super().__init__()
        self.in_channels = in_channels

        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.lin1 = Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.conv3(x, edge_index))
        x = torch.mean(x, 0)
        x = F.dropout(x, p=0.5)
        x = self.lin1(x)

        return x
    

class GraphClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GraphClassifier, self).__init__()
        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, output_size)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        print("x",x.shape)
        return x
    

class GCN_graph3(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim=768, output_dim=7):
        super().__init__()
        self.in_channels = in_channels

        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.lin1 = Linear(hidden_dim, output_dim)
        self.batch_norm1 = LayerNorm(hidden_dim, mode='node')
        self.batch_norm2 = LayerNorm(hidden_dim, mode='node')
        self.batch_norm3 = LayerNorm(hidden_dim, mode='node')

    def forward(self, x, edge_index):
        #print(x.shape)
        x = F.relu(self.batch_norm1(self.conv1(x, edge_index)))
        #print(x.shape)
        x = F.relu(self.batch_norm2(self.conv2(x, edge_index)))
        #print(x.shape)
        x = F.relu(self.batch_norm3(self.conv3(x, edge_index)))
        #print("X before", x.shape) # (Size Node, Size Features)
        x = torch.mean(x, dim=0)
        #print("X after", x.shape) # (Size Features)
        x = self.lin1(x)
        #print(x.shape)
        return x
    
class GCN_graph_nodeClassification(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim=16, output_dim=25):
        super().__init__()
        self.in_channels = in_channels

        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        #self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.lin1 = Linear(hidden_dim, output_dim)
        self.batch_norm1 = LayerNorm(hidden_dim, mode='node')
        self.batch_norm2 = LayerNorm(hidden_dim, mode='node')
        #self.batch_norm3 = LayerNorm(hidden_dim, mode='node')

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.batch_norm1(self.conv1(x, edge_index)))
        #print(x.shape)
        x = F.relu(self.batch_norm2(self.conv2(x, edge_index)))
        #print(x.shape)
        #x = F.relu(self.batch_norm3(self.conv3(x, edge_index)))
        #print("X before", x.shape) # (Size Node, Size Features)
        
        #print("X after", x.shape) # (Size Features)
        x = self.lin1(x)

        x = F.log_softmax(x, dim=1)
        #print(x.shape)
        return x
    
class GCN_graph_nodeClassification2(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim=16, output_dim=25):
        super().__init__()
        self.in_channels = in_channels

        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        #self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.lin1 = Linear(hidden_dim, output_dim)
        self.batch_norm1 = LayerNorm(hidden_dim, mode='node')
        self.batch_norm2 = LayerNorm(hidden_dim, mode='node')
        #self.batch_norm3 = LayerNorm(hidden_dim, mode='node')

    def forward(self,  x, edge_index):
        #x, edge_index = data.x, data.edge_index
        x = F.relu(self.batch_norm1(self.conv1(x, edge_index)))
        #print(x.shape)
        x = F.relu(self.batch_norm2(self.conv2(x, edge_index)))
        #print(x.shape)
        #x = F.relu(self.batch_norm3(self.conv3(x, edge_index)))
        #print("X before", x.shape) # (Size Node, Size Features)
        
        #print("X after", x.shape) # (Size Features)
        x = self.lin1(x)

        x = F.log_softmax(x, dim=1)
        #print(x.shape)
        return x
    
class GCNClassifier(torch.nn.Module):
    def __init__(self, in_features, hidden_dim, num_classes, dropout=False):
        super(GCNClassifier, self).__init__()
        
        # Definizione dei layer della GCN
        self.gcn1 = GCNConv(in_features, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gcn3 = GCNConv(hidden_dim, hidden_dim)
        
        # Layer di classificazione finale per la classificazione di ogni nodo
        self.classifier = Linear(hidden_dim, num_classes)
        
        # Se `dropout` è un valore float, lo usiamo, altrimenti non usiamo dropout
        self.dropout = dropout if dropout else False

    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = Dropout(self.dropout)(x)  # Applica il dropout solo se `dropout` è un valore numerico

        x = self.gcn2(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = Dropout(self.dropout)(x)
        
        x = self.gcn3(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = Dropout(self.dropout)(x)
        
        # Output di classificazione per ogni nodo
        out = self.classifier(x)
        
        return out
    
    def predict_image_class(self, node_logits):
        """
        Funzione per ottenere la classe finale dell'immagine basata sul nodo CLS token.
        """
        cls_logit = node_logits[0]
        cls_pred = cls_logit.argmax(dim=-1)
        
        return cls_pred
    
    def predict_image_class_no_0(self, node_logits):
        """
        Funzione per ottenere la classe finale dell'immagine basata sul nodo CLS token.
        """
        cls_logit = node_logits[0]
        cls_logit[0] = float('-inf')
        cls_pred = cls_logit.argmax(dim=-1)
        
        return cls_pred