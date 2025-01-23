from __future__ import division
from __future__ import print_function
import torch.nn.functional as F
import seaborn as sns
import torch.nn as nn

import time
import argparse
import numpy as np
import wandb
import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Data
from torchvision.ops import focal_loss

from utility import utility
from models import models
from torch_geometric.nn import GCN

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle
import os

'''
Build GCN from graph dataset saved in folders
V. 6.0

Support for WandDB
'''
WANDB = False

# Arguments to run the experiment
parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--verbose', action='store_true', default=False,
                    help='Show results.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.') #seed 42
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, #0.001
                    help='Initial learning rate.')
# parser.add_argument('--lr', type=float, default=0.05,
#                     help='Initial learning rate.')
parser.add_argument('--weight_decay', type=lambda x: float(x) if x.lower() != 'false' else False, default=False,
                    help='Weight decay (L2 loss on parameters). Can be a float value or False.')
parser.add_argument('--hidden_units', type=int, default=64, #128
                    help='Number of hidden units.')
# parser.add_argument('--hidden_units', type=int, default=16,
#                     help='Number of hidden units.')
parser.add_argument("--n_layers_set", nargs="+", type=int, default=[1], #default=[2, 3, 4, 5]
                    help='List of number of layers.')
parser.add_argument('--dropout', type=lambda x: float(x) if x.lower() != 'false' else False, default=False,
                    help='Dropout rate (1 - keep probability). Can be a float value or False.')
parser.add_argument('--patience', type=float, default=5,
                    help='Set patience rate')
parser.add_argument('--loss', type=float, default=0.4,
                    help='Set training loss rate')
parser.add_argument('--dataset', type=str, default="cub_GNN/",
                    help='Set the dataset main folder')
parser.add_argument('--output', type=str, default="output/",
                    help='Set the output experiments folder')
parser.add_argument('--model_name', type=str, default="CUB200",
                    help='Set the model name for experiments')
parser.add_argument('--n_classes', type=int, default=10, 
                    help='Number of classes')
parser.add_argument('--n_features', type=int, default=40, 
                    help='Number of features')
parser.add_argument('--batch', type=int, default=1, 
                    help='Batch size')
args = parser.parse_args()
print(args)
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    elem1 = batch[0]
    print(len(elem1[0]), len(elem1[1]))
    print(elem1[1])
    

args.cuda = not args.no_cuda and torch.cuda.is_available()

def save_args_to_txt(args, experiment_folder):
    # Salva i parametri in un file
    args_file_path = os.path.join(experiment_folder, 'params.txt')
    with open(args_file_path, 'w') as f:
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')

def drawCM(conf_matrix, path, n_classes):
    # Normalizza la matrice di confusione
    conf_matrix_percent = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
    class_labels = utility.litter_classes_background[:n_classes]

    # Crea il grafico
    plt.figure(figsize=(12, 12))
    sns.set(font_scale=1.2)
    sns.heatmap(conf_matrix_percent, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels,
                cbar_kws={'label': 'Percentuale'})
    plt.xlabel('Classe Predetta')
    plt.ylabel('Classe Reale')
    plt.title('Matrice di Confusione')
    plt.savefig(path, dpi=300)

def plot_metric(metric_history, title, xlabel, ylabel, save_path, epoch_interval=5):
    epochs = list(range(epoch_interval, len(metric_history) * epoch_interval + 1, epoch_interval))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metric_history, marker='o', label=title)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    max_labels = 10
    step = max(1, len(epochs) // max_labels)
    plt.xticks(epochs[::step])
    
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.close()

if WANDB:
    wandb.login()

    run = wandb.init(
        # set the wandb project where this run will be logged
        project="gcn-litter",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": args.lr,
        "architecture": "GCN",
        "dataset": "GLITTER2024",
        "epochs": args.epochs,
        "batch_size": args.hidden_units
        }
    )

#np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.cuda:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device("cpu")

num_classes = args.n_classes
num_features = args.n_features


class GraphDataset(Dataset):
    def __init__(self, folder_files, is_training_set=True):
        if is_training_set:
            self.root_dir = folder_files + "graphs_train/"
            self.edge_dir = folder_files + "adjacency_train/"
        else:
            self.root_dir = folder_files + "graphs_test/"
            self.edge_dir = folder_files + "adjacency_test/"

        self.list_graphs = sorted(os.listdir(self.root_dir))
        self.list_edges = sorted(os.listdir(self.edge_dir))

    def __len__(self):
        return len(self.list_graphs)
        

    def __getitem__(self,idx):
        graph_file = os.path.join(self.root_dir, self.list_graphs[idx])
        edge_file = os.path.join(self.edge_dir, self.list_edges[idx])

        with open(graph_file, 'rb') as f:
            features = pickle.load(f)

        loaded_data = np.load(edge_file)
        row, col = loaded_data['row'], loaded_data['col']
        edge_coordinates = np.array([row, col])
        edge_coordinates = torch.tensor(edge_coordinates, dtype=torch.long)

        return features, edge_coordinates

def save_model(epochs, model, optimizer, logits, criterion, model_name):
    """
    Function to save the trained model to disk.
    """
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'logits': logits,
                'loss': criterion,
                }, experiment_folder + '/' + str(epochs) + model_name +'.pth')  
    
dir_to_dataset = args.dataset
experiment_folder = utility.experimentFolder(args.output)

save_args_to_txt(args, experiment_folder)

for n_layers in args.n_layers_set:
    print("N.Layers",n_layers)
    loss_train_vec = np.zeros((args.epochs,), )
    loss_val_vec = np.zeros((args.epochs,), )
    acc_test_vec = np.zeros((args.epochs,), )
    
    ''' LOAD MODEL '''
    model = models.GCNClassifier(num_features, args.hidden_units, num_classes, dropout=args.dropout).to(device)
    
    #model = models.GCN_graph2(num_features, args.hidden_units, num_classes).to(device)
    #criterion = FocalLoss(gamma=2)    
    
    if args.weight_decay:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    # Dataset training
    graph_dataset_train = GraphDataset(folder_files=dir_to_dataset,
                                       is_training_set=True)
    train_dataloader = DataLoader(dataset=graph_dataset_train, batch_size=args.batch, shuffle=True, pin_memory=True, num_workers=os.cpu_count())
    # Dataset test
    graph_dataset_test = GraphDataset(folder_files=dir_to_dataset,
                                      is_training_set=False)
    test_dataloader = DataLoader(dataset=graph_dataset_test, batch_size=args.batch, shuffle=False, pin_memory=True, num_workers=os.cpu_count())
    
    epochs_without_improvement = 0
    early_stop = False
    best_validation_loss = float('inf')
    best_logits = []
    train_true_labels = []
    train_predicted_labels = []

    if WANDB:
        run.watch(model)

    # Liste per monitorare le metriche
    train_loss_history = []
    val_loss_history = []
    accuracy_history = []
    precision_history = []
    recall_history = []
    f1_history = []


    for epoch in range(0, args.epochs):
        all_true_labels = []
        all_predicted_labels = []
        model.train()
        start_time = time.time()
        loss_train_epoch = 0
        loss_val_epoch = 0
        acc_test_epoch = 0
        acc_epoch = 0
        val_id = 0
        #example of loop to access the features, classes and edges, this should be adpted to the order of the elements saved in the graph files
        # TRAINING SET
        #print("Train", len(train_dataloader))
        for i, (data_train, edges_train) in enumerate(train_dataloader):
            optimizer.zero_grad()
            
            # Prepara le feature, etichette e gli edge
            edges_train = edges_train.view(2, -1).long().to(device)
            features = data_train[1].view(data_train[1].size(1), -1).float().to(device)  # Dimensione (197, 768)
            # print(features.shape)
            # print(data_train[0].shape)
            # print(data_train[1].shape)
            y = data_train[2].squeeze().long().to(device)  # Classe target

            
            
            # Crea il grafo per PyTorch Geometric
            _data_train = Data(x=features, edge_index=edges_train, y=y)
            _data_train = _data_train.to(device)

            # Forward pass
            output = model(_data_train.x, _data_train.edge_index)  # Output (197, num_classes)

            # print(y.shape)
            # print(output.shape)
            # input("Enter to continue")


            # print(output)
            # print(output.shape)
            # input("Press ENTER to continue")
            
            # print(y)
            # print(y.shape)
            # input("Press ENTER to continue")
            
            # Loss (calcolata sulla classe vera usando CLS token)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            # Accumula la loss
            loss_train_epoch += loss.item()
            loss_train_vec[epoch] = loss_train_epoch / (i + 1)
        scheduler.step()
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            # VALIDATION SET
            for x, (data_test, edges_test) in enumerate(test_dataloader):
                # Prepara le feature e gli edge
                edges_test = edges_test.view(2, -1).long().to(device)
                features = data_test[1].view(data_test[1].size(1), -1).float().to(device)  # Dimensione (197, 768)
                y = data_test[2].squeeze().long().to(device)  # Classe target
                
                # Crea il grafo per PyTorch Geometric
                _data_test = Data(x=features, edge_index=edges_test, y=y)
                _data_test = _data_test.to(device)

                # Forward pass
                output = model(_data_test.x, _data_test.edge_index)
                
                # print(output)
                # print(output.shape)
                # input("Press ENTER to continue")
            
                # print(y)
                # print(y.shape)
                # input("Press ENTER to continue")
                
                # Loss di validazione (solo nodo CLS token)
                loss_val = criterion(output, y)
                loss_val_epoch += loss_val.item()
                loss_val_vec[epoch] = loss_val_epoch / (x + 1)
                
                # Predizione del modello (classe del CLS token)
                predicted_labels = output.argmax(dim=-1)
                #true_label = y.item()

                # Salva le predizioni e le etichette vere per le metriche
                all_predicted_labels.extend(predicted_labels.cpu())
                all_true_labels.extend(y.cpu())

        # EARLY STOPPING            
        # Check if loss is under the default loss
        validation_loss = loss_val_epoch / len(test_dataloader)
        if validation_loss <= args.loss:
            print("Under loss!")
            # Early Stopping: check if actual loss is better with this validation
            if float(utility.fourFloat_formatter(validation_loss)) < best_validation_loss:
                best_validation_loss = float(utility.fourFloat_formatter(validation_loss))
                epochs_without_improvement = 0  # Reset counter if improvement
            else:
                epochs_without_improvement += 1

        # PATIENCE
        print("epochs_without_improvement", epochs_without_improvement)
        if epochs_without_improvement >= args.patience:
            early_stop = True
            print(f'Early stopping triggered at epoch {epoch}.')
            best_model = save_model(n_layers, model, optimizer, output, criterion='cross_entropy_loss', model_name= args.model_name)

            break  # Stop if patience is over

        # LOGS
        #if (epoch + 1) % 5 == 0:
        if True:
            #print(loss_val_vec)
            model.eval()
            with torch.no_grad():
                
                # Example: Print accuracy
                #accuracy = accuracy_score(all_true_labels, all_predicted_labels)
                print(f'Epoch {epoch+1}/{args.epochs}, Loss Train: {loss_train_vec[epoch]}, Loss Val: {loss_val_vec[epoch]}')
                #print(all_predicted_labels, type(all_predicted_labels))
                true_lab = torch.tensor(all_true_labels)
                pred_lab = torch.tensor(all_predicted_labels)
                accuracy = accuracy_score(true_lab, pred_lab)
                precision = precision_score(true_lab, pred_lab, average='weighted')
                recall = recall_score(true_lab, pred_lab, average='weighted')
                f1 = f1_score(true_lab, pred_lab, average='weighted')
                print("Accuracy:", accuracy)
                print("Precision:", precision)
                print("Recall:", recall)
                print("F1 Score:", f1)

                # Registra metriche
                train_loss_history.append(loss_train_vec[epoch])
                val_loss_history.append(loss_val_vec[epoch])
                accuracy_history.append(accuracy)
                precision_history.append(precision)
                recall_history.append(recall)
                f1_history.append(f1)

                if WANDB:
                    run.log({"train_loss": loss_train_vec[epoch],
                            "val_loss": loss_val_vec[epoch],
                            "accuracy": accuracy,
                            "precision": precision,
                            "recall": recall,
                            "f1": f1
                            })

    
    if not early_stop:
        print(f"Saving Model: {args.model_name}")
        best_model = save_model(n_layers, model, optimizer, output, criterion='cross_entropy_loss', model_name=args.model_name)


    true_lab = torch.tensor(all_true_labels)
    pred_lab = torch.tensor(all_predicted_labels)
    accuracy = accuracy_score(true_lab, pred_lab)
    precision = precision_score(true_lab, pred_lab, average='weighted')
    recall = recall_score(true_lab, pred_lab, average='weighted')
    f1 = f1_score(true_lab, pred_lab, average='weighted')
    confusion_mat = confusion_matrix(true_lab, pred_lab)
    drawCM(confusion_mat, experiment_folder + '/matrix.png', num_classes)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    #print("Confusion Matrix:\n", confusion_mat)

    plot_metric(train_loss_history, "Train Loss", "Epoche", "Loss", os.path.join(experiment_folder, "train_loss.png"), epoch_interval=1)
    plot_metric(val_loss_history, "Validation Loss", "Epoche", "Loss", os.path.join(experiment_folder, "val_loss.png"), epoch_interval=1)
    plot_metric(accuracy_history, "Accuracy", "Epoche", "Accuracy", os.path.join(experiment_folder, "accuracy.png"), epoch_interval=1)
    plot_metric(precision_history, "Precision", "Epoche", "Precision", os.path.join(experiment_folder, "precision.png"), epoch_interval=1)
    plot_metric(recall_history, "Recall", "Epoche", "Recall", os.path.join(experiment_folder, "recall.png"), epoch_interval=1)
    plot_metric(f1_history, "F1 Score", "Epoche", "F1 Score", os.path.join(experiment_folder, "f1_score.png"), epoch_interval=1)

