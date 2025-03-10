import torch
import torch.nn.functional as F
from models.GAT import GAT
from utils.data_processing import loadAndProcesData, constructGraph, convertToPygFormat
from config import Config

def train():
    #Load and preprocess data
    features, labels, raw_data = loadAndProcesData(Config.CSV_FILE)
    print("Data has been processed")
    graph = constructGraph(features,labels, raw_data)
    print("Graph has been constructed")
    data = convertToPygFormat(graph, features, labels)
    print("Data has been converted to Pyg")

    #Split data into train, val and test
    num_nodes = data.num_nodes
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    data.train_mask[:int(Config.TRAIN_RATIO * num_nodes)] = True
    data.val_mask[int(Config.TRAIN_RATIO * num_nodes):int((Config.TRAIN_RATIO + Config.VAL_RATIO) * num_nodes)] = True
    data.test_mask[int((Config.TRAIN_RATIO + Config.VAL_RATIO)*num_nodes):] = True

    #Initialize model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GAT(num_features=data.num_features, hidden_channels=Config.HIDDEN_CHANNELS, num_classes=Config.NUM_CLASSES, heads=Config.HEADS).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)

    #Training loop
    for epoch in range(Config.EPOCHS):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')

    return model, data

if __name__ == "__main__":
    train()