import matplotlib
matplotlib.use('Agg') 
import torch
import torch.nn.functional as F
from models.GAT import GAT
from utils.data_processing import loadAndProcesData, construct_graph, convert_to_pyg_format
from config import Config
import matplotlib.pyplot as plt

def train():
    # Load and preprocess data
    features, labels, raw_data, class_weights = loadAndProcesData(Config.CSV_FILE, Config.FEATURES_CSV)
    print("Data has been processed")
    graph = construct_graph(features, labels, raw_data)
    print("Graph has been created")
    data = convert_to_pyg_format(graph, features, labels)
    print("Data has been converted")

    #Split data into train, val and test
    num_nodes = data.num_nodes
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    data.train_mask[:int(Config.TRAIN_RATIO * num_nodes)] = True
    data.val_mask[int(Config.TRAIN_RATIO * num_nodes):int((Config.TRAIN_RATIO + Config.VAL_RATIO) * num_nodes)] = True
    data.test_mask[int((Config.TRAIN_RATIO + Config.VAL_RATIO)*num_nodes):] = True

    # Initialize model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GAT(num_features=data.num_features, hidden_channels=Config.HIDDEN_CHANNELS, num_classes=Config.NUM_CLASSES, heads=Config.HEADS).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)

    # Initialize lists to store losses
    train_losses = []
    val_losses = []

    # Training loop
    print("Start Training Loop")
    for epoch in range(Config.EPOCHS):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        
        # Use weighted loss function
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask], weight=class_weights.to(device))
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # Validation loss (no class weights)
        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)
            val_loss = F.nll_loss(val_out[data.val_mask], data.y[data.val_mask])
            val_losses.append(val_loss.item())

        print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training vs. Validation Loss')
    plt.savefig('training_validation_loss.png')  # Save the plot to a file
    plt.close()  # Close the plot to free memory

    # Save the model
    torch.save(model.state_dict(), 'gat_network_security_model.pth')
    print("Model saved to 'gat_network_security_model.pth'")

    return model, data

if __name__ == "__main__":
    train()