import DataStorage.GraphGenerator as gg
import Simulation.Simulator as sim
import numpy as np
import pandas as pd
from tqdm import  tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GCNConv , GATConv, BatchNorm
from torch_geometric.utils import from_networkx
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import data
from torch_geometric.loader import DataLoader
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from Models import Models
from node2vec import Node2Vec
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# define rate limit handler function
if __name__ == '__main__':
    def duplicate_graph(g, num_simulations):
        new_graph = nx.Graph()
        
        for i in range(num_simulations):
            # Relabel nodes for uniqueness
            mapping = {node: f"{node}_{i}" for node in g.nodes()}
            relabeled_g = nx.relabel_nodes(g, mapping)
            
            # Combine graphs
            new_graph = nx.union(new_graph, relabeled_g)
            print("Graph concatinated" + str(i))
            
        return new_graph
   
    parameters = {'omega_min': np.pi/24,
                  'omega_max': np.pi*2,
                  "delta_min": np.pi/24,
                  "delta_max": np.pi/2,
                  "jug_min": 0.01,
                  "jug_max": 0.99,
                  "beta_max": 1.2,
                  "beta_min": 0.2}

    Generator=gg.CreateGraphFrmDB()
    Simulator = sim.RumorSimulator()
    
   # Initialize an empty list to store the simulation DataFrames
    simulation_dataframes = []
  
    # Number of simulations to run
    num_simulations = 1  # You can set this to any number
    typeOfSim=0
    NbrSim = 5
    i=0
    blockPeriod=10
    for i in tqdm(range(num_simulations)):
        g = Generator.CreateGraph(parameters,graphModel='FB')
        # Run the simulation
        k=int(0.15*g.number_of_nodes())
        df = Simulator.runSimulation(g,NbrSim=NbrSim ,seedsSize=0.05, typeOfSim=typeOfSim,simName=f'sim{i}',verbose=True,method='none',k=k,blockPeriod=blockPeriod)
        
        # Add an identifier for the simulation run
        df['simulation_run'] = i
        # Store DataFrame in list
        simulation_dataframes.append(df)

    # Concatenate all DataFrames
    combined_df = pd.concat(simulation_dataframes, ignore_index=True)
    g= duplicate_graph(g, num_simulations)
    print("---------------------------------------------------------------------------------")

    for i in range(num_simulations):
        for index, row in combined_df.iterrows():    
            node_id_df = str(index)
            node_id_graph = node_id_df + f"_{i}"
            acceptance = row['AccpR']
            
            if node_id_graph in g:
                g.nodes[node_id_graph]['AccpR'] = acceptance
                #print(f"Node ID: {node_id_graph}, Acceptance Rate: {acceptance}")
            
    acceptance = [g.nodes[node].get('AccpR', 0) for node in g.nodes()]
    acceptance_tensor = torch.tensor(acceptance, dtype=torch.float)
    
    #Extracting Features manually
    # Initialize an empty list to hold feature vectors
    features_list = []

    # Loop through each node in the NetworkX graph
    for node in g.nodes():
        # Extract or compute the features for the node
        # For demonstration, let's say the feature vector for each node is [attribute1, attribute2]
        attribute1 = g.nodes[node].get('degree')  # Replace 'attribute1' with your actual attribute name
        attribute2 = g.nodes[node].get('degree_centrality')  # Replace 'attribute2' with your actual attribute name
        attribute3 = g.nodes[node].get('closeness_centrality')
        attribute4 = g.nodes[node].get('between_centrality')
        attribute5 = g.nodes[node].get('page_rank')
        attribute6 = g.nodes[node].get('group')
        feature_vector = [attribute1,attribute2,attribute3,attribute4,attribute5]
        
        # Append the feature vector to the list
        features_list.append(feature_vector)
    
    # Convert the list of feature vectors to a PyTorch tensor
    features_tensor = torch.tensor(features_list, dtype=torch.float)

    data = from_networkx(g)
    data.x = features_tensor   
    data.y = acceptance_tensor
    # Assuming `data` is your single PyTorch Geometric data object
    num_nodes = data.num_nodes

    # Calculate the sizes of each split
    train_size = int(0.7 * num_nodes)
    val_size = int(0.15 * num_nodes)
    test_size = num_nodes - train_size - val_size

    # Create masks for each set
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # Randomly select nodes for each set and set their mask to True
    train_nodes = torch.randperm(num_nodes)[:train_size]
    val_nodes = torch.randperm(num_nodes)[train_size:train_size + val_size]
    test_nodes = torch.randperm(num_nodes)[train_size + val_size:]

    train_mask[train_nodes] = True
    val_mask[val_nodes] = True
    test_mask[test_nodes] = True

    # Add masks to your data object
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    training_losses = []
    validation_losses = []
    validation_r2_scores = []
    validation_maes = []
    validation_mses = []
    # Initialize model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # Initialize the model and optimizer
    model = Models.SAGERegression(in_feats=features_tensor.shape[1], hid_feats=32, out_feats=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
    # Define a threshold for accuracy
    #threshold = 1  # Adjust as needed

    # Training loop
    for epoch in range(300):  # 100 epochs, adjust as needed
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data)
        # Compute loss only for the training nodes
        loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask].view(-1, 1))
        training_losses.append(loss.item())
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        val_loss = F.mse_loss(out[data.val_mask], data.y[data.val_mask].to(device).view(-1, 1)).item()
        validation_losses.append(val_loss)

        # Evaluation metrics
        pred = out[data.val_mask].detach().cpu().numpy()
        actual = data.y[data.val_mask].detach().cpu().numpy()
        r2 = r2_score(actual, pred)
        mae = mean_absolute_error(actual, pred)
        mse = mean_squared_error(actual, pred)

        validation_r2_scores.append(r2)
        validation_maes.append(mae)
        validation_mses.append(mse)

        print(f'Epoch: {epoch+1}, Validation Loss: {val_loss}, R2 Score: {r2:.2f}, MAE: {mae:.2f}, MSE: {mse:.2f}')

    # Testing
    model.eval()
    test_loss = F.mse_loss(out[data.test_mask], data.y[data.test_mask].to(device).view(-1, 1)).item()
    pred = out[data.test_mask].detach().cpu().numpy()
    actual = data.y[data.test_mask].detach().cpu().numpy()
    test_r2 = r2_score(actual, pred)
    test_mae = mean_absolute_error(actual, pred)
    test_mse = mean_squared_error(actual, pred)

    print(f'Test Loss: {test_loss}, Test R2 Score: {test_r2:.2f}, Test MAE: {test_mae:.2f}, Test MSE: {test_mse:.2f}')

    # Plot training and validation loss
    plt.figure()
    plt.plot(range(1, len(training_losses) + 1), training_losses, label='Training Loss')
    plt.plot(range(1, len(validation_losses) + 1), validation_losses, label='Validation Loss')
    plt.title('Epoch vs. Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot R-squared scores
    plt.figure()
    plt.plot(range(1, len(validation_r2_scores) + 1), validation_r2_scores, label='Validation R2 Score')
    plt.title('Epoch vs. R2 Score')
    plt.xlabel('Epoch')
    plt.ylabel('R2 Score')
    plt.legend()
    plt.show()

    # Plot Mean Absolute Error
    plt.figure()
    plt.plot(range(1, len(validation_maes) + 1), validation_maes, label='Validation MAE')
    plt.title('Epoch vs. MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.show()

    # Plot Mean Squared Error
    plt.figure()
    plt.plot(range(1, len(validation_mses) + 1), validation_mses, label='Validation MSE')
    plt.title('Epoch vs. MSE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.show()
    # Assuming you choose to save the model with the lowest validation loss
    max_val_r2_score = max(validation_r2_scores)
    filename = f"GCN_{max_val_r2_score:.2f}_r2_score.pth"
    torch.save(model.state_dict(), filename)
    print(f"Model saved as {filename}")

    embeddings = [model_node_embeddings.wv[str(node_id)] for node_id in node_ids]
    features_numpy = np.array(embeddings)

    # Convert to PyTorch tensor
    features_tensor = torch.tensor(features_numpy, dtype=torch.float) """
