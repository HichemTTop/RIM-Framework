import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm,GATConv,SAGEConv
from torch_geometric.nn import GINConv, global_add_pool
from torch.nn import Linear, BatchNorm1d
from dgl.nn import GraphConv

class GCNRegression(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCNRegression, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # First Convolutional layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Second Convolutional layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Third Convolutional layer
        x = self.conv3(x, edge_index)

        return x
        

class GAT(nn.Module):
        def __init__(self, num_features, hidden_channels, num_classes):
            super(GAT, self).__init__()
            self.conv1 = GATConv(num_features, hidden_channels)
            self.conv2 = GATConv(hidden_channels, hidden_channels)
            self.conv3 = GATConv(hidden_channels, num_classes)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index

            # First Attention layer
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

            # Second Attention layer
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

            # Third Attention layer
            x = self.conv3(x, edge_index)

            return x
class EnhancedGCN(nn.Module):
    def __init__(self, num_features=64, hidden_channels=128, num_classes=1):
        super(EnhancedGCN, self).__init__()
        
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)
        
        self.conv4 = GCNConv(hidden_channels, num_classes)
        
        # Multi-head attention layer (optional)
        # Uncomment the line below if you want to use GAT instead of GCN
        # self.gat_conv = GATConv(hidden_channels, hidden_channels, heads=2)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # First Convolutional layer
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.5, training=self.training)
        
        # Second Convolutional layer
        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=0.5, training=self.training)
        
        # Third Convolutional layer
        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)
        x3 = F.dropout(x3, p=0.5, training=self.training)
        
        # Skip Connection
        x3 += x1
        
        # Fourth Convolutional layer
        x4 = self.conv4(x3, edge_index)
        
        # Uncomment the lines below if you want to use GAT
        # x5 = self.gat_conv(x3, edge_index)
        # x5 = x5.mean(dim=1)
        
        return x4  # or return x5 if using GAT
class GINRegression(nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GINRegression, self).__init__()
        self.conv1 = GINConv(nn.Linear(num_features, hidden_channels))
        self.conv2 = GINConv(nn.Linear(hidden_channels, hidden_channels))
        self.conv3 = GINConv(nn.Linear(hidden_channels, hidden_channels))
        self.lin = nn.Linear(hidden_channels, 1)  # Ensure this matches the expected output per node

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = self.lin(x)  # Output shape should be [num_nodes, 1]
        return x
class SAGERegression0(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(SAGERegression, self).__init__()
        self.conv1 = SAGEConv(in_feats, hid_feats, 'mean')
        self.conv2 = SAGEConv(hid_feats, out_feats, 'mean')

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
class SAGERegression(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, dropout_rate=0.6):
        super(SAGERegression, self).__init__()
        self.conv1 = SAGEConv(in_feats, hid_feats, 'min')
        self.bn1 = BatchNorm(hid_feats)
        self.conv2 = SAGEConv(hid_feats, out_feats, 'min')
        self.bn2 = BatchNorm(out_feats)
        self.dropout = dropout_rate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        
        return x
class SAGERegression2(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, dropout_rate=0.5):
        super(SAGERegression, self).__init__()
        self.conv1 = SAGEConv(in_feats, hid_feats, 'mean')
        self.bn1 = BatchNorm(hid_feats)
        
        self.conv2 = SAGEConv(hid_feats, hid_feats, 'mean')
        self.bn2 = BatchNorm(hid_feats)
        
        self.conv3 = SAGEConv(hid_feats, hid_feats, 'mean')
        self.bn3 = BatchNorm(hid_feats)
        
        self.conv4 = SAGEConv(hid_feats, out_feats, 'mean')
        self.bn4 = BatchNorm(out_feats)
        
        self.dropout = dropout_rate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        
        return x