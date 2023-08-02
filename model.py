import torch
import torch.nn as nn

class ReturnModel(nn.Module):
    def __init__(self, input_size1, input_size2, hidden_size, output_size):
        super().__init__()
        
        self.gru1 = nn.GRU(input_size=input_size1, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        self.gru2 = nn.GRU(input_size=input_size2, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
        self.concat = nn.Linear(2*hidden_size, 2*hidden_size)
        
        self.fc = nn.Linear(2*hidden_size, output_size)

    def forward(self, x1, x2):
        out1, _ = self.gru1(x1) 
        out1 = self.bn1(out1[:, -1, :])
        
        out2, _ = self.gru2(x2)
        out2 = self.bn2(out2[:, -1, :])
        
        out = torch.cat([out1, out2], dim=1)
        out = self.concat(out)
        out = self.fc(out)
        return out

# This is the correct model taking time serial input
class GRUModel_serial(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        
        self.gru1 = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru1(x) 
        out = self.bn1(out[:, -1, :])
        out = self.fc(out)
        return out

# This model is wrong
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.gru(x)
        output = self.bn(output)
        output = self.fc(output)
        return output
    

# 模型定义
class SepModel(nn.Module):
    def __init__(self, day_model, input_size, hidden_size, output_size):
        super().__init__()

        # train 15min model
        self.min_model = GRUModel_serial(input_size, hidden_size, output_size)        
        # Freeze day model parameters
        self.day_model = day_model
        self.day_model.eval()

    def forward(self, x1, x2):
        y1 = self.min_model(x1)
        y2 = self.day_model(x2)

        return y1 + y2
