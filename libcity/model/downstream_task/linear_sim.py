import logging
from torch import nn
import torch


class EncoderLSTMForRoadEmb(nn.Module):
    def __init__(self,  input_size, hidden_size, n_layers=2, dropout=0.5):
        super(EncoderLSTMForRoadEmb, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, dropout=dropout)

    def forward(self, road_emb, input_mask):
        PathEmbedOri = road_emb  # (B, T, hidden)
        PathEmbed = PathEmbedOri.transpose(1, 0)  # (T, B, hidden)

        outputs, hidden = self.lstm(PathEmbed, None)
        outputs = outputs.transpose(0, 1)  # (B, T, hidden)

        input_mask = input_mask.unsqueeze(-1)

        return torch.sum(outputs * input_mask, 1) / torch.sum(input_mask, 1)


class LinearSim(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self._logger = logging.getLogger()
        self._logger.info("Building Downstream LinearSim model with LSTM")

        self.lstm = EncoderLSTMForRoadEmb(128,128,2,0.1)

    def forward(self, road_emb,input_mask):
        traj_emb = self.lstm(road_emb,input_mask)
        return traj_emb  # (B, 128)
