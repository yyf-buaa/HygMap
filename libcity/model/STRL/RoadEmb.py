import torch
import torch.nn as nn

class RoadEmbedding(nn.Module):

    def __init__(self, config, data_feature):
        super().__init__()
        self.emb_dim = config['emb_dim']
        self.hid_dim = config['output_dim']

        #self.road_len_highway = data_feature['road_len_highway']
        self.road_len_lanes = int(data_feature['road_len_lanes'])
        self.road_len_maxspeed = int(data_feature['road_len_maxspeed'])
        # self.road_len_tunnel = data_feature['road_len_tunnel']
        # self.road_len_bridge = data_feature['road_len_bridge']
        # self.road_len_roundabout = data_feature['road_len_roundabout']
        # self.road_len_oneway = data_feature['road_len_oneway']
        self.road_len_lon_encode = data_feature['road_len_lon_encode']
        self.road_len_lat_encode = data_feature['road_len_lat_encode']
        self.road_len_length_encode = data_feature['road_len_length_encode']
        self.road_len_lanes_emb = nn.Embedding(self.road_len_lanes, self.emb_dim, padding_idx=0)
        self.road_len_maxspeed_emb = nn.Embedding(self.road_len_maxspeed, self.emb_dim, padding_idx=0)
        # self.road_len_tunnel_emb = nn.Embedding(self.road_len_tunnel, self.emb_dim, padding_idx=0)
        # self.road_len_bridge_emb = nn.Embedding(self.road_len_bridge, self.emb_dim, padding_idx=0)
        # self.road_len_roundabout_emb = nn.Embedding(self.road_len_roundabout, self.emb_dim, padding_idx=0)
        # self.road_len_oneway_emb = nn.Embedding(self.road_len_oneway, self.emb_dim, padding_idx=0)
        self.road_len_length_encode_emb = nn.Embedding(self.road_len_length_encode, self.emb_dim)
        self.road_len_lon_encode_emb = nn.Embedding(self.road_len_lon_encode, self.emb_dim)
        self.road_len_lat_encode_emb = nn.Embedding(self.road_len_lat_encode, self.emb_dim)

        self.seq_cat_emb = nn.Linear(self.emb_dim * 5, self.hid_dim)

    def forward(self, batch_seq_cat):
        """
            batch_seq_cat: (B, len_seqs_cat_feas)
        """
        road_len_lanes_emb = self.road_len_lanes_emb(batch_seq_cat[:, 0])
        road_len_maxspeed_emb = self.road_len_maxspeed_emb(batch_seq_cat[:, 1])
        # road_len_tunnel_emb = self.road_len_tunnel_emb(batch_seq_cat[:, 3])
        # road_len_bridge_emb = self.road_len_bridge_emb(batch_seq_cat[:, 4])
        # road_len_roundabout_emb = self.road_len_roundabout_emb(batch_seq_cat[:, 5])
        # road_len_oneway_emb = self.road_len_oneway_emb(batch_seq_cat[:, 6])
        road_len_length_encode_emb = self.road_len_length_encode_emb(batch_seq_cat[:, 2])
        road_len_lon_encode_emb = self.road_len_lon_encode_emb(batch_seq_cat[:, 3])
        road_len_lat_encode_emb = self.road_len_lat_encode_emb(batch_seq_cat[:, 4])

        cat_list = [road_len_lanes_emb, road_len_maxspeed_emb,
                                  road_len_length_encode_emb, road_len_lon_encode_emb,
                                  road_len_lat_encode_emb
                                  ]
        sparse_vec = torch.cat(cat_list, dim=1)  # (B, 5 * emb_dim)
        sparse_vec = self.seq_cat_emb(sparse_vec)   # (B, hid_dim)
        return sparse_vec

