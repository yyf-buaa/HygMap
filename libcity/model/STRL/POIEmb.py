import torch
import torch.nn as nn

class POIEmbedding(nn.Module):

    def __init__(self, config, data_feature):
        super().__init__()
        self.emb_dim = config['emb_dim']
        self.hid_dim = config['output_dim']

        self.poi_len_lon_encode = data_feature['poi_len_lon_encode']
        self.poi_len_lat_encode = data_feature['poi_len_lat_encode']

        #self.poi_len_country_emb = nn.Embedding(self.poi_len_country, self.emb_dim, padding_idx=0)
        self.poi_len_lon_encode_emb = nn.Embedding(self.poi_len_lon_encode, self.emb_dim)
        self.poi_len_lat_encode_emb = nn.Embedding(self.poi_len_lat_encode, self.emb_dim)
        self.seq_cat_emb = nn.Linear(self.emb_dim * 2, self.hid_dim)

    def forward(self, batch_seq_cat):
        """
            batch_seq_cat: (B, len_seqs_cat_feas)
        """
        #poi_len_country_emb = self.poi_len_country_emb(batch_seq_cat[:, 0])
        poi_len_lon_encode_emb = self.poi_len_lon_encode_emb(batch_seq_cat[:, 0])
        poi_len_lat_encode_emb = self.poi_len_lat_encode_emb(batch_seq_cat[:, 1])

        cat_list = [poi_len_lon_encode_emb,
                                  poi_len_lat_encode_emb
                                  ]
        sparse_vec = torch.cat(cat_list, dim=1)  # (B, 5 * emb_dim)
        sparse_vec = self.seq_cat_emb(sparse_vec)   # (B, hid_dim)
        return sparse_vec

