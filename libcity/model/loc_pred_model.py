from abc import ABC
from itertools import zip_longest
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from logging import getLogger
import torch.utils
import torch.utils.data
from torch.nn.utils.rnn import pack_padded_sequence
from libcity.utils import accuracy, weight_init
from torch.utils.data import Dataset
import copy
import pdb
    

class EncoderLSTMForRoadEmb(nn.Module):
    def __init__(self,  embed_layer, input_size, output_size, hidden_size, loc_map, n_layers=1, dropout=0):
        super(EncoderLSTMForRoadEmb, self).__init__()

        self.embed_layer = embed_layer
        key_list = list(loc_map.keys())
        loc_map[max(key_list) + 1] = embed_layer.shape[0] - 1
        self.loc_map = torch.tensor(list(loc_map.values()))
        self.out_linear = nn.Linear(hidden_size, output_size)

        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, dropout=dropout)

    def forward(self, inputs):
        
        input_mask = inputs['mask']
        road_emb = self.embed_layer[self.loc_map[inputs['seq']]].float()

        PathEmbedOri = road_emb  # (B, T, hidden)
        PathEmbed = PathEmbedOri.transpose(1, 0)  # (T, B, hidden)

        outputs, hidden = self.lstm(PathEmbed, None)
        outputs = outputs.transpose(0, 1)  # (B, T, hidden)

        input_mask = input_mask.unsqueeze(-1)
        
        return self.out_linear(torch.sum(outputs * input_mask, 1) / torch.sum(input_mask, 1))


def pad_to_tensor(data,fill_value=0):
    src=np.transpose(np.array(list(zip_longest(*data, fillvalue=fill_value))))
    return torch.from_numpy(src)

def split_src_trg(full,pre_len=1):
    src_seq, trg_seq = zip(*[[s[:-pre_len], s[-pre_len:]] for s in full])
    return src_seq,trg_seq


class collection:
    def __init__(self,padding_value,device) -> None:
        self.padding_value=padding_value
        self.device=device

    def collection_TUL(self,batch):
        # build batch
        user_index, full_seq, weekday, timestamp, length, time_delta, dist, lat, lng = zip(*batch)
        inputs_seq=pad_to_tensor(full_seq,self.padding_value).long().to(self.device)

        mask = [[1] * len(i) for i in inputs_seq]
        mask = pad_to_tensor(mask, 0).long().to(self.device)

        inputs_weekday=pad_to_tensor(weekday,0).long().to(self.device)
        inputs_timestamp=pad_to_tensor(timestamp).to(self.device)
        length=np.array(length)
        imputs_time_delta = pad_to_tensor(time_delta).to(self.device)
        dist =  pad_to_tensor(dist).to(self.device)
        lat = pad_to_tensor(lat).to(self.device)
        lng = pad_to_tensor(lng).to(self.device)
        inputs_hour = (inputs_timestamp % (24 * 60 * 60) / 60 / 60).long()
        src_duration = ((inputs_timestamp[:, 1:] - inputs_timestamp[:, :-1]) % (24 * 60 * 60) / 60 / 60).long()
        src_duration = torch.clamp(src_duration, 0, 23)
        res=torch.zeros([src_duration.size(0),1],dtype=torch.long).to(self.device)
        inputs_duration = torch.hstack([res,src_duration])
        # user_index=torch.tensor(user_index).long().to(self.device)
        targets=torch.tensor(user_index).long().to(self.device)
        
        inputs={
            'seq':inputs_seq,
            'timestamp':inputs_timestamp,
            'length':length,
            'time_delta':imputs_time_delta,
            'hour':inputs_hour,
            'duration':inputs_duration,
            'weekday':inputs_weekday,
            'dist':dist,
            'lat':lat,
            'lng':lng,
            'user':user_index,
            'mask':mask
        }

        return inputs, targets
    
    def collection_LP(self,batch):
        # build batch
        user_index, full_seq, weekday, timestamp, length, time_delta, dist, lat, lng = zip(*batch)
        src_seq, trg_seq = split_src_trg(full_seq)
        
        mask = [[1] * len(i) for i in src_seq]
        mask = pad_to_tensor(mask, 0).long().to(self.device)

        inputs_seq=pad_to_tensor(src_seq,self.padding_value).long().to(self.device)
        targets=torch.tensor(trg_seq).squeeze().to(self.device)
        
        src_weekday,_ = split_src_trg(weekday)
        inputs_weekday=pad_to_tensor(src_weekday,0).long().to(self.device)

        src_time,_ = split_src_trg(timestamp)
        inputs_timestamp=pad_to_tensor(src_time).to(self.device)
        
        length=np.array(length)-1

        src_td,_ = split_src_trg(time_delta)
        imputs_time_delta = pad_to_tensor(src_td).to(self.device)

        src_dist,_ = split_src_trg(dist)
        dist =  pad_to_tensor(src_dist).to(self.device)

        src_lat,_ = split_src_trg(lat)
        src_lng,_ = split_src_trg(lng)
        lat = pad_to_tensor(src_lat).to(self.device)
        lng = pad_to_tensor(src_lng).to(self.device)
        inputs_hour = (inputs_timestamp % (24 * 60 * 60) / 60 / 60).long()
        try:
            src_duration = ((inputs_timestamp[:, 1:] - inputs_timestamp[:, :-1]) % (24 * 60 * 60) / 60 / 60).long()
        except:
            pdb.set_trace()
        src_duration = torch.clamp(src_duration, 0, 23)
        res=torch.zeros([src_duration.size(0),1],dtype=torch.long).to(self.device)
        inputs_duration = torch.hstack([res,src_duration])
        user_index=torch.tensor(user_index).long().to(self.device)
        
        inputs={
            'seq':inputs_seq,
            'timestamp':inputs_timestamp,
            'length':length,
            'time_delta':imputs_time_delta,
            'hour':inputs_hour,
            'duration':inputs_duration,
            'weekday':inputs_weekday,
            'dist':dist,
            'lat':lat,
            'lng':lng,
            'user':user_index,
            'mask': mask
        }

        return inputs, targets

class List_dataset(Dataset):
    def __init__(self,data):
        self.data=data
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

def trajectory_based_classification(train_set, test_set, num_class, embed_layer,embed_size,hidden_size, num_epoch, num_loc,batch_size,loc_map,task='LP', aux_embed_size=32, device="CPU"):
    # build dataset
    eval_ind=int(len(train_set)*0.8)
    eval_set=train_set[eval_ind:]
    train_set=train_set[:eval_ind]
    logger=getLogger()
    collect=collection(num_loc,device)

    train_dataloader=torch.utils.data.DataLoader(List_dataset(train_set),batch_size=batch_size,shuffle=True,collate_fn=collect.collection_LP if task=="LP" else collect.collection_TUL)
    eval_dataloader=torch.utils.data.DataLoader(List_dataset(eval_set),batch_size=batch_size,shuffle=False,collate_fn=collect.collection_LP if task=="LP" else collect.collection_TUL)
    test_dataloader=torch.utils.data.DataLoader(List_dataset(test_set),batch_size=batch_size,shuffle=False,collate_fn=collect.collection_LP if task=="LP" else collect.collection_TUL)
    # build model

    model = EncoderLSTMForRoadEmb(embed_layer=embed_layer,output_size=num_class,input_size=embed_size,hidden_size=hidden_size,loc_map=loc_map)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_func = nn.CrossEntropyLoss()
    # train 
    best_model=model
    best_acc=0
    patience=10000000
    for epoch in range(num_epoch):
        losses=[]
        for (inputs,targets) in train_dataloader:
            preds=model(inputs)
            if preds.shape[0] == 1:
                continue
            # pdb.set_trace()
            loss=loss_func(preds,targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # valid
        model.eval()
        y_preds=[]
        y_trues=[]
        for (inputs,targets) in eval_dataloader:
            preds=model(inputs)
            if preds.shape[0] == 1:
                continue
            preds=preds.argmax(-1)
            y_preds.extend(preds.cpu().detach().tolist())
            y_trues.extend(targets.cpu().detach().tolist())

        model.train()
        
        c_acc=accuracy_score(y_trues,y_preds)
        if c_acc > best_acc:
            patience=10000000
            best_acc=c_acc
            best_model=copy.deepcopy(model)
        else:
            patience-=1
            if patience==0:
                break
        
        logger.info(f"epoch:{epoch} loss:{round(np.mean(losses),4)} valid_acc:{round(c_acc,4)} best_acc:{round(best_acc,4)}")
    model=best_model
    # test
    model.eval()
    y_preds=[]
    y_trues=[]
    for (inputs,targets) in test_dataloader:
        preds=model(inputs)
        y_preds.extend(preds.cpu().detach())
        if len(targets.shape)==0:
            targets=targets.unsqueeze(0)
        y_trues.extend(targets.cpu().detach())

    y_preds=torch.vstack(y_preds)
    y_trues=torch.vstack(y_trues)

    acc1,acc5=accuracy(y_preds,y_trues,topk=(1,5))
    pres = y_preds.argmax(-1)
    f1ma=f1_score(y_trues.numpy(), pres.numpy(), average='macro')
    result=[acc1.item(),acc5.item(),f1ma]
    logger.info(f"task:{task} acc1:{round(acc1.item(),4)} acc5:{round(acc5.item(),4)} f1ma:{round(f1ma,4)}")
    return result

