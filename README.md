# [VLDB2024] Jointly Learning Representations for Heterogeneous Spatial-temporal Entities via Graph Contrastive Learning

This is a PyTorch implementation of HeterOgeneous Map Entity Graph Contrastive Learning (**HOME-GCL**) for generic road segment and land parcel representation learning, **submitted to VLDB2024.**

## Abstract

The electronic map plays a crucial role in geographic information systems, serving various urban managerial scenarios and daily life services. Developing effective Spatial-Temporal Representation Learning (STRL) methods is crucial to extracting embedding information from electronic maps and converting map entities into representation vectors for downstream applications. However, existing STRL methods typically focus on one specific category of map entities, such as POIs, road segments, or land parcels, which is insufficient for real-world diverse map-based applications and might lose latent structural and semantic information interacting between entities of different types. Moreover, using representations generated by separate models for different map entities can introduce inconsistencies. Motivated by this, we propose a novel method named HOME-GCL for learning representations of multiple categories of map entities. Our approach utilizes a heterogeneous map entity graph (HOME graph) that integrates both road segments and land parcels into a unified framework. A HOME encoder with parcel-segment joint feature encoding and heterogeneous graph transformer is then deliberately designed to convert segments and parcels into representation vectors. Moreover, we introduce two types of contrastive learning tasks, namely intra-entity and inter-entity tasks, to train the encoder in a self-supervised manner. Extensive experiments on three large-scale datasets covering road segment-based, land parcel-based, and trajectory-based tasks demonstrate the superiority of our approach. To the best of our knowledge, HOME-GCL is the first attempt to jointly learn representations for road segments and land parcels using a unified model.

## Requirements

Our code is based on **Python version 3.9, PyTorch version 2.0.1, and torch-geometric version 2.3.1.** 

Please make sure you have installed Python, [PyTorch](https://pytorch.org/), and [torch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) correctly. 

Then you can install all the dependencies with the following command by pip:

```
pip install -r requirements.txt
```

## Data

We conduct our experiments on three datasets, including **BJ** (Beijing), **CD** (Chengdu) and **XA** (Xi'an).

You can download the pre-processed Chengdu and Xi'an datasets from this [link](https://bhpan.buaa.edu.cn/link/AAEEAB5C149807497699646C6BA18DC743) or this [link](https://pan.baidu.com/s/1l6TWh7HgYgKL-p2J6LDzqQ?pwd=o67k).  The downloaded data need to be placed in the `./raw_data` directory, forming the following structure: `raw_data/cd/*` and `raw_data/xa/*`.

Since the Beijing dataset is too large, you can get it from this link [here](https://github.com/aptx1231/START).

## Train & Test

You can train **HOME-GCL** through the following commands：

```shell
python run_model.py --dataset ${name}
```

You need to replace the \${name} as `cd` or `xa`. 

A field `exp_id` is generated to mark the experiment number during the experiment. Once the training is completed, the performance of the representations generated by the model on the road segment-based downstream task and the land parcel-based downstream task is automated.

- The pre-trained model will be stored at `libcity/cache/{exp_id}/model_cache/{exp_id}_{model_name}_{dataset}.pt`.
- The road segment representations generated by the model will be stored at `libcity/cache/{exp_id}/evaluate_cache/road_embedding_{model_name}_{dataset}_{dim}.npy`.
- The land parcel representations generated by the model will be stored at `libcity/cache/{exp_id}/evaluate_cache/region_embedding_{model_name}_{dataset}_{dim}.npy`.
- The model's evaluation results for the road segment-based downstream task and the land parcel-based downstream task are stored in the `libcity/cache/{exp_id}/evaluate_cache/{exp_id}_evaluate_{model_name}_{dataset}_{dim}.csv`.

For trajectory-based downstream tasks, you can run the following command:

```shell
python traj_task.py --dataset ${name} --emb_id ${exp_id}
```

You need to replace the \${name} as `cd` or `xa`.

You need to replace the \${exp_id} as the `exp_id` generated above.

The experiment generates a new `exp_id2` and the test results of the two trajectory tasks are saved at `libcity/cache/{exp_id2}/evaluate_cache/*.csv`.
