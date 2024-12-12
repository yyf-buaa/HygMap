# [ICDE2025]Representing All Categories of Map Entities based on Hybrid Graph Modeling

## Data

Both datasets (Porto, San Francisco) are from the **veccity** project.[Bigscity-VecCity/VecCity: Official repository of paper "VecCity: A Taxonomy-guided Library for Map Entity Representation Learning".](https://github.com/Bigscity-VecCity/VecCity)

The data after **Map Entity Modeling** is stored in the https://bhpan.buaa.edu.cn/link/AAB0C8968DA0CC4208B383FAC92817465B.

## Code

HeteroGE:libcity/model/STRL/IntraEncoder.py

HyperGE:libcity/model/STRL/InterEncoder.py

# run

```
python run_model.py --dataset {dataset} --model {model}
```

