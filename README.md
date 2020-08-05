# SERank
An efficient and effective learning to rank algorithm by mining information across ranking candidates.
This repository contains the tensorflow implementation of SERank model. The code is developed based on [TF-Ranking](https://github.com/tensorflow/ranking).

Compared with [GSF(Groupwise Scoring Function)](https://arxiv.org/pdf/1811.04415.pdf), our method obtains comparable ranking performance gain, while only requiring little computation overhead. 

The SERank model has been suceessfully deployed in [Zhihu Search ranking](https://www.zhihu.com/search?type=content&q=deep%20learning), which is one of the largest Community Question Answering platform in China.

![image info](./pics/flops.png)

![image info](./pics/serank_ndcg.jpg)

## Dependencies
- [tensorflow-ranking](https://github.com/tensorflow/ranking) >= 0.2.2
- [tensorflow](https://github.com/tensorflow/tensorflow) >= 2.0

## Dataset
The demo dataset in this repo is randomly sampled from MSLR web30k dataset.
You may download the whole web30K dataset from [Microsoft Learning to Rank Datasets
 Page](https://www.microsoft.com/en-us/research/project/mslr/) and place `train.txt`, `vali.txt`, `test.txt` in the data folder.
 
## Model
Our main idea is to develop a sequencewise model structure which accepts a list of ranking candidates, and jointly score all candidates.
We introduce SENet structure into the ranking model, where the basic idea is using SENet structure to compute feature importance according the the context of ranking list.
![image info](./pics/seblock.jpg)

For the detail about the model structure, you may refer to our paper published on [arxiv](https://arxiv.org/abs/2006.04084).

## How to Train
`bash run_train.sh`

## Citation
SERank: Optimize Sequencewise Learning to Rank Using Squeeze-and-Excitation Network. [[arxiv](https://arxiv.org/abs/2006.04084)]

```
@article{wang2020serank,
  title={SERank: Optimize Sequencewise Learning to Rank Using Squeeze-and-Excitation Network},
  author={Wang, RuiXing and Fang, Kuan and Zhou, RiKang and Shen, Zhan and Fan, LiWen},
  journal={arXiv preprint arXiv:2006.04084},
  year={2020}
}
```
