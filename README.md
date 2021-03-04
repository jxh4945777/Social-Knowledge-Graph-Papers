# Social-Knowledge-Graph-Papers
Research about Social Knowledge Graph

动态图专题： https://github.com/Cantoria/dynamic-graph-papers/

- [Static Graph Representation](#static-graph-representation)
    + [Semi-Supervised Classification with Graph Convolutional Networks](#semi-supervised-classification-with-graph-convolutional-networks)
    + [Inductive representation learning on large graphs](#inductive-representation-learning-on-large-graphs)
- [Heterogeneous Graph/Heterogeneous Information Network Representation](#heterogeneous-graph-heterogeneous-information-network-representation)
  * [Heterogeneous Graph/Heterogeneous Information Network Representation - 最新综述](#heterogeneous-graph-heterogeneous-information-network-representation-------)
    + [Heterogeneous Network Representation Learning: A Unified Framework with Survey and Benchmark](#heterogeneous-network-representation-learning--a-unified-framework-with-survey-and-benchmark)
    + [Heterogeneous Network Representation Learning](#heterogeneous-network-representation-learning)
    + [异质信息网络分析与应用综述](#-------------)
  * [Heterogeneous Graph/Heterogeneous Information Network Representation - 相关前沿研究(2019 -至今)](#heterogeneous-graph-heterogeneous-information-network-representation----------2019-----)
    + [Heterogeneous Graph Attention Network](#heterogeneous-graph-attention-network)
    + [Heterogeneous Graph Transformer](#heterogeneous-graph-transformer)
    + [An Adaptive Embedding Framework for Heterogeneous Information Networks](#an-adaptive-embedding-framework-for-heterogeneous-information-networks)
    + [Modeling Relational Data with Graph Convolutional Networks](#modeling-relational-data-with-graph-convolutional-networks)
    + [Relation Structure-Aware Heterogeneous Information Network Embedding](#relation-structure-aware-heterogeneous-information-network-embedding)
    + [Fast Attributed Multiplex Heterogeneous Network Embedding](#fast-attributed-multiplex-heterogeneous-network-embedding)
    + [Genetic Meta-Structure Search for Recommendation on Heterogeneous Information Network](#genetic-meta-structure-search-for-recommendation-on-heterogeneous-information-network)
    + [Homogenization with Explicit Semantics Preservation for Heterogeneous Information Network](#homogenization-with-explicit-semantics-preservation-for-heterogeneous-information-network)
    + [Heterogeneous Graph Structure Learning for Graph Neural Networks](#heterogeneous-graph-structure-learning-for-graph-neural-networks)
    + [Learning Intents behind Interactions with Knowledge Graph for Recommendation](#learning-intents-behind-interactions-with-knowledge-graph-for-recommendation)
    + [MultiSage: Empowering GCN with Contextualized Multi-Embeddings on Web-Scale Multipartite Networks](#multisage--empowering-gcn-with-contextualized-multi-embeddings-on-web-scale-multipartite-networks)
- [Dynamic Graph Representation](#dynamic-graph-representation)
  * [Dynamic Graph Representation -- 最新综述](#dynamic-graph-representation--------)
    + [Representation Learning for Dynamic Graphs: A Survey](#representation-learning-for-dynamic-graphs--a-survey)
    + [Foundations and modelling of dynamic networks using Dynamic Graph Neural Networks: A survey](#foundations-and-modelling-of-dynamic-networks-using-dynamic-graph-neural-networks--a-survey)
    + [Temporal Link Prediction: A Survey](#temporal-link-prediction--a-survey)
  * [Dynamic Graph Representation -- 相关前沿研究(2019 - 至今)](#dynamic-graph-representation-----------2019------)
    + [DYREP: LEARNING REPRESENTATIONS OVER DYNAMIC GRAPHS](#dyrep--learning-representations-over-dynamic-graphs)
    + [Context-Aware Temporal Knowledge Graph Embedding](#context-aware-temporal-knowledge-graph-embedding)
    + [Real-Time Streaming Graph Embedding Through Local Actions](#real-time-streaming-graph-embedding-through-local-actions)
    + [Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks](#predicting-dynamic-embedding-trajectory-in-temporal-interaction-networks)
    + [dyngraph2vec-Capturing Network Dynamics using Dynamic Graph Representation Learning](#dyngraph2vec-capturing-network-dynamics-using-dynamic-graph-representation-learning)
    + [EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs](#evolvegcn--evolving-graph-convolutional-networks-for-dynamic-graphs)
    + [Temporal Graph Networks for Deep Learning on Dynamic Graphs](#temporal-graph-networks-for-deep-learning-on-dynamic-graphs)
    + [Modeling Dynamic Heterogeneous Network for Link Prediction using Hierarchical Attention with Temporal RNN](#modeling-dynamic-heterogeneous-network-for-link-prediction-using-hierarchical-attention-with-temporal-rnn)
    + [DySAT: Deep Neural Representation Learning on Dynamic Graphs via Self-Attention Networks](#dysat--deep-neural-representation-learning-on-dynamic-graphs-via-self-attention-networks)
    + [Evolving network representation learning based on random walks](#evolving-network-representation-learning-based-on-random-walks)
    + [Relationship Prediction in Dynamic Heterogeneous Information Networks](#relationship-prediction-in-dynamic-heterogeneous-information-networks)
    + [Link Prediction on Dynamic Heterogeneous Information Networks](#link-prediction-on-dynamic-heterogeneous-information-networks)
    + [TemporalGAT: Attention-Based Dynamic Graph Representation Learning](#temporalgat--attention-based-dynamic-graph-representation-learning)
    + [Continuous-Time Relationship Prediction in Dynamic Heterogeneous Information Networks](#continuous-time-relationship-prediction-in-dynamic-heterogeneous-information-networks)
    + [Continuous-Time Dynamic Graph Learning via Neural Interaction Processes](#continuous-time-dynamic-graph-learning-via-neural-interaction-processes)
    + [A Data-Driven Graph Generative Model for Temporal Interaction Networks](#a-data-driven-graph-generative-model-for-temporal-interaction-networks)
    + [Motif-Preserving Temporal Network Embedding](#motif-preserving-temporal-network-embedding)
- [Dynamic & Heterogeneous Graph Representation](#dynamic---heterogeneous-graph-representation)
  * [Dynamic & Heterogeneous Graph Representation -- 相关前沿研究(2019 - 至今)](#dynamic---heterogeneous-graph-representation-----------2019------)
    + [DHNE: Network Representation Learning Method for Dynamic Heterogeneous Networks](#dhne--network-representation-learning-method-for-dynamic-heterogeneous-networks)
    + [Modeling Dynamic Heterogeneous Network for Link Prediction using Hierarchical Attention with Temporal RNN](#modeling-dynamic-heterogeneous-network-for-link-prediction-using-hierarchical-attention-with-temporal-rnn-1)
    + [Dynamic Heterogeneous Information NetworkEmbedding with Meta-path based Proximity](#dynamic-heterogeneous-information-networkembedding-with-meta-path-based-proximity)
    + [Relationship Prediction in Dynamic Heterogeneous Information Networks](#relationship-prediction-in-dynamic-heterogeneous-information-networks-1)
    + [Link Prediction on Dynamic Heterogeneous Information Networks](#link-prediction-on-dynamic-heterogeneous-information-networks-1)
- [Social Relation Reasoning](#social-relation-reasoning)
  * [Social Relation Reasoning -- 相关前沿研究(2017-至今)](#social-relation-reasoning-----------2017----)
    + [TransNet: Translation-Based Network Representation Learning for Social Relation Extraction](#transnet--translation-based-network-representation-learning-for-social-relation-extraction)
    + [Deep Reasoning with Knowledge Graph for Social Relationship Understanding](#deep-reasoning-with-knowledge-graph-for-social-relationship-understanding)
    + [Relation Learning on Social Networks with Multi-Modal Graph Edge Variational Autoencoders](#relation-learning-on-social-networks-with-multi-modal-graph-edge-variational-autoencoders)
    + [Graph Attention Networks over Edge Content-Based Channels](#graph-attention-networks-over-edge-content-based-channels)
- [Knowledge Graph (\#TODO)](#knowledge-graph----todo-)
  * [Knowledge Graph - 最新综述(\#TODO)](#knowledge-graph----------todo-)
    + [A Survey on Knowledge Graphs: Representation, Acquisition and Applications](#a-survey-on-knowledge-graphs--representation--acquisition-and-applications)
  * [Knowledge Graph - 相关前沿研究(\#TODO)](#knowledge-graph------------todo-)
- [Others](#others)
- [Related Datasets](#related-datasets)
- [其他参考资料](#------)

## Static Graph Representation
挑选了引用数较高、知名度较大的一些静态图表示学习的工作。

#### Semi-Supervised Classification with Graph Convolutional Networks
* 作者：Thomas N. Kipf, et al. (University of Amsterdam)
* 发表时间：2016
* 发表于：ICLR 2017
* 标签：图神经网络
* 概述：提出了图卷积神经网络的概念，并使用其聚合、激活节点的一阶邻居特征。
* 链接：https://arxiv.org/pdf/1609.02907.pdf
* 相关数据集：
    * Citeseer
    * Cora
    * Pubmed
    * NELL
* 是否有开源代码：有
#### Inductive representation learning on large graphs
* 作者： Hamilton W, et al.(斯坦福大学Leskovec团队)
* 发表时间：2017
* 发表于：Advances in neural information processing systems
* 标签：Inductive Graph Embedding
* 概述：针对以往transductive的方式（不能表示unseen nodes）的方法作了改进，提出了一种inductive的方式改进这个问题，该方法学习聚合函数，而不是某个节点的向量
* 链接：https://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf
* 相关数据集：
    * Citation
    * Reddit
    * PPI
* 是否有开源代码：有


## Heterogeneous Graph/Heterogeneous Information Network Representation
异质图/异构图(Heterogeneous Graph) = 异质信息网络(Heterogeneous Information Network)

### Heterogeneous Graph/Heterogeneous Information Network Representation - 最新综述
该部分包括了异质图的最新综述论文

#### Heterogeneous Network Representation Learning: A Unified Framework with Survey and Benchmark
* 作者： Carl Yang, et al.(UIUC韩家炜团队)
* 发表时间：2020
* 发表于：TKDE
* 标签：Heterogeneous Network Reprensentation Learning
* 概述：本文是异质图相关研究的综述文章，系统性地梳理了异质图的经典工作以及前沿工作，将已有工作规范到统一的框架内，且提出了异质图表示学习的Benchmark，并且对于经典的异质图方法进行了复现与评测。
* 链接：https://arxiv.org/abs/2004.00216
* 相关数据集：
    * DBLP
    * Yelp
    * Freebase
    * PubMed
* 是否有开源代码：有 https://github.com/yangji9181/HNE

#### Heterogeneous Network Representation Learning
* 作者： Yuxiao Dong, et al.(UCLA Yizhou Sun 团队)
* 发表时间：2020
* 发表于：IJCAI 2020
* 标签：Heterogeneous Network Reprensentation Learning
* 概述：本文是UCLA, THU, Microsoft合作的一篇异质图相关研究的综述文章，首先阐述了异质图信息挖掘的含义以及相关研究，并且从传统图嵌入表示和异质图神经网络两个角度阐述了异质图的表示学习，且与知识图谱相关的表示学习工作进行了对比。最后从预训练、多任务学习、动态性等角度对于异质图的研究进行了展望。
* 链接：http://web.cs.ucla.edu/~yzsun/papers/2020_IJCAI_HIN_Survey.pdf
* 相关数据集：
    * OAG
* 是否有开源代码：有 https://github.com/HeterogeneousGraph

#### 异质信息网络分析与应用综述
* 作者： Chuan Shi, et al.
* 发表时间：2020
* 发表于：软件学报
* 标签：Heterogeneous Information Network
* 概述：本文是一篇关于异质信息网络的最新中文综述，对于异质信息网络给出了明确的定义，并且对于现有异质信息网络的从网络结构的角度进行了归类，对于异质信息网络表示学习相关的工作也进行了归类为基于图分解的方法、基于随机游走的方法、基于编码器-解码器的方法以及基于图神经网络的方法。同时本文对于异质信息网络的应用进行了叙述，最后对于异质信息网络的发展提出了展望。
* 链接：http://www.shichuan.org/doc/94.pdf
* 是否有开源代码：有 https://github.com/BUPT-GAMMA/OpenHINE

### Heterogeneous Graph/Heterogeneous Information Network Representation - 相关前沿研究(2019 -至今)

#### Heterogeneous Graph Attention Network
* 作者： Xiao Wang, et al. (BUPT 石川团队)
* 发表时间：2019
* 发表于：WWW 2019
* 标签：Heterogeneous Network Reprensentation Learning, Hierarchical Attention
* 概述：本文是异质图与图神经网络结合的一篇研究工作，不同于其他图神经网络直接聚合邻居信息，HAN通过Meta-Path采集到多跳的邻居并据此将异质图同质化再聚合邻居节点，以实现聚合元路径上节点的信息；HAN同时提出分层attention机制，用于衡量不同邻居的权重，以及不同语义(元路径)信息的权重。在以上思路的基础上学习异质图中节点的表示，并且通过节点分类和节点聚类两个下游任务预测模型的有效性，以及分析了模型的可解释性。
* 链接：http://www.shichuan.org/doc/66.pdf
* 相关数据集：
    * DBLP
    * IMDB
    * ACM
* 是否有开源代码：有 https://github.com/Jhy1993/HAN

#### Heterogeneous Graph Transformer
* 作者： Ziniu Hu, et al. (UCLA Yizhou Sun团队)
* 发表时间：2020
* 发表于：WWW 2020
* 标签：Heterogeneous Network Reprensentation Learning, Transformer, Multi-Head Attention
* 概述：考虑到已有异质图的研究存在以下几点局限：1. 需要人工设计Meta-path；2.无法建模动态信息；3.对于大规模的异质图，缺乏有效的采样方式。针对于以上三点，本文首选给出Meta Relation的概念，直接建模相连的异质节点，基于此设计了类Transformer的网络结构用于图表示学习。考虑到异质图的动态特性，本文提出了RTE编码方式，用于建模异质图的动态演化。考虑到大规模异质图上网络的训练，本文提出了HGSampling方式，用于均匀采样不同类型的节点信息，以实现高效的图表示学习。
* 链接：https://arxiv.org/abs/2003.01332
* 相关数据集：
    * OAG
* 是否有开源代码：有 https://github.com/acbull/pyHGT

#### An Adaptive Embedding Framework for Heterogeneous Information Networks
* 作者： Daoyuan Chen, et al. (阿里)
* 发表时间：2020
* 发表于：CIKM 2020
* 标签：Heterogeneous Information Network, Knowledge Graph, Joint Learning
* 概述：作为模式十分丰富的异质图之一，知识图谱的表示学习一直是研究的重点之一，本文关注于传统的Trans系列知识图谱表示学习方法没法很好地捕获到高阶(多跳)关系之间节点的相似性，因此设计了一种联合学习的方式，首先通过Trans系列方法学习几点的表示，并且通过类似于(h+r-t)的打分函数进行打分，该分数用于指导图上的随机游走概率(即Trans方法学的越不好，越有可能游走到这些节点)，然后通过Skip-Gram再进行节点表示的优化。同时本文针对于Skip-Gram的游走路径长度，以及窗口大小，设计了一套自适应机制。本框架适用于大多数Trans系列方法，具有很强的灵活性，值得借鉴。
* 链接：https://dl.acm.org/doi/10.1145/3340531.3411989
* 相关数据集：
    * FILM
    * Cora
    * Citeseer
    * WN18
    * FB15K-237
* 是否有开源代码：无

#### Modeling Relational Data with Graph Convolutional Networks
* 作者： Michael Schlichtkrull, Thomas N. Kipf, et al. (阿姆斯特丹Kipf团队)
* 发表时间：2018
* 发表于：ESWC 2018
* 标签：Knowledge Graph, Multi Relation, Graph Neural Network
* 概述：本文关注于真实世界图中边的异质性，例如FB15K-237和WN18包含多种类型的边。现有图神经网络GCN无法建模边的异质性，因此本文提出了R-GCN模型，在信息传递时对于不同类型的边使用不同的权值矩阵，同时考虑到在边比较多的情况下矩阵的数目也较多，因此采取了共享权值的方式，将每种类型边的权值矩阵视作多个基的带权加和，以此缩小参数量。对于实验部分，本文在FB15K，和WN18两个数据集上，从实体分类以及连接预测(知识图谱补全)两个实验角度验证了模型的有效性。
* 链接：https://arxiv.org/abs/1703.06103
* 相关数据集：
    * WN18
    * FB15K-237
* 是否有开源代码：有(https://github.com/tkipf/relational-gcn)

#### Relation Structure-Aware Heterogeneous Information Network Embedding
* 作者： Yuanfu Lu, et al. (BUPT 石川团队)
* 发表时间：2019
* 发表于：AAAI 2019
* 标签：Heterogeneous Graph, Relation Structure, Random Walk
* 概述：本文关注到异质图中不同Meta-path的结构性区别，核心就是将预定义的Meta-path通过统计分析分成两种类型-从属关系/交互关系，对于从属关系，本文计算节点相似度的方法是直接通过欧氏距离；对于交互关系，本文计算节点之间的关系是通过类似于TransE的Translation方法。通过两种不同类型关系的联合学习，最终能够做到考虑不同关系类型(从属/交互)的节点表示。最终本文通过节点聚类、节点分类、连接预测验证了模型的有效性。
* 链接：https://arxiv.org/abs/1905.08027
* 相关数据集：
    * DBLP
    * Yelp
    * AMiner
* 是否有开源代码：有(https://github.com/rootlu/RHINE)

#### Fast Attributed Multiplex Heterogeneous Network Embedding
* 作者： Zhijun Liu, et al. 
* 发表时间：2020
* 发表于：CIKM 2020
* 标签：Heterogeneous Graph, Fast Learning
* 概述：本文考虑到现有异质图表示学习方法从效率角度难以应用于大规模异质图数据上，因此提出了一个新的模型框架FAME，用于快速学习异质图上节点的表示。其主要贡献在于
提出了一个新的图表示学习方法，使用随机映射的方式代替feature trasformation的方式(即随机删掉部分维度)。实验部分，本文在多个数据集上验证了模型的有效性，无论是从效率上，还是准确率上，都高于现有的Baseline方法。
* 链接：https://dl.acm.org/doi/10.1145/3340531.3411944
* 相关数据集：
    * Alibaba
    * Amazon
    * Aminer
    * IMDB
* 是否有开源代码：有(https://github.com/ZhijunLiu95/FAME)

#### Genetic Meta-Structure Search for Recommendation on Heterogeneous Information Network
* 作者： Zhenyu Han, et al. (THU)
* 发表时间：2020
* 发表于：CIKM 2020
* 标签：Heterogeneous Graph, Genetic Algorithm
* 概述：本文考虑到异质图能够很好地建模推荐系统，但手动设计Meta-Path需要大量的人工，因此需要研究自动发现Meta-Path的方法。受优化问题中遗传算法的启发，本文设计了一个类似于遗传算法的Meta-Structure自动挖掘策略，用于推荐系统。实验部分，本文在Yelp, Douban Movie, Amazon三个数据集上进行了实验验证模型的有效性，同时通过给出Case Study，验证模型能够学习到新的有用的Meta-Structure。
* 链接：https://dl.acm.org/doi/10.1145/3340531.3412015
* 相关数据集：
    * Yelp
    * Douban
    * Movie
    * Amazon
* 是否有开源代码：有(https://github.com/0oshowero0/GEMS)

#### Homogenization with Explicit Semantics Preservation for Heterogeneous Information Network
* 作者： Tiancheng Huang, et al. (ZJU)
* 发表时间：2020
* 发表于：CIKM 2020
* 标签：Heterogeneous Graph, Homogenization
* 概述：本文考虑到现有异质图算法在将图同质化的过程中(例如HAN)忽略了路径上的节点的丰富信息，且损失了大量的原本图中的信息。因此本文从异质图的同质化角度入手，设计了新的表示学习方法，能够使转化同质子图的过程中同时考虑路径上节点的信息。具体来讲，本文首先设定对称的Meta-path作为考虑对象，对于路径中对称的节点衡量其相似性，以此作为Meta-path重要性的参照。实验部分，本文在DBLP, IMDB，Yelp数据集上以节点分类和节点聚类作为任务进行了实验，验证了模型的有效性。
* 链接：https://dl.acm.org/doi/10.1145/3340531.3412015
* 相关数据集：
    * Yelp
    * IMDB
    * DBLP
* 是否有开源代码：有(https://dl.acm.org/doi/10.1145/3340531.3412135)

#### Heterogeneous Graph Structure Learning for Graph Neural Networks
* 作者： Jianan Zhao, et al. (BUPT石川团队)
* 发表时间：2021
* 发表于：AAAI 2021
* 标签：Heterogeneous Graph, Structure Learning, Graph Neural Network
* 概述：本文关注于现实世界中异质图是存在噪音和缺失的现象，因此针对于此首次提出异质图结构学习的相关工作，希望通过建模异质图的节点特征和已有图的拓补结构特征，能够学习到新的异质图结构，实现对于现有异质图缺失的结构的补充。具体来讲，本文提出了异质图结构学习模型HGSL，首先根据节点的特征信息以及邻居信息(对于关系r, 度量节点相似度，并连接相似节点生成Feature Similarity Graph -> 对于连接的节点间的邻居也进行连接 生成两个Feature Propagation Graph -> 通过Attention机制将三个生成的图进行融合)得到Feature Graph，然后对于关系r, 根据不同Meta-path利用Metapath2Vec学到的向量表示用于度量节点相似度，并生成多个子图，融合得到Semantic Graph，最终对于Feature Graph与Semantic Graph进行融合得到新的异质图结构，实现了缺失结构信息的学习与补充。实验部分，本文在DBLP, ACM, Yelp数据集上以节点分类为任务验证了模型的有效性，并且进行了相关分析。
* 链接：https://github.com/Andy-Border/HGSL/tree/main/paper
* 相关数据集：
    * Yelp
    * ACM
    * DBLP
* 是否有开源代码：有(https://github.com/Andy-Border/HGSL)

#### Learning Intents behind Interactions with Knowledge Graph for Recommendation
* 作者： Xiang Wang, et al. (新加坡国立、浙大、eBay)
* 发表时间：2021
* 发表于：WWW 2021
* 标签：Heterogeneous Graph, Knowledge Graph, Recommendation System, Graph Neural Network
* 概述：本文是一篇对于用户内容推荐算法的研究，对于User-Item的内容推荐，以往工作未考虑到其间存在的用户的意图(Intent)，因此本文定义了用户的意图，即user-intent-item，并且对此提出了Knowledge Graph Intent Graph，用KG中的relation集合来代表intent；并针对性地提出了GNN-based Method - KGIG，主要包括结合Intent的用户信息建模，以及考虑多跳异质关系路径的信息聚合，用于精准用户内容推荐。本文在三个数据集上验证了模型的有效性，且给出了全面地分析。
* 链接：https://jiyang3.web.engr.illinois.edu/files/multisage.pdf
* 相关数据集：
    * OAG
    * Printest
* 是否有开源代码：无

#### MultiSage: Empowering GCN with Contextualized Multi-Embeddings on Web-Scale Multipartite Networks
* 作者：Carl Yang, Jiawei Han, Jure Leskovec et al. (UIUC韩家炜团队, Standford Jure团队)
* 发表时间：2020
* 发表于：KDD 2020
* 标签：Recommendation System, Graph Neural Network, Web-Scale 
* 概述：本文是一篇对于用户内容推荐算法的研究，对于内容推荐主要考虑到了背景信息的作用，提出了Contextual Masking机制，用于考虑不同的上下文情下的内容表示，同时利用attention机制比较不同context的重要性差异；除此之外，本文考虑到了工业级的大规模数据推荐，提出了一套解决方案，对于中心节点的邻居，通过parallel pagerank based random walk用于进行邻居采样，然后通过Hadoop2+AWS进行数据的计算。本文在两个大规模数据集(但也是进行了采样并非完整数据集)进行了实验验证模型的有效性。
* 链接：https://arxiv.org/abs/2102.07057
* 相关数据集：
    * Amazon-Book
    * Last-FM
    * Alibaba-iFashion
* 是否有开源代码：有(https://github.com/huangtinglin/Knowledge_Graph_based_Intent_Network)
## Dynamic Graph Representation

### Dynamic Graph Representation -- 最新综述

#### Representation Learning for Dynamic Graphs: A Survey
* 作者：Seyed Mehran Kazemi, et al. (Borealis AI)
* 发表时间：2020.3
* 发表于：JMLR 21 (2020) 1-73
* 标签：动态图表示，综述
* 概述：针对目前动态图表示已有的方法，从encoder/decoder的角度进行了概述，覆盖面很全，是了解动态图研究的必读工作。
* 链接：https://deepai.org/publication/relational-representation-learning-for-dynamic-knowledge-graphs-a-survey

#### Foundations and modelling of dynamic networks using Dynamic Graph Neural Networks: A survey
* 作者：Joakim Skarding, et al. (University of Technology Sydney)
* 发表时间：2020.5
* 发表于：arXiv
* 标签：动态图表示，综述，动态图神经网络
* 概述：该文侧重于从图神经网络的角度与具体任务的角度去讲述目前动态网络的研究方向。在第二章中，作者将动态图的有关定义整理为体系，从3个维度（时态粒度、节点动态性、边持续的时间）上，分别定义了8种动态网络的定义。在第三章中，阐述了编码动态网络拓扑结构的深度学习模型；在第四章中，阐述了被编码的动态网络信息如何用于预测，即动态网络的解码器、损失函数、评价指标等。在最后一章，作者阐述了动态图表示、建模的一些挑战，并对未来的发展方向进行了展望。
* 链接：https://arxiv.org/abs/2005.07496

#### Temporal Link Prediction: A Survey
* 作者： Divakaran A, et al.
* 发表时间：2019
* 发表于：New Generation Computing (2019)
* 关键词：时态链接预测，综述
* 概述：从离散动态图（DTDG）的角度出发，本文针对时态链接预测任务给出了相关定义，并从实现方法的角度出发，构建了时态链接预测的分类体系，分别从矩阵分解/概率模型/谱聚类/时间序列模型/深度学习等不同方法实现的模型进行了比较与论述。文章还列举出了时态链接预测任务的相关数据集（论文互引网络、通讯网络、社交网络、人类交往网络数据等）。最后，文章对时态链接预测任务的难点进行了展望。
* 链接：https://link.springer.com/article/10.1007%2Fs00354-019-00065-z

### Dynamic Graph Representation -- 相关前沿研究(2019 - 至今)

#### DYREP: LEARNING REPRESENTATIONS OVER DYNAMIC GRAPHS
* 作者： Rakshit Trivedi, et al. (Georgia Institute of Technology & DeepMind)
* 发表时间：2019
* 发表于：ICLR 2019
* 关键词：CTDG
* 概述：在本文中，作者提出了一套动态图节点表示学习框架，该框架能很好地建模网络的动态演化特征，并能够对unseen nodes进行表示。有对于动态图结构中节点的交互行为，作者将其分为association与communication两种，前者代表长期稳定的联系，网络拓扑结构发生了变化，后者代表短暂、临时的联系。在节点的信息传播方面，作者将节点的信息传播定义为Localized Embedding Propagation/Self-Propagation/Exogenous Drive，分别代表节点邻居的信息聚合传播，节点自身信息传播以及外因驱动（由时间控制）。作者在dynamic link prediction & time prediction任务上对该方法的有效性进行了验证。
* 链接：https://openreview.net/pdf?id=HyePrhR5KX
* 相关数据集：
    * Social Evolution Dataset
    * Github Dataset
* 是否有开源代码：无（有第三方开源代码）

#### Context-Aware Temporal Knowledge Graph Embedding
* 作者： Yu Liu, et al. (昆士兰大学)
* 发表时间：2019
* 发表于：WISE 2019
* 关键词：时态知识图谱，知识表示
* 概述：作者认为现有的knowledge graph embedding方法忽略了时态一致性；时态一致性能够建模事实与事实所在上下文（上下文是指包含参与该事实的所有实体）的关系。为了验证时态知识图谱中事实的有效性，作者提出了上下文选择的双重策略：1、验证组成该事实的三元组是否可信；2、验证这个事实的时态区间是否与其上下文冲突。作者在实体预测/上下文选择任务上证明了方法的有效性。
* 链接：https://link.springer.com/chapter/10.1007/978-3-030-34223-4_37
* 相关数据集：
    * YAGO11k
    * Wikidata12k
* 是否有开源代码：无

#### Real-Time Streaming Graph Embedding Through Local Actions
* 作者： Xi Liu, et al. (德州农工大学)
* 发表时间：2019
* 发表于：WWW 2019
* 关键词：streaming graph
* 概述：本文认为已有的动态图嵌入式学习方法强烈依赖节点属性，时间复杂度高，新节点加入后需要重新训练等缺点。本文提出了streaming graph的概念，提出了一种动态图表示的在线近似算法。该算法能够为新加入图中的节点快速高效生成节点表示，并能够为新加入节点“影响”到的节点更新节点的表示。
* 链接：https://dl.acm.org/doi/abs/10.1145/3308560.3316585
* 相关数据集：
    * Blog
    * CiteSeer
    * Cora
    * Flickr
    * Wiki
* 是否有开源代码：无

#### Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks
* 作者： Srijan Kumar, et al. (斯坦福大学，Jure团队)
* 发表时间：2019
* 发表于：KDD 2019
* 关键词：CTDG，user-item dynamic embedding
* 概述：这篇论文解决的问题是建模user-item之间的序列互动问题。而表示学习能够为建模user-item之间的动态演化提供很好的解决方案。目前工作的缺陷是只有在user作出变化时才会更新其表示，并不能生成user/item未来的轨迹embedding。因此，作者设计了JODIE（Joint Dynamic User-Item Embeddings），其包括更新部分与预测部分。更新部分由一个耦合循环神经网络（coupled recurrent neural network）学习user与item未来轨迹。其使用了两个循环神经网络更新user/item在每次interaction的表示，还能表示user/item未来的embedding变化轨迹（trajectory）。预测部分由一个映射算子组成，其能够学习user在未来任意某个时间点的embedding表示。为了让这个方法可扩展性更强，作者提出了一个t-Batch算法，能够创建时间一致性的batch（time-consistent batch），且能够提升9倍训练速度。为了验证方法的有效性，作者在4个实验数据集上做了实验，对比了6种方法，发现在预测未来互动（predicting future interaction）任务上提升了20%，在状态变化预测（state change prediction任务上提升了12%）
* 链接：https://cs.stanford.edu/~srijan/pubs/jodie-kdd2019.pdf
* 相关数据集：
    * Reddit
    * Wikipedia
    * Last FM
    * MOOC
* 是否有开源代码：有(https://snap.stanford.edu/jodie/)

#### dyngraph2vec-Capturing Network Dynamics using Dynamic Graph Representation Learning
* 作者： Palash Goyal, et al. (南加州州立大学)
* 发表时间：2020
* 发表于：Knowledge-Based Systems
* 关键词：DTDG
* 概述：本文首先针对动态图表示学习进行了定义，即：学习到一个函数的映射，这个映射能将每个时间点的图中节点映射为向量y，并且这个向量能够捕捉到节点变化的时态模式。基于此，作者提出了一种能够捕捉动态图演化的动力学特征，生成动态图表示的方法，本质上是输入为动态图的前T个时间步的snapshot，输出为T+1时刻的图嵌入式表达。在实验中，作者采用了AE/RNN/AERNN三种编码器进行了实验。此外，作者设计了一个图embedding生成库DynamicGEM。
* 链接：https://www.sciencedirect.com/science/article/pii/S0950705119302916
* 相关数据集：
    * SBM dataset
    * Hep-th Dataset
    * AS Dataset
* 是否有开源代码：有(https://github.com/palash1992/DynamicGEM)

#### EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs
* 作者： Aldo Pareja, et al.(MIT-IBM Watson AI Lab)
* 发表时间：2019
* 发表于：AAAI 2020
* 标签：图卷积网络，DTDG
* 概述：本文不同于传统的DTDG表示学习工作，没有用RNN编码各个snapshot之间的表示，而是使用RNN去编码GCN的参数，从而学习图的演化规律。
* 链接：https://www.aaai.org/Papers/AAAI/2020GB/AAAI-ParejaA.5679.pdf
* 相关数据集：
    * Stochastic Block Model
    * Bitcoin OTC
    * Bitcoin Alpha
    * UC Irvine messages
    * Autonomous systems
    * Reddit Hyperlink Network
    * Elliptic      
* 是否有开源代码：有（https://github.com/IBM/EvolveGCN）

#### Temporal Graph Networks for Deep Learning on Dynamic Graphs
* 作者：Rossi, Emanuele, et al.（Twitter）
* 发表时间：2020.6
* 发表于：arXiv
* 标签：动态图表示，CTDG
* 概述：提出了CTDG动态图的一套通用表示框架，并提出了一种能够并行加速训练效率的算法。
* 链接：https://arxiv.org/pdf/2006.10637.pdf
* 相关数据集：
    * Wikipedia（这个数据集是不是开源的Wikidata？论文中无说明）
    * Reddit
    * Twitter
* 是否有开源代码：无

#### Modeling Dynamic Heterogeneous Network for Link Prediction using Hierarchical Attention with Temporal RNN
* 作者：Hansheng Xue, Luwei Yang, et al.（澳大利亚国立大学, 阿里巴巴）
* 发表时间：2020.4
* 发表于：arXiv
* 标签：动态图表示，异构图，注意力机制，DTDG
* 概述：本文同时考虑到图的异构性和动态性的特点，对于图的每个时间切片，利用node-level attention和edge-level attention以上两个层次的注意力机制实现异质信息的有效处理，并且通过循环神经网络结合self-attention研究节点embedding的演化特性，并且通过链接预测任务进行试验，验证模型的有效性。
* 链接：https://arxiv.org/pdf/2004.01024.pdf
* 相关数据集：
    * Twitter
    * Math-Overflow
    * Ecomm
    * Alibaba.com
* 是否有开源代码：有(https://github.com/skx300/DyHATR)

#### DySAT: Deep Neural Representation Learning on Dynamic Graphs via Self-Attention Networks
* 作者： Aravind Sankar, et al.(UIUC)
* 发表时间：2020
* 发表于：WSDE 2020
* 标签：DTDG，注意力机制
* 概述：作者提出了DYNAMIC SELF-ATTENTION NETWORK机制，通过结构化注意力模块与时态注意力模块对动态变化的节点进行表示。
* 链接：http://yhwu.me/publications/dysat_wsdm20.pdf
* 相关数据集：
    * Enron Email
    * UCI Email
    * MovieLens-10M
    * Yelp      
* 是否有开源代码：有(https://github.com/aravindsankar28/DySAT)

#### Evolving network representation learning based on random walks
* 作者： Farzaneh Heidari, et al.(York University)
* 发表时间：2020
* 发表于：Journal Applied Network Science 2020 (5)
* 标签：DTDG，随机游走
* 概述：针对DTDG动态图的4种演化行为（增加/删除节点，增加/删除边），作者提出了一种在动态图上更新已采样随机游走路径的算法，并设计了网络结构演化程度的Peak Detection算法，从而以较小代价更新不断演化的节点表示。
* 链接：https://appliednetsci.springeropen.com/articles/10.1007/s41109-020-00257-3
* 相关数据集：
    * Protein-Protein Interactions
    * BlogCatalog (Reza and Huan)
    * Facebook Ego Network(Leskovec and Krevl 2014)
    * Arxiv HEP-TH (Leskovec and Krevl 2014)  
    * Synthetic Networks (Watts-Strogatz (Newman 2003) random networks)    
* 是否有开源代码：有(https://github.com/farzana0/EvoNRL)

#### Relationship Prediction in Dynamic Heterogeneous Information Networks
* 作者： Amin Milani Fard, et al.(New York Institute of Technology)
* 发表时间：2019
* 发表于：Advances in Information Retrieval 2019 (4)
* 标签：DTDG，异质信息
* 概述：本文在考虑图动态性的同时，考虑图的异质性，认为不同类型节点对之间的关系自然有所区别，因此提出了动态异质图表示学习，并且做了规范定义。并且提出MetaDynaMix 方法，通过meta-path标注每个节点和边的特征，在此基础上通过矩阵分解得到特征向量，并用于计算关系预测时的概率。
* 链接：https://www.researchgate.net/publication/332257507_Relationship_Prediction_in_Dynamic_Heterogeneous_Information_Networks
* 相关数据集：
    * Publication Network (DBLP+ ACM)
    * Movies Network (IMDB)
* 是否有开源代码：无

#### Link Prediction on Dynamic Heterogeneous Information Networks
* 作者： Chao Kong, et al.(Anhui Polytechnic University)
* 发表时间：2019
* 发表于：Lecture Notes in Computer Science 2019
* 标签：DTDG，异质信息，广度学习，图神经网络
* 概述：本文考虑到动态图相关研究中异质信息缺乏有效的利用，且对于大规模图的表示学习过程中，深度学习方法效率较低，因此提出了一种宽度学习(?)的框架，并且与图神经网络相结合，实现高效的动态异质图表示学习。
* 链接：https://link.springer.com/chapter/10.1007%2F978-3-030-34980-6_36
* 相关数据集：
    * Reddit
    * Stack Overflow
    * Ask Ubuntu
* 是否有开源代码：无

#### TemporalGAT: Attention-Based Dynamic Graph Representation Learning
* 作者： Ahmed Fathy and Kan Li(Beijing Institute of Technology)
* 发表时间： 2020
* 发表于：PAKDD 2020
* 标签：DTDG，图神经网络
* 概述：目前的方法使用了时态约束权重（temporal regularized weights）来使节点在相邻时态状态的变化是平滑的，但是这种约束权重是不变的，无法反映图中节点随时间演化的规律。本文借鉴了GAT的思路，提出了TCN。但作者提到本文的贡献只是提高了精度，感觉并不是很有说服力。
* 链接：https://link.springer.com/chapter/10.1007/978-3-030-47426-3_32
* 相关数据集：
    * Enron
    * UCI
    * Yelp
* 是否有开源代码：无

#### Continuous-Time Relationship Prediction in Dynamic Heterogeneous Information Networks
* 作者： SINA SAJADMANESH, et al.(Sharif University of Technology)
* 发表时间：2018
* 发表于：ACM Transactions on Knowledge Discovery from Data (5)
* 标签：CTDG，异质信息
* 概述：本文同时关注到图的动态性与异质性，针对于连续时间的关系预测问题进行了定义，并且提出了一种新的特征抽取框架，通过Meta-Path以及循环神经网络实现对于异质信息与时间信息的有效利用，并且提出NP-GLM框架，用于实现关系预测(预测关系创建的时间节点)。 
* 链接：https://www.researchgate.net/publication/320195531_Continuous-Time_Relationship_Prediction_in_Dynamic_Heterogeneous_Information_Networks
* 相关数据集：
    * DBLP
    * Delicious
    * MovieLens
* 是否有开源代码：无

#### Continuous-Time Dynamic Graph Learning via Neural Interaction Processes
* 作者： Xiaofu Chang, et al.(Ant Group)
* 发表时间：2020
* 发表于：CIKM '20: Proceedings of the 29th ACM International Conference on Information & Knowledge Management
* 标签：CTDG，异质信息，时态点序列过程
* 概述：针对动态图中并存的拓扑信息与时态信息，本文提出了TDIG(Temporal Dependency Interaction Graph)的概念，并基于该概念提出了一种新的编码框架TDIG-MPNN，能够产生连续时间上的节点动态表示。该框架由TDIG-HGAN与TDIG-RGNN组成。前者能够聚合来自异质邻居节点的局部时态与结构信息；后者使用LSTM架构建模长序列的信息传递，整合了TDIG-HGAN的输出，捕捉全局的信息。此外，作者采用了一种基于注意力机制的选择算法，能够针对某一节点u，计算历史与其关联的节点对其不同重要程度分值。在训练过程中，作者将其定义为一个时态点序列过程(Temporal Point Process)问题进行优化。在实验中，作者针对时态链接预测问题，通过hit@10/Mean Rank指标对一些经典的静态图表示学习算法与STOA的动态图表示学习方法进行了对比，作者提出的模型在多个Transductive与一个Inductive数据集上取得了最好的效果。
* 链接：https://dl.acm.org/doi/pdf/10.1145/3340531.3411946
* 相关数据集：
    * CollegeMsg (Transductive)
    * Amazon (Transductive)
    * LastFM  (Transductive)
    * Huabei Trades (Inductive)
* 是否有开源代码：无

#### A Data-Driven Graph Generative Model for Temporal Interaction Networks
* 作者： Dawei Zhou, et al.(UIUC)
* 发表时间：2020
* 发表于：KDD 2020
* 标签：CTDG，图生成模型
* 概述：这篇论文是一篇深度图生成领域的文章，作者将动态图生成领域与transformer模型结合，设计了一种端到端的图生成模型TagGen。TagGen包含一种新颖的采样机制，能够捕捉到时态网络中的结构信息与时态信息。而且TagGen能够参数化双向自注意力机制，选择local operation，从而生成时态随机游走序列。最后，一个判别器（discriminator）在其中选择更贴近于真实数据的随机游走序列，将这些序列返回至一个组装模块（assembling module），生成新的随机游走序列。
作者在7个数据集上进行了实验，在跨度不同的指标中，TagGen表现更好；在具体任务（异常检测，链接预测）中，TagGen大幅度提升了性能。
* 链接：https://www.kdd.org/kdd2020/accepted-papers/view/a-data-driven-graph-generative-model-for-temporal-interaction-networks
* 相关数据集：
    * DBLP 
    * SO
    * MO
    * WIKI
    * EMAIL
    * MSG
    * BITCOIN
* 是否有开源代码：有 (https://github.com/davidchouzdw/TagGen)

#### Motif-Preserving Temporal Network Embedding
* 作者： Hong Huang, et al.(hust)
* 发表时间：2020
* 发表于：IJCAI 2020
* 标签：CTDG，motif，hawkes
* 概述：本论文采用了一种meso-dynamics的建模方法，通过一种时序网络上的motif——open triad，考虑三个节点之间的triad结构，利用Hawkes过程建模节点对之间的密度函数，来学习时态网络中的embedding。论文在节点分类、链接预测（这一部分实验写的不清楚，不太明白是怎么做的实验）、链接推荐上取得了较好的效果。）
* 链接：https://www.ijcai.org/Proceedings/2020/0172.pdf
* 相关数据集：
    * School 
    * Digg
    * Mobile
    * dblp
* 是否有开源代码：无

## Dynamic & Heterogeneous Graph Representation
### Dynamic & Heterogeneous Graph Representation -- 相关前沿研究(2019 - 至今)
#### DHNE: Network Representation Learning Method for Dynamic Heterogeneous Networks
* 作者： Ying Yin, et al.
* 发表时间：2019
* 发表于：IEEE Access
* 标签：DTDG，异质信息，动态信息， random walk
* 概述：本文同时考虑到图的异质性与动态性，通过构建Historical-Current图将中心节点的历史邻居信息与当前邻居信息进行拼接，并在此基础上进行Random Walk采样，通过Skip-Gram更新节点在当前时间的向量表示。本文在包含时间信息的DBLP和Aminer数据集上通过节点分类的下游任务验证了模型的有效性。
* 链接：https://ieeexplore.ieee.org/document/8843962
* 相关数据集：
    * AMiner
    * DBLP
* 是否有开源代码：有(https://github.com/Yvonneupup/DHNE)

#### Modeling Dynamic Heterogeneous Network for Link Prediction using Hierarchical Attention with Temporal RNN
* 作者： Hansheng Xue, et al.
* 发表时间：2020
* 发表于：ArXiv
* 标签：DTDG，异质信息，动态信息， 图神经网络
* 概述：本文提出一个能够同时学习图中动态信息和异质信息的框架DyHATR，通过类似于HAN的异质图神经网络建模每个时间步上节点的表示，其中通过分层注意力机制同时关注到聚合信息时不同节点的重要性，以及不同Meta-path的重要性。在对于每个时间切片图中学到节点的表示基础上，通过RNN来建模节点表示的演化。本文通过Link Prediction实验验证了模型的有效性。
* 链接：https://ieeexplore.ieee.org/document/8843962
* 相关数据集：
    * Twitter
    * Math-Overflow
    * Ecomm
* 是否有开源代码：有(https://github.com/skx300/DyHATR)

#### Dynamic Heterogeneous Information NetworkEmbedding with Meta-path based Proximity
* 作者： Xiao Wang, et al.
* 发表时间：2020
* 发表于：TKDE
* 标签：DTDG，异质信息，动态信息， 矩阵分解
* 概述：对于动态异质图，本文提出一种新的增量式更新方法，用于在考虑图演化的情况下节点向量表示的更新。首先本文对于静态异质图的表示学习，提出了新的StHNE模型，能够同时考虑到一阶邻居相似性以及二阶邻居相似性用于作为节点表示的参照；在此基础上，对于动态演化的异质图，本文提出DyHNE模型，将图的演化转化成特征值和特征向量的变化，并且据此提出了一套新的增量式更新的方法，用于更新节点的表示。本文通过节点分类以及关系预测验证了模型的有效性。
* 链接：https://yuanfulu.github.io/publication/TKDE-DyHNE.pdf
* 相关数据集：
    * Yelp
    * DBLP
    * AMiner
* 是否有开源代码：有(https://github.com/rootlu/DyHNE)

#### Relationship Prediction in Dynamic Heterogeneous Information Networks
* 作者： Amin Milani Fard, et al.(New York Institute of Technology)
* 发表时间：2019
* 发表于：Advances in Information Retrieval 2019 (4)
* 标签：DTDG，异质信息
* 概述：本文在考虑图动态性的同时，考虑图的异质性，认为不同类型节点对之间的关系自然有所区别，因此提出了动态异质图表示学习，并且做了规范定义。并且提出MetaDynaMix 方法，通过meta-path标注每个节点和边的特征，在此基础上通过矩阵分解得到特征向量，并用于计算关系预测时的概率。
* 链接：https://www.researchgate.net/publication/332257507_Relationship_Prediction_in_Dynamic_Heterogeneous_Information_Networks
* 相关数据集：
    * Publication Network (DBLP+ ACM)
    * Movies Network (IMDB)
* 是否有开源代码：无

#### Link Prediction on Dynamic Heterogeneous Information Networks
* 作者： Chao Kong, et al.(Anhui Polytechnic University)
* 发表时间：2019
* 发表于：Lecture Notes in Computer Science 2019
* 标签：DTDG，异质信息，广度学习，图神经网络
* 概述：本文考虑到动态图相关研究中异质信息缺乏有效的利用，且对于大规模图的表示学习过程中，深度学习方法效率较低，因此提出了一种宽度学习(?)的框架，并且与图神经网络相结合，实现高效的动态异质图表示学习。
* 链接：https://link.springer.com/chapter/10.1007%2F978-3-030-34980-6_36
* 相关数据集：
    * Reddit
    * Stack Overflow
    * Ask Ubuntu
* 是否有开源代码：无

## Social Relation Reasoning
### Social Relation Reasoning -- 相关前沿研究(2017-至今)
#### TransNet: Translation-Based Network Representation Learning for Social Relation Extraction
* 作者： Cunchao Tu, et al.(THUNLP)
* 发表时间：2017
* 发表于：IJCAI 2017
* 标签：Social Relation Extraction, Auto Encoder, Traslation Learning
* 概述：关注到社交网络中社交关系的表示学习，考虑到社交关系往往隐含在社交交互(文本)中，社交关系复杂且存在复合，难以用单一的标签表示，因此本文首先提出了社交关系抽取(Social Relation Extraction)任务并且给出了形式化的定义，将社交关系看作是多标签的复合(Multi One-hot)。此外本文正对于社交关系抽取，提出了模型TransNet，其核心分为两部分，Auto Encoder部分用于将高维的Multi One-hot向量嵌入至低维空间并且尽可能还原原始信息，类TransE部分用于对于Encoder端得到的社交关系的向量表示进行约束，使其符合(Head + Relation = Tail)的Translation关系。与之对应，模型的Loss Function也分为两部分，分别为AE的重建误差，以及Translation Learning的Score Function。关于实验部分，本文在三种不同规模的Aminer数据集上，以SRE为任务进行了实验验证模型的有效性。
* 链接：https://www.ijcai.org/proceedings/2017/399
* 相关数据集：
    * AMiner
* 是否有开源代码：有 (https://github.com/thunlp/TransNet)

#### Deep Reasoning with Knowledge Graph for Social Relationship Understanding
* 作者： ZhouxiaWang, et al. (中山大学)
* 发表时间：2018
* 发表于：WWW 2018
* 标签：Social Relation Reasoning, Computer Vision, Knowledge Graph
* 概述：本文是一篇计算机视觉领域的文章，核心思想是通过引入知识图谱的概念，用于识别图像中的社交关系。具体来讲，在训练集上，本文首先识别图像中出现的物体，并且根据图片的社交关系Ground Truth标签统计物体与社交关系的共现频率，在此基础上构建背景知识图谱(共两种类型的节点: 物品、社交关系)。在通过训练集构建的背景知识图谱基础上，引入门控图神经网络(GGNN)，用于推理图片对应的社交关系。实验部分，本文在PISC数据集上以社交关系识别作为任务验证了模型的有效性。
* 链接：http://www.ijcai.org/proceedings/2018/0142.pdf
* 相关数据集：
    * PISC
    * COCO
* 是否有开源代码：有 (https://github.com/HCPLab-SYSU/SR)

#### Relation Learning on Social Networks with Multi-Modal Graph Edge Variational Autoencoders
* 作者：Carl Yang, et al. (UIUC韩家炜团队)
* 发表时间：2019
* 发表于：WSDM 2020
* 标签：Relation Learning, Variational AutoEncoder, Social Network
* 概述：本文关注于社交网络中边的表示学习，考虑到社交信息的多模性、信息的不完整性以及充满噪音，因此本文提出了基于Variational AutoEncoder结构的新社交关系表示学习框架 - RELEARN。其Encoder端通过图神经网络GCN用于将图的结构信息以及节点的属性信息压缩，并且通过MLP学习边的表示，通过VAE概率建模的方式能够提高模型的鲁棒性。对于Decoder端，本文设计了Multi-Decoder机制，即通过Edge的嵌入式表示解码出不同类型的信息(本文解码节点属性信息/图结构信息/节点交互信息)。此外，本文考虑到无监督学习的情况，将节点间社交关系看做概率分布，对于每种社交关系设置了全局的基向量，并且通过基向量的带权加和作为边的最终向量表示。关于实验，本文在DBLP数据集以及LinkedIn数据集上，以节点分类和链接预测作为任务验证了模型的有效性。
* 链接：https://arxiv.org/abs/1911.05465
* 相关数据集：
    * DBLP
    * LinkedIn
* 是否有开源代码：有 (https://github.com/yangji9181/RELEARN)

#### Graph Attention Networks over Edge Content-Based Channels
* 作者：Lu Lin, Hongning Wang.  (Virginia University)
* 发表时间：2020
* 发表于：KDD 2020
* 标签：Edge Representation Learning, Variational AutoEncoder, Social Network
* 概述：本文同样聚焦于社交网络中边的表示学习，其核心思想在于认为节点之间的交互隐含着多种话题，且占比不同，这种话题能够通过节点之间的交互文本体现，因此本文聚焦于此对于边的信息与表示进行建模。具体来讲，本文提出了新的模型Topic-GCN，通过类似于Multi-Head Attention的方式建模话题之间的分布，用于代替原有的GAT的Attention机制，同时通过VAE去学习边的表示。实验部分，本文在Yelp和StackOverflow数据集上进行了链接预测以及内容预测实验验证模型的有效性。
* 链接：https://www.kdd.org/kdd2020/accepted-papers/view/graph-attention-networks-over-edge-content-based-channels
* 相关数据集：
    * Yelp
    * StackOverflow
* 是否有开源代码：有 (https://github.com/Louise-LuLin/topic-gcn)

## Knowledge Graph (\#TODO)
### Knowledge Graph - 最新综述(\#TODO)
#### A Survey on Knowledge Graphs: Representation, Acquisition and Applications
* 作者：Shaoxiong Ji, Shirui Pan, Erik Cambria, Senior Member, IEEE, Pekka Marttinen, Philip S. Yu, Fellow IEEE
* 发表时间：2020
* 发表于：Arxiv
* 标签：Knowledge Graph, Representation Learning
* 概述：本文是一篇知识图谱领域的前沿综述，文中给出了知识图谱的具体定义，并且从知识获取、知识表示、动态知识图谱、知识图谱的应用等多个角度围绕知识图谱技术进行了讨论。同时文章还对于知识图谱未来的发展提出了展望。
* 链接：https://arxiv.org/abs/2002.00388
* 是否有开源代码：无

### Knowledge Graph - 相关前沿研究(\#TODO)

## Others

## Related Datasets 
* DBLP
* OAG
* IMDB
* Social Evolution Dataset
* Github Dataset
* GDELT (Global data on events, location, and tone)
* ICEWS (Integrated Crisis Early Warning System)
* FB-FORUM
* UCI Message data
* YELP
* MovieLens-10M
* SNAP数据集合网站：http://snap.stanford.edu/data/index.html
* SNAP时态数据集合：http://snap.stanford.edu/data/index.html#temporal
* KONECT数据集合网站（部分数据集的edge带有时间戳，可看作时序数据）
* Network Repository：http://networkrepository.com/

## 其他参考资料
