# Netpro2vec
A graph embedding technique based on probability distribution representations of graphs and skip-gram learning model

Authors: Ichcha Manipur, Maurizio Giordano, Mario Rosario Guarracino, Lucia Maddalena, Ilaria Granata - 
High Performance Computing and Networking (ICAR), Italian National Council of Research (CNR) - 
Mario Manzo - University of Naples "L'Orientale"

----------------------
Description
----------------------

Netpro2vec is a neural embedding framework, based on probability distribution representations of graphs,namedNetpro2vec. The goal is to look at basic node descriptions other than the degree, such as those induced by the TransitionMatrix and Node Distance Distribution.Netpro2vecprovides embeddings completely independent from the task and nature of the data.The framework is evaluated on synthetic and various real biomedical network datasets through a comprehensive experimentalclassification phase and is compared to well-known competitors

----------------------
Citation Details
----------------------
  
This work is the subject of the article:

Ichcha Manipur, Mario Manzo, Ilaria Granata, Maurizio Giordano*, Lucia Maddalena, andMario R. Guarracino
"Netpro2vec: a Graph Embedding Framework for Biomedical Applications".
"IEEE/ACM TCBB JOURNAL - Special Issue on Deep Learning and Graph Embeddings for Network Biology".
 
Bibtex:

```
@ARTICLE{9425591,  author={Manipur, Ichcha and Manzo, Mario and Granata, Ilaria and Giordano, Maurizio and Maddalena, Lucia and Guarracino, Mario R.},  journal={IEEE/ACM Transactions on Computational Biology and Bioinformatics},   title={Netpro2vec: A Graph Embedding Framework for Biomedical Applications},   year={2022},  volume={19},  number={2},  pages={729-740},  doi={10.1109/TCBB.2021.3078089}}
```

----------------------
License
----------------------
  
The source code is provided without any warranty of fitness for any purpose.
You can redistribute it and/or modify it under the terms of the
GNU General Public License (GPL) as published by the Free Software Foundation,
either version 3 of the License or (at your option) any later version.
A copy of the GPL license is provided in the "GPL.txt" file.

----------------------
Requirements
----------------------

To run the code the following software must be installed on your system:

1. Python 3.6 (later versions may also work)

An the following python packages:

1. Network
2. gensim
5. scipy
6. joblib
7. tqdm

----------------------
Running
----------------------

To test the code you can run the command (in <code>src</code> folder) on MUTAG dataset:
```
$ python test.py --input-path datasets/MUTAG/graphml 
   --labelfile datasets/MUTAG/MUTAG.txt 
   --label-position 2
   --distributions tm1 
   --extractors 1 
   --cutoffs 0.01 
   --aggregators 0 
   --verbose
```