[RIM Framework.pdf](https://github.com/user-attachments/files/20610001/RIM.Framework.pdf)
# Misinformation Mitigation in Online Social Networks Using Continual Learning with Graph Neural Networks
In today’s digital landscape, online social networks (OSNs) facilitate rapid information dissemination. However, they also serve
as conduits for misinformation, leading to severe real-world consequences such as public panic, social unrest, and the erosion
of institutional trust. Existing rumor influence minimization strategies predominantly rely on static models or specific diffusion
mechanisms, restricting their ability to dynamically adapt to the evolving nature of misinformation. To address this gap, this paper
proposes a novel misinformation influence mitigation framework that integrates Graph Neural Networks (GNNs) with continual
learning and employs a Node Blocking strategy as its intervention approach. The framework comprises three key components: (1)
a Dataset Generator, (2) a GNN Model Trainer, and (3) an Influential Node Identifier. Given the scarcity of real-world data on
misinformation propagation, the first component simulates misinformation diffusion processes within social networks, leveraging
the Human Individual and Social Behavior (HISB) model as a case study. The second component employs GNNs to learn from these
synthetic datasets and predict the most influential nodes susceptible to misinformation. Subsequently, these nodes are strategically
targeted and blocked to minimize further misinformation spread. Finally, the continual learning mechanism ensures the model
dynamically adapts to evolving network structures and propagation patterns. Experimental evaluations conducted on multiple
benchmark datasets demonstrate the superiority of the proposed node blocking framework over state-of-the-art methods. Our
results indicate a statistically significant reduction in misinformation spread, with non-parametric statistical tests yielding p-values
below 0.001 (p < 0.001), confirming the robustness of our approach. This work presents a scalable and adaptable solution for
misinformation containment, contributing to the development of more reliable and trustworthy online information ecosystems.

## Installation



1. Install virtualenv using pip:
```
$ pip install virtualenv
```




2. Create a new virtual environment:
```
$ virtualenv env
```



3. Activate the virtual environment:
```
$ source env/bin/activate
```



4. Install the required Python packages:
```
$ pip install -r requirements.txt
```
## Usage

1. Upload the Facebook graph and synthetic graphs to Neo4j using main_load_graph.py:
```
$ python main_load_graph.py
```



2. Run Training by running TrainModel.py:
```
$ python TrainModel.py
```

3. Load Acceptance Rate values to the graph by running load_AccpR.py:
```
$ python load_AccpR.py
```
4. Run simulation by running main_Sim_Compare.py:
```
$ python main_Sim_Compare.py
```
