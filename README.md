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

