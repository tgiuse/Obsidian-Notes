
# Third File





# Fourth File
Why do software carbon emissions matter?

> [!NOTE] Why do software carbon emissions matter?
> Most people associate carbon emission with *hardware* : laptops, servers and smartphones. 
> But *software* also contributes to carbon emissions indirectly

The 3% of global emissions come from the software industry.
A badly written software can increase the CPU cycles, RAM usage, battery drain and network usage. This is due to the fact that softwares are a hidden yet powerful driver of energy consumption via hardware. 

## Software Carbon Intensity
Developed by the Green Software Foundation (GSF), the SCI helps measure how much carbon is emitted due to running software
$$SCI = \frac{E\times I + M}{R}$$
where:


|Symbol     |Meaning     |
| --- | --- |
|E     |    Energy Consumed |
|I     |   Location-based Marginal Carbon Intensity  |
|M     |  Embodied Carbon: emissions from manufacturing & disposal    |
|R     |  Functional Unit   |
*SCI is a rate not an absolute total, it tells you carbon per operation, not the total emissions*

1. Energy Consumption (E)
	it includes both idle and running energy.
2. Carbon Intensity (I)
	since it is region-specific, each country has different carbon levels (moving the software to a region with **clean energy** lowers SCI significantly).
3. Embodied Carbon (M)
	this is the carbon that comes from making and disposing of hardware, it is shared across all software running on the machine, it depends on how long the app runs and what percentage of hardware it uses (if a server emits 100 kg of $\text{CO}_2$ to manufacture and the software uses it 10% of the time at 25% load $\rightarrow$ M is equal to 2.5 kg of $\text{CO}_2$ ) .
4. Functional Unit (R)
	it is a fixed value used to compare the emissions fairly.

There are some principles when it comes to being *carbon-aware* and *energy-efficient*.
1. Use less power to do the same work.
2. Use less hardware.
3. Run the software when clean energy is abundant.

There are also some best practices for when it comes to training ML models: Reduce the dataset (down to 20% can still retain good accuracy), use efficient algorithms.


# Fifth File

What is machine learning? 
Machine learning (ML) is a branch of AI where systems learn from data without being explicitly programmed.
Definitions:
* **Arthur Samuel (1954)**: Computers learn without being explicitly programmed.
* **Tom Mithchell (1997)** : A program learns from experience(E), for a Task(T), measured by performance(P), if performance improves with experience.
* **Yaser Abu-Mostafa (2012)**: Systems adaptively improve from observed data.

There are three types of Machine Learning: 


|  Type|   Description  | Examples |
| --- | --- | --- |
|  Supervised|   Labeled data; known output  |  Classification, regression|
|  Unsupervised   |    No labels; find structure | Clustering (k-means, DBSCAN) |
| Reinforcement    |  Learning through interaction with environment   | Stock trading bots |


## Supervised Learning
Let $X$ = input space (features)
Let $Y$ = output space (targets)
The goal is to find the function $h: X \rightarrow Y s.t. h(x) \approx y$

Common problem of *Supervised Machine Learning* are:
* **Overfitting** : Excellent on training, bad on test
* **Solution** : Regularization, cross-validation, more data or simpler models

The purpose of *Cross Validation* is to evaluate the model's ability to generalize unseen data, typical settings for k-fold CV are the following:
* Split data into k parts 
* Train on k-1 parts, validate on 1
* Rotate and repeat
* Average the results


How do you split the dataset (**NO OVERLAP, independent subsets**)? 

|  Case   | Split Example    |
| --- | --- |
| No hyperparams     | 80% training, 20% test     |
|  With hyperparams   |    60% training, 20% validation, 20% test |
*the second option is the more common one*

For *Recursive Feature Elimination (RFE)* we remove one feature at a time, retrain the model and track the performance, rank the features based on how much performance drops without them.

---

## Unsupervised Learning (Clustering)
Since there are no labels $\rightarrow$ find patterns/ association in data
There are multiple clustering algorithms


| Algorithm    |Strength     |Limitation |
| --- | --- | --- |
|   **k-Means**  |   Simple, fast for convex shapes   | Fails on complex geometries |
|    **Hierarchical** | Shows nested structure     | Memory and computationally expensive |
|  **DBSCAN**   |  Detects concave clusters, noise   | Needs tuning $\varepsilon$ & MinPts |
|   **OPTICS** | Extends DBSCAN with density plots    |More complex |

What is **DBSCAN**? 
There are two key parameters: 
* **$\varepsilon$** which indicates the neighborhood radius
* **MinPts** which indicates the minimum amount of points required to form a cluster.
There are three types of points:
* Core Point : $\ge$ MinPts in $\varepsilon$ neighborhood
* Border Point : $\lt$ the MinPts but near the core point 
* Noise Point : Not in any cluster

*DBSCAN excels where density ( and not shape) defines clusters (e.g. irregular geographical data)*

# Sixth File

Neural Networks represents the backbone of modern deep learning, but they come with environmental consequences. Each hidden unit contains weights that continuously updated during the training process to enable the model to make accurate predictions. The number of hidden layers and units are considered as hyper parameters,  that must be set before the training begins. 

Training deep learning models impacts heavily the environment because of the massive computational requirements. That's why there are multiple types of Neural Networks, such as *Probabilistic Neural Networks* (PNNs).
This type of networks offer a different approach to traditional multi-layer perceptrons. They compute values as probability estimates of class membership instead of direct classification. The key advantage lies in the classification of lazy learning  models, instead of assigning weights to the neurons through backpropagation, PNNs simply assign values to weights based on the training data structure. 

---
Data distillation represents a powerful technique for creating more sustainable machine learning practices.
The fundamental concept involves synthesizing small, high fidelity data summaries that capture the most important knowledge from larger target datasets. These  distilled summaries serve as effective replacements for the original datasets.

For creating the distilled datasets we must take account of the data selection. There are three strategies for sampling:
1. **Uncertain sampling**: identifies data points where the model shows least confidence, indicating areas where additional training may grant maximum  
























