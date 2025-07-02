
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

## Data Distillation

Data distillation represents a powerful technique for creating more sustainable machine learning practices.
The fundamental concept involves synthesizing small, high fidelity data summaries that capture the most important knowledge from larger target datasets. These  distilled summaries serve as effective replacements for the original datasets.

Steps involved in data distillation process:

**Data collection**: Assemble a diverse dataset that is relevant to the task you want the neural network to perform.

**Data preprocessing**: Clean the dataset by addressing issues such as missing values, noise reduction, and normalization of input features. This may also include encoding categorical variables and splitting the dataset into training , validation, and test sets.

**Initial model training** : Train an initial neural network using the complete dataset. This is a reference point for evaluating the effectiveness of the distillation process.

**Prediction generation**: Use the trained model to make predictions on the dataset. This step generates output labels that can help identify which data points are informative 

**Data selection for distillation**: There are three strategies for sampling:
1. **Uncertain sampling**: identifies data points where the model shows least confidence, indicating areas where additional training may grant maximum benefit.
2. **High-confidence sampling**: selects examples where the model performs exceptionally well preserving clear examples of correct behavior.
3. **Diversity sampling**: ensures that the selected subset covers a wide range of scenarios

**Creating the distilled dataset**: Forms a new smaller dataset from the selected examples. 

**Retraining the model**: Train a new neural network model using distilled dataset. This model should perform better or similar to the initial model while using fewer resources.

**Evaluation**: Assess the performance of the newly trained model on a validation set and compare the performances.

---

## Hardware Efficiency

Embodied carbon refers to the total greenhouse gas emissions associated with the entire life cycle of a product or building, from raw material extraction to manufacturing, transportation, construction, use, and disposal. It's essentially the carbon footprint of everything that goes into creating something, before it's even put to use.

If we take into account the embodied carbon, it is clear that by the time we come to buy a device, it has already emitted a good amount of carbon.

How can we improve hardware efficiency?
* extending the lifespan 
* increasing the utilization

Energy efficiency is the cornerstone of the principles of Green Computing, requiring designers to create components that consume minimal power during operation while maintaining performance standards. 

Sustainable materials play a crucial role in reducing environmental impact. This involves selecting recyclable and biodegradable materials sourced through sustainable practices, thereby minimizing the ecological footprint of raw material extraction and manufacturing processes. The emphasis on sustainable sourcing extends throughout the supply chain.

Longevity and durability represent essential design considerations. Hardware built to last longer reduces the frequency of replacements, directly decreasing electronic waste generation. This approach requires robust engineering and quality materials that can withstand extended use without degrading performance.

Modularity enables users to upgrade and repair individual components rather than replacing entire systems. This design philosophy extends hardware lifespan by allowing selective improvements and repairs, reducing overall waste generation. Modular designs also facilitate easier maintenance and customization.

Recyclability ensures that hardware can be efficiently disassembled and processed at the end of its useful life. This principle requires designing products with separation and material recovery in mind, facilitating responsible disposal and enabling valuable material reclamation.

Low emissions manufacturing focuses on reducing environmental impact during production, transportation, and disposal phases. This involves optimizing manufacturing processes, selecting low-impact transportation methods, and ensuring proper end-of-life handling.

Smart resource management incorporates technologies that optimize resource utilization throughout the hardware lifecycle. Dynamic power management adjusts energy consumption based on current needs, while virtualization technologies maximize hardware utilization efficiency by enabling multiple applications to share resources effectively.

# Seventh File

## Neural Network Pruning 

Neural network pruning addresses computational efficiency by removing network components that contribute minimally to overall performance. This technique creates more efficient models requiring less computational power and memory, making them suitable for deployment on resource-constrained devices while maintaining acceptable performance levels.

There are several approaches to pruning:

**Weight pruning** removes individual weights that fall below specified thresholds, effectively zeroing out less important connections. This approach creates sparser networks with fewer active parameters, reducing computational requirements during both training and inference phases.

**Neuron pruning** removes entire neurons or nodes based on their contribution to model accuracy. This method typically involves analyzing each neuron's impact on overall performance and eliminating those with minimal influence. Neuron pruning often provides more significant efficiency gains than weight pruning while maintaining model interpretability.

**Structured pruning** removes entire network structures such as channels, filters, or layers rather than individual components. This approach often yields better performance improvements on standard hardware because it maintains regular computational patterns that processors can handle efficiently.

--- 

## Transfer Learning

Transfer learning is one of the most powerful techniques in modern machine learning. This concept mirrors how humans learn new skills by building upon existing knowledge and experience.
Consider the analogy of a skilled guitar player learning to play the ukulele. The musician's existing knowledge of finger positioning, music theory, rhythm (...) directly accelerates the learning process for the new instrument.
This technique is particularly valuable when working with limited data for new tasks. Instead of training models from scratch, which requires enormous datasets and computational resources, transfer learning enables effective model development with smaller, more focused datasets.

Transfer learning provides four primary advantages that make it essential for sustainable AI development:
1. **Training efficiency** represents the most immediate benefit, as the technique eliminates the need to train models from scratch. This efficiency translates directly into reduced computational requirements and faster development cycles, enabling fine-tuning with significantly smaller datasets.
2. **Model performance** improvements occur because transfer learning leverages pre-trained knowledge that has already captured important patterns and features. This existing knowledge helps reduce overfitting tendencies while enabling faster convergence during training, even when working with limited new data. The pre-trained models provide a strong foundation that new models can build upon.
3. **Operational cost reduction** represents a crucial advantage in today's resource-constrained environment. Training large models from scratch requires enormous computational resources and extensive data collection efforts, both of which are expensive and time-consuming. Transfer learning significantly reduces these requirements, making advanced AI development more accessible and economically viable.
4. **Enhanced adaptability and reusability** make transfer learning particularly valuable for organizations needing to deploy AI across multiple scenarios. A single pre-trained model can serve as the foundation for numerous specialized applications, dramatically increasing the return on investment for model development efforts.

There are three concepts related to transfer learning:
1. **Multi-task learning** extends transfer learning principles by training single models to perform multiple related tasks simultaneously. These models feature shared early layers that process data in common ways, followed by task-specific layers that handle the unique requirements of each task. This architecture enables models to learn general features useful across all tasks while maintaining specialized capabilities for specific applications.
2. **Feature extraction** represents a focused application of transfer learning where pre-trained models extract meaningful features from new data. These extracted features then serve as input for specialized models designed for specific tasks. The process leverages neural networks' ability to automatically identify important data characteristics, applying only the pre-trained first and intermediate layers that contain generalizable knowledge while adapting the final layers for new tasks.
3. **Fine-tuning** goes beyond simple feature extraction by continuing the training process on domain-specific datasets. This technique proves particularly valuable when source and target tasks differ significantly, as it allows the model to adapt its learned representations to new domains. Most large language models today excel at general tasks but require fine-tuning to achieve optimal performance on specific applications.

There are two main applications for Transfer Learning:

**Computer Vision**
Computer vision represents one of the most successful application areas for transfer learning. Neural networks in this domain require vast amounts of data to perform tasks like object detection and image classification effectively. The hierarchical nature of visual processing makes transfer learning particularly effective, as initial layers detect basic features like edges, middle layers identify shapes and forms, and final layers handle task-specific classification.

**Natural language processing**
Natural language processing leverages transfer learning through pre-trained language models that understand general language patterns and structures. These models can then be fine-tuned for specific applications such as sentiment analysis, language translation, or document summarization. The versatility of transfer learning in NLP has enabled the development of sophisticated applications including voice assistants and speech recognition systems.

Transfer learning faces several significant challenges that practitioners must address carefully.
**Domain mismatch** occurs when source and target tasks differ substantially, limiting the effectiveness of knowledge transfer. The technique works best when tasks share underlying similarities, but performance degrades when domains are too dissimilar.
**Data scarcity** remains a concern even with transfer learning, as some minimum amount of target domain data is always required. Extremely limited or poor-quality training data can lead to underfitting, where models fail to capture important patterns in the new domain.

**Overfitting** represents another significant challenge, as excessive fine-tuning can cause models to learn task-specific features that don't generalize well to new data. This problem requires careful monitoring and regularization techniques to prevent models from becoming too specialized.

**Complexity** issues arise when target tasks are inherently difficult or when fine-tuning processes become challenging, costly, and time-consuming. These situations require careful planning and potentially different approaches to achieve acceptable results.

Transfer learning can be useful for green computing in several  ways:

1. **Resource Efficiency** - Reduced computational resources and energy consumption
2. **Faster Training Times** - Less energy through quicker development cycles
3. **Improved Performance with Less Data** - Reduces environmental impact of data collection
4. **Environmental Monitoring Applications** - Direct support for climate prediction and conservation