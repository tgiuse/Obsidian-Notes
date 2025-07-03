
1. $$10kW \times 500s \times 1000 = 5, 000, 000 J$$
2. 

3. Without this natural greenhouse effect, Earth would be an icy, inhospitable planet. The greenhouse effect raises the Earth's average temperature by about 32 K, creating a stable, life-supporting climate.
4. In blockchains there are a couple of options to validate a transaction, those being: Proof of work and Proof of Stake.
   **Proof of work**
   This is  consensus algorithm used to: validate new transactions, add them to the blockchain and prevent malicious actors from tampering with the chain. In simple words it is a computational puzzle that must be solved before a clock is accepted by the network. Before a new block can be added to the blockchain, miners (nodes that do the computation) must: compete to solve a difficult mathematical problem. This problem involves finding a number (called a nonce) that, when combined with the block's data and passed through a cryptographic hash function, results in a hash that meets specific conditions (e.g., starts with a certain number of zeros) → “mining”.
   **Proof of stake**
   Proof of Stake is a consensus mechanism used in blockchain systems as an alternative to Proof of Work. Instead of using computational power to validate transactions and create new blocks, PoS uses ownership (stake). Validators are chosen to create new blocks and the chance of being selected depends on the “stake”. Validators are rewarded with transaction fees.
5. .
6. Federated Learning (FL) represents a paradigm shift in distributed learning. It's a distributed learning approach where a central server coordinates the collaborative training of a global model across numerous clients.
   This approach can reduce the carbon footprint of computation by minimizing data transfer, particularly from edge devices to centralized servers, and potentially reducing the need for energy-intensive data centers. 
7. To reduce the environmental footprint of AI systems themselves, several optimization strategies are crucial 
   
   - Algorithmic Optimization: 
     Making algorithms more efficient has significant benefits, including reducing their environmental footprint. A productive strategy is designing optimization techniques that reduce computational resource requirements, thereby minimizing energy consumption.
   - Hardware Optimization:
	Choosing more computationally efficient hardware directly contributes to energy savings.
	Some graphic processing units (GPUs) offer substantially higher efficiency in terms of floating-point
	operations per second (FLOPS) per watt of power usage compared to others.
	Specialized hardware accelerators, such as tensor processing units (TPUs), are tailored specifically for
	ML tasks and can be customized for specific ML models.
	Edge computing is a key strategy where computation is performed at the locations where data is collected
	or used. This avoids the need to transmit data to a data center or the cloud, and adapts to the limited
	computational and energy resources of IoT devices
   - Data Center Optimization:
	The carbon footprint of a data center is directly proportional to its efficiency and the carbon intensity of its
	location.
	The carbon intensity of the location is a critical factor, given the significant variability between countries
	(e.g., less than 20 gCO2e kWh⁻¹ in Norway and Switzerland vs. over 800 gCO2e kWh⁻¹ in Australia, South
	Africa, and some US states).
	To optimize data center use, researchers have developed algorithms and frameworks that dynamically
	manage server loads, adjust cooling systems, and optimize resource allocation.

 9. Transfer Learning (TL) is a machine learning methodology focused on transferring knowledge across different domains. Its primary goal is to leverage knowledge gained from a related "source domain" to improve learning performance or minimize the number of labeled examples needed in a "target domain". This approach is particularly valuable when collecting sufficient labeled training data for a new task is expensive, time-consuming, or unrealistic.TL contributes to "Green AI" through several mechanisms:
	● Less Training = Less Energy: It bypasses the need to train large models (like GPT, BERT, or large
	CNNs) from scratch, which consume massive amounts of compute resources and electricity.
	● Reuse of Resources: By reusing pre-trained models, TL maximizes the utility of previously
	expended computational effort.
	● Faster Development: TL accelerates research and development cycles, reducing the time and
	computation spent on iterative processes, thereby saving energy indirectly
 10. Traditional AI architectures largely rely on the Von Neumann model, which means the system processes tasks serially and is governed by a clock that synchronizes operations. This design, while powerful for general AI tasks, often consumes a significant amount of energy because data has to constantly shuttle back and forth between the processor and memory. This constant movement of data combined with sequential processing leads to high energy consumption, which is a major concern in green computing where reducing carbon footprint and electricity usage is a priority. In contrast, neuromorphic architectures take inspiration from how the human brain operates. Instead of sequential processing, they use a parallel and event-driven approach. This means computations happen simultaneously across many units (like neurons), and activity is often triggered only by specific events, not continuously clocked. Because of this event-driven, parallel nature, neuromorphic systems can perform complex sensory and real-time tasks with much lower energy requirements. This reduction in energy use translates directly to greener technology, as these systems can maintain performance while minimizing environmental impact.