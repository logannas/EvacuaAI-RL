# EvacuAI-RL: Fire Escape Route Simulation

This project simulates escape routes in indoor environments under emergency conditions, such as fire outbreaks and congestion. The main goal is to model how an agent navigate towards exits while avoiding dynamic obstacles like fire or blocked paths.

The system is built in Python and includes functionalities for:

- Generating graphs that represent the building layout.
- Simulating fire outbreaks and congestion in real-time.
- Modeling and updating agent behavior using environment feedback.
- Supporting reinforcement learning algorithms to improve decision-making.

## Features

- 🔥 **Fire Simulation**: Random fire generation on potential exit paths, simulating risk zones.
- 🚧 **Congestion Modeling**: Nodes can become congested based on visit frequency or predefined rules.
- 🧠 **Agent Navigation**: Agents must find the safest and fastest path to the nearest exit, avoiding obstacles.
- 🗺️ **Graph Representation**: The environment is modeled as a graph where nodes represent locations and edges represent paths.

## Technologies

- Python 3.10+
- NetworkX
- NumPy
- FastAPI (optional for service interface)
- Docker

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the project:

```bash
python -m src.main
```

## Docker

You can also run the project using Docker:

1. Build the image:

```bash
docker build -t evacuai-rl .
```

2. Run the container:

```bash
docker run -p 50051:50051 --network host -it --rm evacuai-rl
```
