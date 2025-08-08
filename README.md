# CV Project Portfolio

End-to-end computer vision pipeline to help students navigate in the Stata building at MIT.
The goal is to classify scenes from video frames, generate semantic embeddings, and optimize navigation paths between points of interest using both learned models and graph-based routing.

This project is a computer vision pipeline that processes video frames, builds a CNN model using TensorFlow/Keras, and performs classification on extracted images. It also integrates OpenAI's CLIP foundation model for semantic embeddings and includes a lightweight navigation module for route optimization and visualization

## Structure

- `src/`: Contains all the source code.
- `data/`: Contains raw data (videos/images).
- `saved_models/`: Stores trained models.
- `notebooks/`: Optional Jupyter notebooks for experimentation.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run preprocessing:
```bash
python src/preprocess.py
```

Train model:
```bash
python src/train.py
```

Predict on new images:
```bash
python src/predict.py
```


## Additional Features

### Foundation Model Embeddings (CLIP)
This project uses the OpenAI CLIP model to extract image embeddings. The module is located in `src/foundation.py`.

Usage:
```python
from src.foundation import get_clip_embedding
embedding = get_clip_embedding("path_to_image.jpg")
```

### Route Optimization Tool
A basic graph navigation tool is included using NetworkX, located in `src/navigation.py`. It computes shortest paths and visualizes the graph.

Usage:
```python
from src.navigation import build_route_graph, shortest_path, plot_graph
```

These functionalities are also demonstrated in `main.py`.

