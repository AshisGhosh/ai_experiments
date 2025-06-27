import einops
import matplotlib.pyplot as plt
import streamlit as st

from generative_models.datasets import CircleDataset, OrderedCircleDataset
from utils import data_dir

st.title("Dataset Visualization")

# dataset = CircleDataset(data_dir("circle_dataset.pkl"))
dataset = CircleDataset(data_dir("x_steps.pkl"))

# dataset = OrderedCircleDataset(data_dir("circle_dataset.pkl"))

idx = st.slider("Select circle", 0, max(1, len(dataset) - 1), 0)
idx = min(idx, len(dataset) - 1)

points = dataset[idx]
points = einops.rearrange(points, "n d -> d n")
plt.figure(figsize=(8, 8))
plt.scatter(points[0], points[1], alpha=0.6)
plt.grid(True)
plt.axis("equal")
st.pyplot(plt.gcf())
plt.close()

st.table(dataset.data[idx])
