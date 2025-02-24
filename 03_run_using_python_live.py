import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# Grid size
GRID_SIZE = 300
NUM_STEPS = 50

# Load ONNX model
session = ort.InferenceSession("conway_game_of_life.onnx")

# Create an initial random binary grid
grid = (np.random.rand(1, 1, GRID_SIZE, GRID_SIZE) > 0.5).astype(np.float32)

# Set up the figure for animation
fig, ax = plt.figure(figsize=(6, 6)), plt.axes()
ax.set_axis_off()
img = ax.imshow(grid[0, 0], cmap="gray", interpolation="nearest")

# Create a list to store grid states
grid_history = [grid.copy()]

# Compute all steps first
print("Computing all steps...")
for step in range(NUM_STEPS - 1):
    grid = session.run(None, {"input": grid})[0].astype(np.float32)
    grid_history.append(grid.copy())
    print(f"Step {step + 1}/{NUM_STEPS - 1} computed.")

# Animation update function
def update(frame):
    img.set_array(grid_history[frame][0, 0])
    return [img]

# Create animation
print("Creating GIF animation...")
animation = FuncAnimation(fig, update, frames=NUM_STEPS, interval=200, blit=True)

# Create output directory if it doesn't exist
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save as GIF
animation.save(f"{OUTPUT_DIR}/conway_game_of_life.gif", writer='pillow', dpi=100)

print(f"GIF animation saved as '{OUTPUT_DIR}/conway_game_of_life.gif'!")