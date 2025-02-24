import onnx
import numpy as np
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper
from onnx import TensorProto
from pathlib import Path

# Define the 3x3 convolution kernel as an ONNX initializer (constant tensor)
kernel_values = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32).reshape(1, 1, 3, 3)
kernel = numpy_helper.from_array(kernel_values, name="conv_kernel")

# Input tensor (dynamic width and height)
input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, "Height", "Width"])
output_tensor = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 1, "Height", "Width"])

# Convolution operation to count neighbors
conv_node = helper.make_node(
    "Conv",
    inputs=["input", "conv_kernel"],
    outputs=["neighbors"],
    kernel_shape=[3, 3],
    pads=[1, 1, 1, 1],  # Ensures edges are handled correctly
    group=1,
)

# Rules of Conway's Game of Life (elementwise operations)
alive = helper.make_node("Equal", inputs=["input", "one_const"], outputs=["alive"])
three_neighbors = helper.make_node("Equal", inputs=["neighbors", "three_const"], outputs=["rule1"])
two_neighbors = helper.make_node("Equal", inputs=["neighbors", "two_const"], outputs=["rule2"])

# Stay alive: alive & (neighbors == 2)
stay_alive = helper.make_node("And", inputs=["alive", "rule2"], outputs=["stay_alive"])

# Birth: (neighbors == 3)
birth = helper.make_node("Or", inputs=["stay_alive", "rule1"], outputs=["birth"])

# Cast the output to bool
cast_node = helper.make_node("Cast", inputs=["birth"], outputs=["output"], to=TensorProto.BOOL)

one_const = numpy_helper.from_array(np.array([1], dtype=np.float32), name="one_const")
two_const = numpy_helper.from_array(np.array([2], dtype=np.float32), name="two_const")
three_const = numpy_helper.from_array(np.array([3], dtype=np.float32), name="three_const")

# Create the graph
graph = helper.make_graph(
    [conv_node, alive, three_neighbors, two_neighbors, stay_alive, birth, cast_node],
    "conway_game_of_life",
    [input_tensor],
    [output_tensor],
    [kernel, one_const, two_const, three_const],
)

# Create the ONNX model
model = helper.make_model(graph, producer_name="conway_onnx", opset_imports=[helper.make_opsetid("", 21)])

Path("./web/public").mkdir(parents=True, exist_ok=True)
onnx.save(model, "web/public/conway_game_of_life.onnx")
print("ONNX model saved!")