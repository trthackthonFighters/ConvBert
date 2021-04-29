import onnx_graphsurgeon as gs
import onnx
import numpy as np

graph = gs.import_onnx(onnx.load("./ConvBert.onnx"))


for node in graph.nodes:
    if node.op == "OneHot":
        depth = node.inputs[1].values
        attrs = {"depth": int(depth)}
        onehot = gs.Node(op="OnehotPlugin", name = node.name, attrs = attrs)
        graph.nodes.append(onehot)

        inp_output_tensor = node.inputs[0]
        inp_output_tensor.outputs = [onehot]
        onehot.outputs = node.outputs
        node.outputs.clear()
        print(onehot)

# Remove the non-used node from the graph completely
graph.cleanup()

onnx.save(gs.export_onnx(graph), "./ConvBert_onehot.onnx")

