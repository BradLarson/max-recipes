"""Simple MAX Graph example that adds two vectors."""

import numpy as np
from max import engine
from max.driver import Tensor
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops


def main():
    vector1_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    vector2_data = np.array([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)
    
    input_type = TensorType(
        dtype=DType.float32,
        shape=(8,),
        device=DeviceRef.CPU()
    )
    
    # We'll just do simple one-operation vector addition for our graph
    with Graph("vector_add", input_types=(input_type, input_type)) as graph:
        vector1, vector2 = graph.inputs
        
        output = vector1 + vector2  # Same as ops.add()
        
        graph.output(output)
        
    # Cmopile the graph
    session = engine.InferenceSession()
    model = session.load(graph)

    # Save the graph to a MEF file
    model._export_mef("graph.mef")
    
    # Run the graph, just to verify it works
    results = model.execute(vector1_data, vector2_data)    
    result_array = results[0].to_numpy()
    
    print(f"Vector 1: {vector1_data}")
    print(f"Vector 2: {vector2_data}")
    print(f"Result:   {result_array}")
    print(f"\nExpected: {vector1_data + vector2_data}")
    
    assert np.allclose(result_array, vector1_data + vector2_data), "Addition result is incorrect!"
    print("\n✓ Vector addition successful!")


if __name__ == "__main__":
    main()