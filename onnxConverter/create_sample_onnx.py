#!/usr/bin/env python3
"""
Create a simple ONNX model for demonstration purposes
This creates a minimal CNN model that can be used to test the converter
"""

import os
import numpy as np
import onnx
from onnx import helper, TensorProto

def create_simple_cnn_onnx():
    """Create a simple CNN ONNX model manually"""
    
    # Define the model name
    model_name = "SimpleCNN"
    
    # Define input
    input_tensor = helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, [1, 3, 32, 32]
    )
    
    # Define output  
    output_tensor = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, [1, 10]
    )
    
    # Create weights as constants
    conv1_weight = helper.make_tensor(
        'conv1_weight', TensorProto.FLOAT,
        [16, 3, 3, 3],  # [out_channels, in_channels, kernel_h, kernel_w]
        np.random.randn(16, 3, 3, 3).astype(np.float32).tobytes(),
        raw=True
    )
    
    conv2_weight = helper.make_tensor(
        'conv2_weight', TensorProto.FLOAT,
        [32, 16, 3, 3],
        np.random.randn(32, 16, 3, 3).astype(np.float32).tobytes(),
        raw=True
    )
    
    fc_weight = helper.make_tensor(
        'fc_weight', TensorProto.FLOAT,
        [32, 10],  # Flattened conv output to 10 classes
        np.random.randn(32, 10).astype(np.float32).tobytes(),
        raw=True
    )
    
    # Define nodes
    nodes = [
        # Conv1: input -> conv1_out
        helper.make_node(
            'Conv',
            inputs=['input', 'conv1_weight'],
            outputs=['conv1_out'],
            name='conv1',
            kernel_shape=[3, 3],
            strides=[1, 1],
            pads=[1, 1, 1, 1]
        ),
        
        # ReLU1: conv1_out -> relu1_out
        helper.make_node(
            'Relu',
            inputs=['conv1_out'],
            outputs=['relu1_out'],
            name='relu1'
        ),
        
        # MaxPool1: relu1_out -> pool1_out
        helper.make_node(
            'MaxPool',
            inputs=['relu1_out'],
            outputs=['pool1_out'],
            name='pool1',
            kernel_shape=[2, 2],
            strides=[2, 2]
        ),
        
        # Conv2: pool1_out -> conv2_out
        helper.make_node(
            'Conv',
            inputs=['pool1_out', 'conv2_weight'],
            outputs=['conv2_out'],
            name='conv2',
            kernel_shape=[3, 3],
            strides=[1, 1],
            pads=[1, 1, 1, 1]
        ),
        
        # ReLU2: conv2_out -> relu2_out
        helper.make_node(
            'Relu',
            inputs=['conv2_out'],
            outputs=['relu2_out'],
            name='relu2'
        ),
        
        # GlobalAveragePool: relu2_out -> gap_out
        helper.make_node(
            'GlobalAveragePool',
            inputs=['relu2_out'],
            outputs=['gap_out'],
            name='global_avg_pool'
        ),
        
        # Flatten: gap_out -> flatten_out
        helper.make_node(
            'Flatten',
            inputs=['gap_out'],
            outputs=['flatten_out'],
            name='flatten',
            axis=1
        ),
        
        # MatMul: flatten_out, fc_weight -> output
        helper.make_node(
            'MatMul',
            inputs=['flatten_out', 'fc_weight'],
            outputs=['output'],
            name='fc'
        )
    ]
    
    # Create the graph
    graph = helper.make_graph(
        nodes,
        model_name,
        [input_tensor],
        [output_tensor],
        initializer=[conv1_weight, conv2_weight, fc_weight]
    )
    
    # Create the model
    model = helper.make_model(graph, producer_name='onnx-example')
    model.opset_import[0].version = 11
    
    # Check model
    onnx.checker.check_model(model)
    
    return model

def main():
    """Create sample ONNX models"""
    print("🔨 Creating sample ONNX models...")
    
    # Create simple CNN
    print("Creating SimpleCNN.onnx...")
    simple_cnn = create_simple_cnn_onnx()
    os.makedirs('examples/models', exist_ok=True)
    onnx.save(simple_cnn, 'examples/models/SimpleCNN.onnx')
    print("✅ Created examples/models/SimpleCNN.onnx")
    
    print("\n📋 Model Information:")
    print(f"   Input: {simple_cnn.graph.input[0].name} {[dim.dim_value for dim in simple_cnn.graph.input[0].type.tensor_type.shape.dim]}")
    print(f"   Output: {simple_cnn.graph.output[0].name} {[dim.dim_value for dim in simple_cnn.graph.output[0].type.tensor_type.shape.dim]}")
    print(f"   Operations: {len(simple_cnn.graph.node)}")
    
    for i, node in enumerate(simple_cnn.graph.node):
        print(f"     {i+1}. {node.op_type} ({node.name})")
    
    print(f"\n🎯 Ready to convert! Run:")
    print(f"   python onnx_to_polaris.py examples/models/SimpleCNN.onnx")
    
    return 0

if __name__ == '__main__':
    exit(main())
