#!/usr/bin/env python3
"""
Smart Workload Builder: ONNX to Polaris Code Generator

This tool automatically generates Polaris-compatible workload Python code from ONNX model files.
It bridges the gap between ONNX models and Polaris TTSIM workloads for rapid prototyping.

Usage:
    python onnx_to_polaris.py input_model.onnx --output workloads/GeneratedModel.py
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import re

# ONNX imports
try:
    import onnx
    from onnx import shape_inference, TensorProto
    HAS_ONNX = True
except ImportError:
    print("⚠️  ONNX not available. Install with: pip install onnx")
    HAS_ONNX = False
    sys.exit(1)

import numpy as np


class ONNXToPolaris:
    """Main class for converting ONNX models to Polaris workloads"""
    
    def __init__(self):
        self.onnx_to_polaris_ops = {
            # Basic operations
            'Conv': self._map_conv,
            'Relu': self._map_relu,
            'MaxPool': self._map_maxpool,
            'AveragePool': self._map_avgpool,
            'BatchNormalization': self._map_batchnorm,
            'Add': self._map_add,
            'Mul': self._map_mul,
            'Reshape': self._map_reshape,
            'Transpose': self._map_transpose,
            'Flatten': self._map_flatten,
            'Dropout': self._map_dropout,
            'Softmax': self._map_softmax,
            'Sigmoid': self._map_sigmoid,
            'Tanh': self._map_tanh,
            
            # Matrix operations
            'MatMul': self._map_matmul,
            'Gemm': self._map_gemm,
            
            # Pooling
            'GlobalAveragePool': self._map_global_avgpool,
            'AdaptiveAvgPool2d': self._map_adaptive_avgpool,
            
            # Normalization
            'LayerNormalization': self._map_layernorm,
            
            # Activation functions
            'Gelu': self._map_gelu,
            'LeakyRelu': self._map_leakyrelu,
            
            # Shape operations
            'Concat': self._map_concat,
            'Split': self._map_split,
            'Squeeze': self._map_squeeze,
            'Unsqueeze': self._map_unsqueeze,
        }
        
        self.unsupported_ops = set()
        self.model_info = {}
        self.input_specs = {}
        self.output_specs = {}
        
    def convert_onnx_to_polaris(self, onnx_path: str, output_path: str, 
                               class_name: Optional[str] = None,
                               config: Optional[Dict] = None) -> str:
        """
        Main conversion function
        
        Args:
            onnx_path: Path to input ONNX model
            output_path: Path for output Python file
            class_name: Optional custom class name
            config: Optional configuration parameters
            
        Returns:
            Generated Python code as string
        """
        print(f"🔄 Converting ONNX model: {onnx_path}")
        
        # Load and parse ONNX model
        model = self._load_onnx_model(onnx_path)
        graph_info = self._parse_onnx_graph(model)
        
        # Generate class name if not provided
        if class_name is None:
            model_name = Path(onnx_path).stem
            class_name = self._sanitize_class_name(model_name)
        
        # Generate Python code
        python_code = self._generate_python_code(class_name, graph_info, config or {})
        
        # Write to file
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(python_code)
            
        print(f"✅ Generated Polaris workload: {output_path}")
        print(f"📝 Class name: {class_name}")
        
        if self.unsupported_ops:
            print(f"⚠️  Unsupported operations found: {', '.join(self.unsupported_ops)}")
            print("   These will need manual implementation")
            
        return python_code
    
    def _load_onnx_model(self, onnx_path: str) -> onnx.ModelProto:
        """Load and validate ONNX model"""
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
            
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        
        # Perform shape inference
        try:
            model = shape_inference.infer_shapes(model)
        except Exception as e:
            print(f"⚠️  Shape inference failed: {e}")
            print("   Proceeding without shape inference...")
            
        return model
    
    def _parse_onnx_graph(self, model: onnx.ModelProto) -> Dict[str, Any]:
        """Parse ONNX graph and extract relevant information"""
        graph = model.graph
        
        # Extract inputs
        inputs = {}
        for input_info in graph.input:
            # Skip initializer inputs (weights)
            if input_info.name not in [init.name for init in graph.initializer]:
                inputs[input_info.name] = self._parse_value_info(input_info)
        
        # Extract outputs
        outputs = {}
        for output_info in graph.output:
            outputs[output_info.name] = self._parse_value_info(output_info)
        
        # Extract initializers (weights/constants)
        initializers = {}
        for init in graph.initializer:
            initializers[init.name] = self._parse_tensor(init)
        
        # Extract operations
        operations = []
        for node in graph.node:
            op_info = {
                'name': node.name or f"{node.op_type}_{len(operations)}",
                'op_type': node.op_type,
                'inputs': list(node.input),
                'outputs': list(node.output),
                'attributes': {attr.name: self._parse_attribute(attr) for attr in node.attribute}
            }
            operations.append(op_info)
        
        self.input_specs = inputs
        self.output_specs = outputs
        
        return {
            'inputs': inputs,
            'outputs': outputs,
            'initializers': initializers,
            'operations': operations,
            'name': graph.name or "GeneratedModel"
        }
    
    def _parse_value_info(self, value_info) -> Dict[str, Any]:
        """Parse ONNX ValueInfo (input/output specification)"""
        result = {'name': value_info.name}
        
        if value_info.type.HasField('tensor_type'):
            tensor_type = value_info.type.tensor_type
            
            # Extract data type
            elem_type = tensor_type.elem_type
            dtype_map = {
                TensorProto.FLOAT: 'float32',
                TensorProto.DOUBLE: 'float64',
                TensorProto.INT32: 'int32',
                TensorProto.INT64: 'int64',
                TensorProto.BOOL: 'bool',
                TensorProto.FLOAT16: 'float16'
            }
            result['dtype'] = dtype_map.get(elem_type, 'float32')
            
            # Extract shape
            shape = []
            for dim in tensor_type.shape.dim:
                if dim.HasField('dim_value'):
                    shape.append(dim.dim_value)
                elif dim.HasField('dim_param'):
                    shape.append(dim.dim_param)  # Dynamic dimension
                else:
                    shape.append(-1)  # Unknown dimension
            result['shape'] = shape
            
        return result
    
    def _parse_tensor(self, tensor) -> Dict[str, Any]:
        """Parse ONNX tensor (initializer/constant)"""
        result = {
            'name': tensor.name,
            'shape': list(tensor.dims),
            'dtype': tensor.data_type
        }
        
        # Extract data if needed (for small constants)
        if len(tensor.dims) == 0 or np.prod(tensor.dims) < 10:
            try:
                numpy_tensor = onnx.numpy_helper.to_array(tensor)
                result['data'] = numpy_tensor
            except:
                pass
                
        return result
    
    def _parse_attribute(self, attr) -> Any:
        """Parse ONNX node attribute"""
        if attr.type == onnx.AttributeProto.INT:
            return attr.i
        elif attr.type == onnx.AttributeProto.FLOAT:
            return attr.f
        elif attr.type == onnx.AttributeProto.STRING:
            return attr.s.decode('utf-8')
        elif attr.type == onnx.AttributeProto.INTS:
            return list(attr.ints)
        elif attr.type == onnx.AttributeProto.FLOATS:
            return list(attr.floats)
        elif attr.type == onnx.AttributeProto.STRINGS:
            return [s.decode('utf-8') for s in attr.strings]
        else:
            return str(attr)
    
    def _sanitize_class_name(self, name: str) -> str:
        """Convert filename to valid Python class name"""
        # Remove file extension and special characters
        name = re.sub(r'[^a-zA-Z0-9_]', '', name)
        
        # Ensure it starts with letter
        if name and name[0].isdigit():
            name = f"Model_{name}"
        
        # Capitalize first letter
        name = name.capitalize()
        
        return name or "GeneratedModel"
    
    def _generate_python_code(self, class_name: str, graph_info: Dict, config: Dict) -> str:
        """Generate complete Python workload code"""
        
        # Generate imports
        imports = self._generate_imports()
        
        # Generate class definition
        class_def = self._generate_class_definition(class_name, graph_info, config)
        
        # Generate standalone runner (optional)
        standalone = self._generate_standalone_runner(class_name)
        
        return f"{imports}\n\n{class_def}\n\n{standalone}"
    
    def _generate_imports(self) -> str:
        """Generate required imports"""
        return '''#!/usr/bin/env python
# SPDX-FileCopyrightText: Generated by ONNX to Polaris Converter
# SPDX-License-Identifier: Apache-2.0
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import ttsim.front.functional.op as F
import ttsim.front.functional.tensor_op as T
import ttsim.front.functional.sim_nn as SimNN
from ttsim.ops import SimTensor
import numpy as np
import logging'''
    
    def _generate_class_definition(self, class_name: str, graph_info: Dict, config: Dict) -> str:
        """Generate main class definition"""
        
        # Generate __init__ method
        init_method = self._generate_init_method(class_name, graph_info, config)
        
        # Generate create_input_tensors method
        input_method = self._generate_input_tensors_method(graph_info)
        
        # Generate forward pass (__call__ method)
        forward_method = self._generate_forward_method(graph_info)
        
        # Generate helper methods
        helper_methods = self._generate_helper_methods()
        
        return f'''class {class_name}(SimNN.Module):
    """Generated from ONNX model"""
    
{init_method}

{input_method}

{forward_method}

{helper_methods}'''
    
    def _generate_init_method(self, class_name: str, graph_info: Dict, config: Dict) -> str:
        """Generate __init__ method"""
        
        # Extract configuration parameters
        config_params = []
        for input_name, input_info in graph_info['inputs'].items():
            shape = input_info.get('shape', [])
            if len(shape) >= 2:
                if len(shape) == 4:  # Image input [batch, channels, height, width]
                    config_params.extend([
                        f"self.bs = cfg.get('bs', {shape[0] if isinstance(shape[0], int) else 1})",
                        f"self.num_channels = cfg.get('num_channels', {shape[1] if isinstance(shape[1], int) else 3})",
                        f"self.img_height = cfg.get('img_height', {shape[2] if isinstance(shape[2], int) else 224})",
                        f"self.img_width = cfg.get('img_width', {shape[3] if isinstance(shape[3], int) else 224})"
                    ])
                elif len(shape) == 3:  # Sequence input [batch, seq_len, features]
                    config_params.extend([
                        f"self.bs = cfg.get('bs', {shape[0] if isinstance(shape[0], int) else 1})",
                        f"self.seq_len = cfg.get('seq_len', {shape[1] if isinstance(shape[1], int) else 512})",
                        f"self.num_features = cfg.get('num_features', {shape[2] if isinstance(shape[2], int) else 768})"
                    ])
        
        if not config_params:
            config_params = ["self.bs = cfg.get('bs', 1)"]
        
        # Generate operations initialization
        operations = []
        op_counter = {}
        
        for op_info in graph_info['operations']:
            op_type = op_info['op_type']
            if op_type in self.onnx_to_polaris_ops:
                try:
                    polaris_op = self.onnx_to_polaris_ops[op_type](op_info, graph_info)
                    if polaris_op:
                        operations.append(polaris_op)
                except Exception as e:
                    print(f"⚠️  Error mapping {op_type}: {e}")
                    self.unsupported_ops.add(op_type)
            else:
                self.unsupported_ops.add(op_type)
        
        # Generate weight tensor initializations for initializers used in forward pass
        weight_tensors = []
        for op_info in graph_info['operations']:
            for input_name in op_info.get('inputs', []):
                if input_name in graph_info.get('initializers', {}) and 'weight' in input_name.lower():
                    initializer = graph_info['initializers'][input_name]
                    shape = initializer.get('shape', [])
                    shape_str = str(shape) if shape else "[1]"
                    weight_tensors.append(f"self.{input_name.lower()} = F._from_shape('{input_name.lower()}', {shape_str}, is_param=True)")
        
        # Remove duplicates
        weight_tensors = list(set(weight_tensors))
        
        config_str = '\n        '.join(config_params)
        ops_str = '\n        '.join(operations) if operations else "# Operations will be added here"
        weights_str = '\n        '.join(weight_tensors) if weight_tensors else ""
        
        weights_section = f"""
        # Weight tensors
        {weights_str}""" if weights_str else ""
        
        return f'''    def __init__(self, name, cfg):
        super().__init__()
        self.name = name
        
        # Configuration parameters
        {config_str}
        
        # Operations
        {ops_str}{weights_section}
        
        super().link_op2module()'''
    
    def _generate_input_tensors_method(self, graph_info: Dict) -> str:
        """Generate create_input_tensors method"""
        
        input_tensors = []
        for input_name, input_info in graph_info['inputs'].items():
            shape = input_info.get('shape', [1])
            dtype = input_info.get('dtype', 'float32')
            
            # Convert dynamic dimensions to configuration parameters
            shape_str = []
            for dim in shape:
                if isinstance(dim, int):
                    shape_str.append(str(dim))
                elif isinstance(dim, str):
                    if 'batch' in dim.lower():
                        shape_str.append('self.bs')
                    else:
                        shape_str.append('1')  # Default for unknown dynamic dims
                else:
                    shape_str.append('1')
            
            shape_expr = '[' + ', '.join(shape_str) + ']'
            
            np_dtype = f"np.{dtype}"
            input_tensors.append(
                f"'{input_name}': F._from_shape('{input_name}', {shape_expr}, is_param=False, np_dtype={np_dtype})"
            )
        
        tensors_str = ',\n            '.join(input_tensors)
        
        return f'''    def create_input_tensors(self):
        self.input_tensors = {{
            {tensors_str}
        }}
        return'''
    
    def _generate_forward_method(self, graph_info: Dict) -> str:
        """Generate forward pass (__call__ method)"""
        
        # Start with input tensors
        forward_lines = []
        
        # Map input names to variable names
        var_map = {}
        for input_name in graph_info['inputs'].keys():
            var_name = f"x_{len(var_map)}" if len(var_map) > 0 else "x"
            var_map[input_name] = var_name
            forward_lines.append(f"{var_name} = self.input_tensors['{input_name}']")
        
        # Process operations in order
        for i, op_info in enumerate(graph_info['operations']):
            try:
                forward_code = self._generate_forward_operation(op_info, var_map, i)
                if forward_code:
                    forward_lines.extend(forward_code)
            except Exception as e:
                forward_lines.append(f"# Error processing {op_info['op_type']}: {e}")
        
        # Return output
        output_vars = []
        for output_name in graph_info['outputs'].keys():
            if output_name in var_map:
                output_vars.append(var_map[output_name])
        
        if len(output_vars) == 1:
            return_stmt = f"return {output_vars[0]}"
        elif len(output_vars) > 1:
            return_stmt = f"return {', '.join(output_vars)}"
        else:
            return_stmt = "return x  # Adjust as needed"
        
        forward_str = '\n        '.join(forward_lines)
        
        return f'''    def __call__(self):
        {forward_str}
        {return_stmt}'''
    
    def _generate_helper_methods(self) -> str:
        """Generate helper methods"""
        return '''    def get_forward_graph(self):
        return super()._get_forward_graph(self.input_tensors)
    
    def analytical_param_count(self):
        return 0  # TODO: Implement parameter counting'''
    
    def _generate_standalone_runner(self, class_name: str) -> str:
        """Generate standalone test runner"""
        return f'''def run_standalone(outdir: str = '.') -> None:
    """Standalone test runner"""
    print(f'Creating {class_name}....')
    
    # Default configuration
    default_cfg = {{'bs': 1}}
    
    model = {class_name}('test_model', default_cfg)
    model.create_input_tensors()
    
    print('Input tensors:')
    for name, tensor in model.input_tensors.items():
        print(f'  {{name}}: {{tensor.shape}}')
    
    try:
        output = model()
        print(f'Output shape: {{output.shape if hasattr(output, "shape") else type(output)}}')
        
        # Generate computation graph
        gg = model.get_forward_graph()
        onnx_path = f'{{outdir}}/{class_name.lower()}.onnx'
        print(f'Exporting to ONNX: {{onnx_path}}')
        gg.graph2onnx(onnx_path, do_model_check=True)
        
    except Exception as e:
        print(f'Error during execution: {{e}}')


if __name__ == '__main__':
    logging_format = "%(levelname)s:%(name)s:%(filename)s:%(lineno)d:%(message)s"
    logging.basicConfig(level=logging.WARNING, format=logging_format)
    run_standalone()'''

    # Operation mapping methods
    def _generate_forward_operation(self, op_info: Dict, var_map: Dict, op_index: int) -> List[str]:
        """Generate forward pass code for a single operation"""
        op_type = op_info['op_type']
        
        if op_type not in self.onnx_to_polaris_ops:
            return [f"# Unsupported operation: {op_type}"]
        
        # Get input variables
        input_vars = []
        for input_name in op_info['inputs']:
            if input_name in var_map:
                input_vars.append(var_map[input_name])
            else:
                # This might be a constant/initializer - create a placeholder
                if 'weight' in input_name.lower():
                    input_vars.append(f"self.{input_name.lower()}")
                else:
                    input_vars.append(f"# Missing input: {input_name}")
        
        # Generate output variable names
        output_vars = []
        for output_name in op_info['outputs']:
            if output_name not in var_map:
                var_map[output_name] = f"out_{op_index}_{len(output_vars)}"
            output_vars.append(var_map[output_name])
        
        # Generate operation-specific code
        try:
            return self._generate_operation_forward_code(op_info, input_vars, output_vars)
        except Exception as e:
            return [f"# Error in {op_type}: {e}"]
    
    def _generate_operation_forward_code(self, op_info: Dict, input_vars: List[str], output_vars: List[str]) -> List[str]:
        """Generate forward code for specific operation types"""
        op_type = op_info['op_type']
        attrs = op_info['attributes']
        
        if op_type == 'Conv':
            return [f"{output_vars[0]} = self.{op_info['name'].lower()}({input_vars[0]})"]
        elif op_type == 'Relu':
            return [f"{output_vars[0]} = self.{op_info['name'].lower()}({input_vars[0]})"]
        elif op_type == 'MaxPool':
            return [f"{output_vars[0]} = self.{op_info['name'].lower()}({input_vars[0]})"]
        elif op_type == 'GlobalAveragePool':
            return [f"{output_vars[0]} = self.{op_info['name'].lower()}({input_vars[0]})"]
        elif op_type == 'Flatten':
            return [f"{output_vars[0]} = {input_vars[0]}.reshape({input_vars[0]}.shape[0], -1)"]
        elif op_type == 'Add':
            if len(input_vars) >= 2:
                return [f"{output_vars[0]} = {input_vars[0]} + {input_vars[1]}"]
        elif op_type == 'Reshape':
            shape = attrs.get('shape', [])
            if shape:
                return [f"{output_vars[0]} = {input_vars[0]}.reshape({shape})"]
            else:
                return [f"{output_vars[0]} = {input_vars[0]}.reshape(-1)  # TODO: Specify shape"]
        elif op_type == 'MatMul':
            if len(input_vars) >= 2:
                return [f"{output_vars[0]} = T.matmul({input_vars[0]}, {input_vars[1]})"]
            else:
                return [f"# TODO: MatMul needs weight tensor - {input_vars}"]
        elif op_type == 'Softmax':
            axis = attrs.get('axis', -1)
            return [f"{output_vars[0]} = self.{op_info['name'].lower()}({input_vars[0]})"]
        
        return [f"# TODO: Implement {op_type} forward pass"]

    # Operation mapping methods (implementations for _map_* methods)
    def _map_conv(self, op_info: Dict, graph_info: Dict) -> str:
        """Map ONNX Conv to Polaris Conv2d"""
        attrs = op_info['attributes']
        
        # Extract attributes
        kernel_shape = attrs.get('kernel_shape', [3, 3])
        strides = attrs.get('strides', [1, 1])
        pads = attrs.get('pads', [0, 0, 0, 0])
        
        # For now, assume square kernels and uniform padding
        kernel_size = kernel_shape[0] if kernel_shape else 3
        stride = strides[0] if strides else 1
        padding = pads[0] if pads else 0
        
        # Extract in_channels and out_channels from weight tensor shape
        # Conv weight tensor format is usually [out_channels, in_channels, height, width]
        in_channels = "self.num_channels"  # Default fallback
        out_channels = 64  # Default fallback
        
        # Try to get actual dimensions from inputs
        if len(op_info.get('inputs', [])) > 1:
            weight_name = op_info['inputs'][1]  # Second input is usually the weight
            if weight_name in graph_info.get('initializers', {}):
                weight_tensor = graph_info['initializers'][weight_name]
                if hasattr(weight_tensor, 'dims') and len(weight_tensor.dims) >= 2:
                    out_channels = weight_tensor.dims[0]
                    in_channels = weight_tensor.dims[1]
                elif 'shape' in weight_tensor and len(weight_tensor['shape']) >= 2:
                    out_channels = weight_tensor['shape'][0]
                    in_channels = weight_tensor['shape'][1]
        
        op_name = op_info['name'].lower().replace('/', '_')
        return f"self.{op_name} = F.Conv2d(self.name + '.{op_name}', {in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding})"
    
    def _map_relu(self, op_info: Dict, graph_info: Dict) -> str:
        op_name = op_info['name'].lower().replace('/', '_')
        return f"self.{op_name} = F.Relu(self.name + '.{op_name}')"
    
    def _map_maxpool(self, op_info: Dict, graph_info: Dict) -> str:
        attrs = op_info['attributes']
        kernel_shape = attrs.get('kernel_shape', [2, 2])
        strides = attrs.get('strides', [2, 2])
        
        kernel_size = kernel_shape[0] if kernel_shape else 2
        stride = strides[0] if strides else 2
        
        op_name = op_info['name'].lower().replace('/', '_')
        return f"self.{op_name} = F.MaxPool2d(self.name + '.{op_name}', kernel_size={kernel_size}, stride={stride})"
    
    def _map_avgpool(self, op_info: Dict, graph_info: Dict) -> str:
        attrs = op_info['attributes']
        kernel_shape = attrs.get('kernel_shape', [2, 2])
        
        op_name = op_info['name'].lower().replace('/', '_')
        return f"self.{op_name} = F.AveragePool2d(self.name + '.{op_name}', kernel_shape={kernel_shape})"
    
    def _map_batchnorm(self, op_info: Dict, graph_info: Dict) -> str:
        op_name = op_info['name'].lower().replace('/', '_')
        return f"self.{op_name} = F.BatchNorm2d(self.name + '.{op_name}', num_features)  # TODO: Set num_features"
    
    def _map_add(self, op_info: Dict, graph_info: Dict) -> str:
        return None  # Add is handled in forward pass as x + y
    
    def _map_mul(self, op_info: Dict, graph_info: Dict) -> str:
        return None  # Mul is handled in forward pass as x * y
    
    def _map_reshape(self, op_info: Dict, graph_info: Dict) -> str:
        return None  # Reshape is handled in forward pass
    
    def _map_transpose(self, op_info: Dict, graph_info: Dict) -> str:
        return None  # Transpose is handled in forward pass
    
    def _map_flatten(self, op_info: Dict, graph_info: Dict) -> str:
        return None  # Flatten is handled in forward pass as reshape
    
    def _map_dropout(self, op_info: Dict, graph_info: Dict) -> str:
        attrs = op_info['attributes']
        ratio = attrs.get('ratio', 0.5)
        
        op_name = op_info['name'].lower().replace('/', '_')
        return f"self.{op_name} = F.Dropout(self.name + '.{op_name}', {ratio})"
    
    def _map_softmax(self, op_info: Dict, graph_info: Dict) -> str:
        op_name = op_info['name'].lower().replace('/', '_')
        return f"self.{op_name} = F.Softmax(self.name + '.{op_name}')"
    
    def _map_sigmoid(self, op_info: Dict, graph_info: Dict) -> str:
        op_name = op_info['name'].lower().replace('/', '_')
        return f"self.{op_name} = F.Sigmoid(self.name + '.{op_name}')"
    
    def _map_tanh(self, op_info: Dict, graph_info: Dict) -> str:
        op_name = op_info['name'].lower().replace('/', '_')
        return f"self.{op_name} = F.Tanh(self.name + '.{op_name}')"
    
    def _map_matmul(self, op_info: Dict, graph_info: Dict) -> str:
        return None  # MatMul is handled in forward pass
    
    def _map_gemm(self, op_info: Dict, graph_info: Dict) -> str:
        # GEMM is General Matrix Multiplication (like Linear layer)
        op_name = op_info['name'].lower().replace('/', '_')
        return f"self.{op_name} = SimNN.Linear(self.name + '.{op_name}', in_features, out_features)  # TODO: Set dimensions"
    
    def _map_global_avgpool(self, op_info: Dict, graph_info: Dict) -> str:
        op_name = op_info['name'].lower().replace('/', '_')
        return f"self.{op_name} = F.AveragePool2d(self.name + '.{op_name}', output_size=(1, 1), adaptive=True)"
    
    def _map_adaptive_avgpool(self, op_info: Dict, graph_info: Dict) -> str:
        attrs = op_info['attributes']
        output_size = attrs.get('output_size', (1, 1))
        
        op_name = op_info['name'].lower().replace('/', '_')
        return f"self.{op_name} = F.AveragePool2d(self.name + '.{op_name}', output_size={output_size}, adaptive=True)"
    
    def _map_layernorm(self, op_info: Dict, graph_info: Dict) -> str:
        op_name = op_info['name'].lower().replace('/', '_')
        return f"self.{op_name} = F.LayerNorm(self.name + '.{op_name}', normalized_shape)  # TODO: Set normalized_shape"
    
    def _map_gelu(self, op_info: Dict, graph_info: Dict) -> str:
        op_name = op_info['name'].lower().replace('/', '_')
        return f"self.{op_name} = F.Gelu(self.name + '.{op_name}')"
    
    def _map_leakyrelu(self, op_info: Dict, graph_info: Dict) -> str:
        attrs = op_info['attributes']
        alpha = attrs.get('alpha', 0.01)
        
        op_name = op_info['name'].lower().replace('/', '_')
        return f"self.{op_name} = F.LeakyRelu(self.name + '.{op_name}', negative_slope={alpha})"
    
    def _map_concat(self, op_info: Dict, graph_info: Dict) -> str:
        return None  # Concat is handled in forward pass
    
    def _map_split(self, op_info: Dict, graph_info: Dict) -> str:
        attrs = op_info['attributes']
        axis = attrs.get('axis', 0)
        
        op_name = op_info['name'].lower().replace('/', '_')
        return f"self.{op_name} = F.SplitOpHandle(self.name + '.{op_name}', count=2, axis={axis})  # TODO: Set count"
    
    def _map_squeeze(self, op_info: Dict, graph_info: Dict) -> str:
        return None  # Squeeze is handled in forward pass
    
    def _map_unsqueeze(self, op_info: Dict, graph_info: Dict) -> str:
        return None  # Unsqueeze is handled in forward pass


def generate_yaml_config(class_name: str, module_filename: str, input_shape: list = None, model_name: str = None) -> str:
    """Generate corresponding YAML configuration for Polaris"""
    if model_name is None:
        model_name = class_name.lower()
    
    # Generate instances based on input shape
    instances = {
        f"{model_name}_default": {"bs": 1}
    }
    
    if input_shape and len(input_shape) >= 3:
        # For image inputs [batch, channels, height, width]
        if len(input_shape) == 4:
            _, channels, height, width = input_shape
            instances.update({
                f"{model_name}_batch4": {"bs": 4, "num_channels": channels, "img_height": height, "img_width": width},
                f"{model_name}_batch8": {"bs": 8, "num_channels": channels, "img_height": height, "img_width": width},
                f"{model_name}_high_res": {"bs": 1, "num_channels": channels, "img_height": height*2, "img_width": width*2}
            })
        # For sequence inputs [batch, seq_len, features]
        elif len(input_shape) == 3:
            _, seq_len, features = input_shape
            instances.update({
                f"{model_name}_batch4": {"bs": 4, "seq_len": seq_len, "num_features": features},
                f"{model_name}_batch8": {"bs": 8, "seq_len": seq_len, "num_features": features},
                f"{model_name}_long_seq": {"bs": 1, "seq_len": seq_len*2, "num_features": features}
            })
    else:
        # Default instances for unknown shapes
        instances.update({
            f"{model_name}_batch4": {"bs": 4},
            f"{model_name}_batch8": {"bs": 8}
        })
    
    # Format instances for YAML
    instances_yaml = ""
    for name, config in instances.items():
        config_str = ", ".join([f"{k}: {v}" for k, v in config.items()])
        instances_yaml += f"      {name}: {{{config_str}}}\n"
    
    return f"""# Generated configuration for {class_name}
# This file can be included in your main workload configuration
workloads:
  - api: TTSIM
    name: {model_name}
    basedir: ../workloads
    module: {class_name}@{module_filename}
    params:
      # Add any model-specific parameters here
    instances:
{instances_yaml.rstrip()}"""


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description="Convert ONNX models to Polaris workloads",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python onnx_to_polaris.py model.onnx
  python onnx_to_polaris.py model.onnx --output workloads/GeneratedModel.py
  python onnx_to_polaris.py model.onnx --class-name MyModel --config-output config.yaml
        """
    )
    
    parser.add_argument('input', help='Input ONNX model file')
    parser.add_argument('--output', '-o', 
                       help='Output Python file (default: workloads/Generated{ModelName}.py)')
    parser.add_argument('--class-name', '-c',
                       help='Generated class name (default: derived from filename)')
    parser.add_argument('--model-name', '-m',
                       help='Model name for YAML config (default: derived from filename)')
    parser.add_argument('--config-output', 
                       help='Output YAML configuration file')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Default batch size (default: 1)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    if not HAS_ONNX:
        print("❌ ONNX is required but not installed")
        print("   Install with: pip install onnx")
        return 1
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        model_name = Path(args.input).stem
        class_name = args.class_name or f"Generated{model_name.capitalize()}"
        output_path = f"../workloads/{class_name}.py"
    
    # Create converter and run
    converter = ONNXToPolaris()
    
    try:
        config = {'bs': args.batch_size}
        python_code = converter.convert_onnx_to_polaris(
            args.input, output_path, args.class_name, config
        )
        
        # Generate YAML configuration if requested
        if args.config_output:
            class_name = args.class_name or converter._sanitize_class_name(Path(args.input).stem)
            model_name = args.model_name or Path(args.input).stem.lower()
            # Get input shape from first input tensor
            input_shape = None
            if converter.input_specs:
                first_input = next(iter(converter.input_specs.values()))
                input_shape = first_input.get('shape', None)
            
            yaml_config = generate_yaml_config(class_name, Path(output_path).name, input_shape, model_name)
            
            with open(args.config_output, 'w') as f:
                f.write(yaml_config)
            
            print(f"📝 Generated configuration: {args.config_output}")
        
        print("🎉 Conversion completed successfully!")
        print(f"   Output: {output_path}")
        print(f"   Next steps:")
        print(f"   1. Review and adjust the generated code")
        print(f"   2. Test with: python {output_path}")
        print(f"   3. Add to Polaris configuration YAML")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
