# ONNX to Polaris Converter

🚀 **Convert ONNX models to Polaris workloads in minutes**

## Quick Start

### 1. Convert ONNX Model
```bash
# Basic conversion
python onnx_to_polaris.py your_model.onnx

# Full customization
python onnx_to_polaris.py your_model.onnx \
    --output ../workloads/MyModel.py \
    --class-name MyModel \
    --config-output ../config/my_model.yaml
```

### 2. Run Complete Pipeline
```bash
# One command: ONNX → Polaris → Simulation → Results
./complete_onnx_to_polaris_pipeline.sh your_model.onnx ModelName modelname
```

### 3. Run Polaris Simulation
```bash
# Direct Polaris command
python ../polaris.py \
    --wlspec ../config/my_model.yaml \
    --filterwl modelname \
    --archspec ../config/all_archs.yaml \
    --wlmapspec ../config/wl2archmapping.yaml \
    --study my_study \
    --odir ../output \
    --outputformat yaml
```

## Files

- **`onnx_to_polaris.py`** - Main converter tool
- **`complete_onnx_to_polaris_pipeline.sh`** - End-to-end automation
- **`run_polaris_simulation.py`** - Simulation runner
- **`create_sample_onnx.py`** - Sample model creator for testing
- **`examples/`** - Sample models and generated outputs

## Features

✅ **25+ ONNX Operations** - Conv, ReLU, MaxPool, MatMul, Transformer ops, etc.  
✅ **Smart Configuration** - Auto-generates YAML configs with multiple instances  
✅ **Polaris Integration** - Direct simulation execution  
✅ **Results Analysis** - Automatic performance analysis  

## Example Usage

```bash
# Create sample model and test the pipeline
python create_sample_onnx.py
./complete_onnx_to_polaris_pipeline.sh examples/models/SimpleCNN.onnx SimpleCNN simplecnn
```

## Generated Output

- **Python Workload**: Complete Polaris-compatible class
- **YAML Configuration**: Ready-to-use Polaris config
- **Simulation Results**: Performance metrics and analysis

---

**Reduces ONNX model integration from hours to minutes!** 🎯

