# Convolutional Spiking Neural Network for MNIST

A Rust implementation of a Convolutional Spiking Neural Network (Conv-SNN) achieving **96.71% accuracy** on MNIST digit classification.

## Key Features

- **Hybrid Temporal Coding**: Deterministic early-phase (t<5) transitioning to stochastic spike generation
- **Asynchronous Layer Updates**: Hidden layer processes spikes every alternate timestep for efficiency
- **Hybrid Reset Dynamics**: Hard reset for conv layer, soft reset for hidden layer
- **High Accuracy**: 96.71% test accuracy without complex backprop through time
- **Energy Efficient**: 1.9x-8.8x more efficient than traditional ANNs (measured)

## Architecture

```
Input (28x28) → Conv(5x5, stride=2) → 16x12x12 → LIF → FC(2304→1024) → LIF → FC(1024→10) → Output
```

- **Convolutional Layer**: 16 filters of 5x5 with stride 2
- **Spiking Neurons**: 2,304 + 1,024 LIF neurons
- **Temporal Coding**: 10 timesteps per forward pass
- **Learning**: Rate-based surrogate gradient (not STDP)

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 96.71% |
| Batch Size | 64 |
| Learning Rate | 1e-2 with 0.95 decay every 5 epochs |
| Timesteps | 10 |
| Training Time | ~100s per epoch* |
| Convergence | ~20-30 epochs |

*On modern multi-core CPU. Actual time varies with hardware.

## SOTA Comparison

| Model | Accuracy | Energy Efficiency | Notes |
|-------|----------|-------------------|-------|
| **This Work** | **96.71%** | **1.9-8.8x (measured)** | Rate-based surrogate gradient + async updates |
| Diehl & Cook 2015 | 95.0% | Not reported* | Unsupervised STDP |
| Tavanaei & Maida 2019 | 98.4% | Not reported* | BP-STDP |
| Shrestha & Orchard 2018 | 99.1% | Not reported* | SLAYER algorithm |
| Kim & Panda 2021 | 98.6%** | 10-100x* | BNTT for SNNs |
| Traditional CNN | 99.7% | 1x (baseline) | Standard backprop |

\* Energy efficiency claims from papers; most don't provide measurements  
\** On different datasets/configurations

### What Makes Our Approach Unique

1. **Hybrid Temporal Coding**: Combines deterministic (t<5) and stochastic (t≥5) spike generation
2. **Asynchronous Processing**: Hidden layer updates every other timestep (50% compute reduction)
3. **Layer-Specific Reset**: Hard reset for conv, soft reset for hidden layer
4. **Simple Learning**: Rate-based surrogate gradients without complex BPTT

## Installation

```bash
# Clone the repository
git clone https://github.com/karthikkolli/mnist-snn.git
cd mnist-snn

# Build the project
cargo build --release

# The MNIST dataset will be automatically downloaded from Hugging Face on first run
```

## Usage

```bash
# Build and run with default settings (30 epochs)
cargo run --release

# Run with custom number of epochs
cargo run --release -- 50
```

## Reproducibility

The implementation uses fixed random seeds (seed=42) for:
- Weight initialization
- Training data shuffling  
- Stochastic spike generation

This improves reproducibility, though minor variations may occur due to parallel execution and floating-point operations.

## Project Structure

```
mnist_snn/
├── Cargo.toml             
├── README.md              
├── LICENSE                 # MIT License
├── ENERGY_CALCULATION.md   # Detailed efficiency analysis
├── snn/                   # SNN library
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs         # Library exports
│       └── conv_snn.rs    # Core SNN implementation
└── mnist_solver/          # MNIST training binary
    ├── Cargo.toml
    └── src/
        └── main.rs        # Training loop & auto-download
```

## Technical Details

### Spiking Neural Network

Our SNN uses modified Leaky Integrate-and-Fire (LIF) neurons with several innovations:

#### Membrane Dynamics
```
V(t+1) = decay * V(t) + I(t)
if V(t) > threshold:
    spike = 1
    # Reset mechanism:
    # Conv layer: V(t) = 0       (hard reset)
    # Hidden layer: V(t) *= 0.5   (soft reset)
```

#### Temporal Coding Strategy
```python
# Hybrid spike generation
if t < 5:
    # Deterministic: Higher intensity → Earlier spikes
    spike = 1 if pixel_value > (1.0 - t/5.0) else 0
else:
    # Stochastic: Poisson process
    spike = 1 if random() < pixel_value else 0
```

#### Asynchronous Updates
```python
# Conv layer updates every timestep
conv_output = conv_layer(spikes)

# Hidden layer updates every other timestep
if t % 2 == 0:
    hidden_output = hidden_layer(conv_spikes)
```

Key parameters:
- Decay factor: 0.9
- Threshold: 0.8  
- Timesteps: 10
- Conv→Hidden update ratio: 2:1

#### Learning Method
We use a **rate-based surrogate gradient** approach that accumulates spike counts over timesteps and treats them as continuous values for backpropagation. This is simpler and more efficient than complex surrogate methods like SLAYER or BPTT.

### Energy Efficiency

SNNs are inherently energy-efficient due to:
- Sparse, event-driven computation (measured spike rates: 6-11%)
- Binary spike communication vs 32-bit floats
- Asynchronous layer updates reducing operations by 50%

Our measurements show:
- **1.9x** efficiency from sparse operations alone (MAC count)
- **2.9x** with optimized memory bandwidth  
- **8.8x** on neuromorphic hardware

See `ENERGY_CALCULATION.md` for detailed analysis.

## References

1. Diehl, P. U., & Cook, M. (2015). Unsupervised learning of digit recognition using spike-timing-dependent plasticity. Frontiers in computational neuroscience, 9, 99.

2. Tavanaei, A., & Maida, A. (2019). BP-STDP: Approximating backpropagation using spike timing dependent plasticity. Neurocomputing, 330, 39-47.

3. Shrestha, S. B., & Orchard, G. (2018). SLAYER: Spike layer error reassignment in time. NeurIPS.

4. Kim, Y., & Panda, P. (2021). Revisiting batch normalization for training low-latency deep spiking neural networks from scratch. Frontiers in neuroscience, 15, 773954.

## License

MIT License - see LICENSE file for details
