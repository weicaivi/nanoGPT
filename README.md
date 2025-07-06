# nanoGPT in C++

A character-level language model implementation in C++ using LibTorch (PyTorch C++ API), inspired by Andrej Karpathy's nanoGPT. This project implements a transformer-based language model from scratch with multi-head attention, feed-forward networks, and layer normalization.

## Features

### 🏗️ Architecture

- **Multi-Layer Transformer**: 6 transformer blocks with residual connections
- **Multi-Head Attention**: 6 attention heads with causal masking
- **Feed-Forward Networks**: 4x expansion with ReLU activation
- **Layer Normalization**: Pre-norm architecture for stable training
- **Position Embeddings**: Learned positional encodings
- **Dropout Regularization**: 0.2 dropout rate for preventing overfitting

### 📊 Model Specifications

- **Embedding Dimension**: 384
- **Number of Heads**: 6 (64 dimensions per head)
- **Number of Layers**: 6
- **Context Length**: 256 tokens
- **Batch Size**: 64
- **Vocabulary**: Character-level (99 unique characters)

### 🎯 Training Features

- **AdamW Optimizer**: Learning rate 1e-3
- **Cross-Entropy Loss**: Standard language modeling loss
- **Train/Validation Split**: 90/10 split for evaluation
- **Gradient Clipping**: Built-in in AdamW optimizer
- **Evaluation Intervals**: Every 500 iterations

## Project Structure

```
nanoGPT/
├── train.cpp              # Main training and generation code
├── CMakeLists.txt         # CMake build configuration
├── libtorch/              # LibTorch C++ library
├── Neopolitan.txt         # Training dataset (Elena Ferrante novel)
└── README.md             # This file
```

## Dependencies

- **LibTorch**: PyTorch C++ API (included in `libtorch/`)
- **CMake**: Build system
- **C++17**: Modern C++ features
- **macOS/Linux**: Tested on macOS with Apple Silicon

## Building the Project

### Prerequisites

1. Install CMake (version 3.10 or higher)
2. Ensure you have a C++17 compatible compiler
3. Download libtorch library into a directory nanoGPT/libtorch/

### Build Steps

```bash
# Clone or navigate to the project directory
cd nanoGPT

# Create build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build the project
cmake --build .

# The executable will be created as `train`
```

## Usage

### Training the Model

```bash
# Run training (this will train the model and generate samples)
./train
```

The training process will:

1. Load the training data from `Neopolitan.txt`
2. Create character-level vocabulary
3. Train the model for 5000 iterations
4. Generate text samples with custom prompts

### Output Format

```
Dataset: 2992059 characters, 99 unique tokens
Train/val split: 2692853/299206 tokens
Model: 6 layers, 6 heads, 384 dim
Training: 64 batch, 256 context, 0.001 lr
Initial loss: 4.75 (train) 4.75 (val)
iter 100: loss 3.45
iter 200: loss 3.12
...
iter 500: train 2.89, val 2.91
...
Training complete!
Prompt: "The story begins with a mysterious letter that arrived"
Generated: "The story begins with a mysterious letter that arrived..."
```

## Model Architecture Details

### Transformer Block

Each transformer block consists of:

1. **Layer Norm 1** → **Multi-Head Attention** → **Residual Connection**
2. **Layer Norm 2** → **Feed-Forward Network** → **Residual Connection**

### Multi-Head Attention

- **Input**: (B, T, n_embd) where B=batch, T=sequence length, n_embd=384
- **Projections**: Key, Query, Value projections to full embedding dimension
- **Heads**: Split into 6 heads of 64 dimensions each
- **Causal Masking**: Lower triangular mask prevents looking at future tokens
- **Scaling**: Attention scores scaled by √(head_size) for variance control

### Feed-Forward Network

- **Architecture**: Linear → ReLU → Linear → Dropout
- **Expansion**: 384 → 1536 → 384 (4x expansion)
- **Activation**: ReLU for non-linearity
- **Dropout**: 0.2 for regularization

## Text Generation

The model supports custom prompts and variable-length generation:

```cpp
// Example usage in code
std::string prompt = "In the quiet town of Naples, where the streets";
std::string generated = generate_with_prompt(model, prompt, stoi, itos, 600);
```

### Generation Features

- **Custom Prompts**: Use any text as a starting point
- **Variable Length**: Generate any number of tokens
- **Context Window**: Automatically crops to last 256 tokens during generation
- **Temperature**: Uses softmax sampling for natural text

## Training Data

The model is trained on "Neopolitan" by Elena Ferrante, a character-level dataset with:

- **Size**: ~3 million characters
- **Vocabulary**: 99 unique characters
- **Language**: English with some Italian words
- **Style**: Literary fiction

## Performance

### Training Metrics

- **Initial Loss**: ~4.75 (random initialization)
- **Final Loss**: ~2.0 (after 5000 iterations)
- **Convergence**: Stable training with good train/validation alignment
- **Memory Usage**: Efficient with gradient checkpointing

### Generation Quality

- **Coherence**: Maintains context and narrative flow
- **Style**: Captures the literary style of the training data
- **Diversity**: Generates varied and interesting continuations

## Customization

### Hyperparameters

You can modify the global hyperparameters in `train.cpp`:

```cpp
const int batch_size = 64;        // Training batch size
const int block_size = 256;       // Context window size
const int max_iters = 5000;       // Training iterations
const int n_embd = 384;           // Embedding dimension
const int n_head = 6;             // Number of attention heads
const int n_layer = 6;            // Number of transformer layers
const float learning_rate = 1e-3f; // Learning rate
const float dropout = 0.2;        // Dropout rate
```

### Training Data

Replace `Neopolitan.txt` with your own text file to train on different data.

### Model Size

Adjust `n_embd`, `n_head`, and `n_layer` to change model capacity:

- **Smaller**: Faster training, less capacity
- **Larger**: More capacity, longer training time

## Technical Notes

### Memory Management

- Uses LibTorch's automatic memory management
- Efficient tensor operations with minimal copying
- Gradient accumulation handled automatically

### Numerical Stability

- Layer normalization prevents gradient explosion
- Scaled attention scores control variance
- Dropout prevents overfitting

### Causal Masking

- Ensures model only sees past tokens during training
- Critical for language modeling tasks
- Implemented with lower triangular mask

## Future Enhancements

Potential improvements and extensions:

- **Tokenization**: Word-level or subword tokenization
- **Larger Models**: Increase model size for better performance
- **Attention Variants**: Implement different attention mechanisms
- **Optimization**: Learning rate scheduling, gradient clipping
- **Inference**: Model saving/loading, faster generation
- **Multi-GPU**: Distributed training support

## License

This project is for educational purposes. The training data (Neopolitan) is copyrighted material.

## Acknowledgments

- Inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT)
- Built with PyTorch C++ API (LibTorch)
- Training data from Elena Ferrante's Neopolitan novels
