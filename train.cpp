#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <random>
#include <torch/torch.h>

// Global hyperparameters
const int batch_size = 64;
const int block_size = 256;
const int max_iters = 5000;
const int eval_interval = 500;
const float learning_rate = 1e-3f;
const int eval_iters = 500;
const int n_embd = 384;
const int n_head = 6;      // Number of attention heads
const int head_size = 64;   // Size of each attention head (n_embd / n_head = 32 / 4 = 8)
const int n_layer = 6;     // Number of transformer blocks
const float dropout = 0.2;

// Data structure to hold training data
struct TrainingData {
    std::string text;                    // Raw text from file
    torch::Tensor encoded_text;          // Text encoded as tensor
    std::unordered_map<char, int> stoi;  // String to integer mapping (char -> int)
    std::unordered_map<int, char> itos;  // Integer to string mapping (int -> char)
    int vocab_size;                      // Size of vocabulary
};

// Function to read text file and return the content
std::string read_text_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    
    // Read the entire file into a string
    std::string text((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
    file.close();
    
    return text;
}

// Function to create vocabulary from text
std::unordered_map<char, int> create_vocabulary(const std::string& text) {
    std::unordered_set<char> unique_chars;
    
    // Collect all unique characters
    for (char c : text) {
        unique_chars.insert(c);
    }
    
    // Create mapping from characters to integers
    std::unordered_map<char, int> stoi;
    int idx = 0;
    for (char c : unique_chars) {
        stoi[c] = idx++;
    }
    
    return stoi;
}

// Function to create reverse mapping (int -> char)
std::unordered_map<int, char> create_reverse_vocabulary(const std::unordered_map<char, int>& stoi) {
    std::unordered_map<int, char> itos;
    for (const auto& pair : stoi) {
        itos[pair.second] = pair.first;
    }
    return itos;
}

// Function to encode text using vocabulary and return as tensor
torch::Tensor encode_text(const std::string& text, const std::unordered_map<char, int>& stoi) {
    std::vector<int64_t> encoded;
    encoded.reserve(text.size());
    
    for (char c : text) {
        encoded.push_back(stoi.at(c));
    }
    
    // Convert to tensor
    auto options = torch::TensorOptions().dtype(torch::kInt64);
    return torch::from_blob(encoded.data(), {static_cast<long>(encoded.size())}, options).clone();
}

// Function to decode tensor back to text
std::string decode_text(const torch::Tensor& encoded, const std::unordered_map<int, char>& itos) {
    std::string decoded;
    auto accessor = encoded.accessor<int64_t, 1>();
    
    for (int i = 0; i < encoded.size(0); i++) {
        decoded += itos.at(accessor[i]);
    }
    
    return decoded;
}

// Main function to load and prepare training data
TrainingData load_training_data(const std::string& filename) {
    // Read the text file
    std::string text = read_text_file(filename);
    
    // Create vocabulary
    auto stoi = create_vocabulary(text);
    auto itos = create_reverse_vocabulary(stoi);
    int vocab_size = stoi.size();
    
    // Encode the text as tensor
    auto encoded_text = encode_text(text, stoi);
    
    // Verify encoding/decoding works correctly
    std::string decoded = decode_text(encoded_text, itos);
    if (decoded != text) {
        throw std::runtime_error("Encoding/decoding verification failed!");
    }
    
    return {text, encoded_text, stoi, itos, vocab_size};
}

// Function to split data into train and validation sets
std::pair<torch::Tensor, torch::Tensor> split_data(const torch::Tensor& data, double train_ratio = 0.9) {
    int total_size = data.size(0);
    int train_size = static_cast<int>(total_size * train_ratio);
    
    // Split the data
    auto train_data = data.slice(0, 0, train_size);
    auto val_data = data.slice(0, train_size, total_size);
    
    return {train_data, val_data};
}

// Function to get training batches as tensors
std::pair<torch::Tensor, torch::Tensor> get_batches(const torch::Tensor& train_data, const torch::Tensor& val_data, 
                                                   int batch_size, int block_size, const std::string& split = "train") {
    // Select the appropriate dataset based on split
    torch::Tensor data;
    if (split == "train") {
        data = train_data;
    } else {
        data = val_data;
    }
    
    // Generate random starting indices using PyTorch's random functions
    auto ix = torch::randint(data.size(0) - block_size, {batch_size}, torch::dtype(torch::kInt64));
    
    // Create vectors to hold the individual sequences
    std::vector<torch::Tensor> x_sequences;
    std::vector<torch::Tensor> y_sequences;
    x_sequences.reserve(batch_size);
    y_sequences.reserve(batch_size);
    
    // Extract sequences using the random indices
    for (int i = 0; i < batch_size; i++) {
        int start_idx = ix[i].item<int64_t>();
        
        // Input sequence (x) is tokens 0 to block_size-1
        x_sequences.push_back(data.slice(0, start_idx, start_idx + block_size));
        
        // Target sequence (y) is tokens 1 to block_size (shifted by 1)
        y_sequences.push_back(data.slice(0, start_idx + 1, start_idx + block_size + 1));
    }
    
    // Stack the sequences into batch tensors
    auto x = torch::stack(x_sequences, 0);
    auto y = torch::stack(y_sequences, 0);
    
    return {x, y};
}

// Function to print tensor statistics
void print_tensor_stats(const torch::Tensor& tensor, const std::string& name) {
    std::cout << name << " shape: " << tensor.sizes() << std::endl;
    std::cout << name << " dtype: " << tensor.dtype() << std::endl;
    std::cout << name << " min: " << tensor.min().item<int64_t>() << std::endl;
    std::cout << name << " max: " << tensor.max().item<int64_t>() << std::endl;
    std::cout << name << " mean: " << tensor.to(torch::kFloat32).mean().item<float>() << std::endl;
}

// Function to demonstrate the context-target relationship for a single sequence
void demonstrate_context_target(const torch::Tensor& x, const torch::Tensor& y, 
                               const std::unordered_map<int, char>& itos, int block_size) {
    std::cout << "\n=== Context-Target Demonstration ===" << std::endl;
    std::cout << "For a single sequence with block_size=" << block_size << ":" << std::endl;
    
    auto x_accessor = x.accessor<int64_t, 1>();
    auto y_accessor = y.accessor<int64_t, 1>();
    
    for (int t = 0; t < block_size; t++) {
        std::cout << "Step " << t << ":" << std::endl;
        
        // Context: x[0:t+1]
        std::cout << "  Context (x[0:" << (t+1) << "]): ";
        for (int j = 0; j <= t; j++) {
            std::cout << x_accessor[j] << " ";
        }
        std::cout << " -> \"" << decode_text(x.slice(0, 0, t+1), itos) << "\"" << std::endl;
        
        // Target: y[t]
        std::cout << "  Target (y[" << t << "]): " << y_accessor[t] 
                  << " -> '" << itos.at(y_accessor[t]) << "'" << std::endl;
        std::cout << std::endl;
    }
}

// Function to calculate bag-of-words using lower triangular matrix trick
torch::Tensor calculate_xbow(const torch::Tensor& x) {
    // x is (B, T) tensor of token indices
    auto B = x.size(0);
    auto T = x.size(1);
    
    // Create one-hot encoding of x: (B, T, vocab_size)
    // First, we need to get the vocab_size from the max value in x
    auto vocab_size = x.max().item<int64_t>() + 1;
    
    // Create one-hot encoding
    auto x_one_hot = torch::zeros({B, T, vocab_size}, torch::dtype(torch::kFloat32));
    x_one_hot.scatter_(2, x.unsqueeze(-1), 1.0); // (B, T, vocab_size)
    
    // Create lower triangular mask: (T, T)
    auto mask = torch::tril(torch::ones({T, T}, torch::dtype(torch::kFloat32)));
    
    // Apply mask to get cumulative sums: (B, T, vocab_size)
    // We'll use matrix multiplication: mask @ x_one_hot
    auto xbow = torch::matmul(mask, x_one_hot); // (B, T, vocab_size)
    
    // Normalize by the number of tokens up to each position
    // Create normalization factors: (T,)
    auto counts = torch::arange(1, T + 1, torch::dtype(torch::kFloat32));
    
    // Apply normalization: (B, T, vocab_size) / (T,) -> (B, T, vocab_size)
    xbow = xbow / counts.unsqueeze(0).unsqueeze(-1);
    
    return xbow;
}

// Function to demonstrate xbow calculation
void demonstrate_xbow(const torch::Tensor& x, const std::unordered_map<int, char>& itos, int block_size) {
    std::cout << "\n=== Bag-of-Words (XBOW) Demonstration ===" << std::endl;
    
    auto xbow = calculate_xbow(x);
    std::cout << "XBOW shape: " << xbow.sizes() << std::endl;
    
    // Show the first sequence
    auto x_sample = x.index({0}); // Get first sequence: (T,)
    auto xbow_sample = xbow.index({0}); // (T, vocab_size)
    
    std::cout << "Original sequence: ";
    auto x_accessor = x_sample.accessor<int64_t, 1>();
    for (int i = 0; i < block_size; i++) {
        std::cout << x_accessor[i] << " ";
    }
    std::cout << " -> \"" << decode_text(x_sample, itos) << "\"" << std::endl;
    
    // Show the bag-of-words representation at each position
    for (int t = 0; t < block_size; t++) {
        std::cout << "Position " << t << " (context: ";
        for (int j = 0; j <= t; j++) {
            std::cout << x_accessor[j] << " ";
        }
        std::cout << "):" << std::endl;
        
        // Get the BOW vector at position t
        auto bow_vector = xbow_sample.index({t}); // (vocab_size,)
        
        // Show non-zero elements (the tokens that appear in the context)
        auto non_zero_indices = torch::nonzero(bow_vector).squeeze(-1); // Always 1D
        if (non_zero_indices.numel() == 0) {
            std::cout << "  BOW vector (non-zero elements): <none>" << std::endl;
            continue;
        }
        std::cout << "  BOW vector (non-zero elements): ";
        auto indices_accessor = non_zero_indices.accessor<int64_t, 1>();
        for (int i = 0; i < non_zero_indices.size(0); i++) {
            int token_idx = indices_accessor[i];
            float value = bow_vector[token_idx].item<float>();
            std::cout << token_idx << "(" << value << ") ";
        }
        std::cout << std::endl;
    }
}

// Multi-head Self-Attention layer
class MultiHeadAttention : public torch::nn::Module {
public:
    MultiHeadAttention(int n_embd, int n_head) {
        // Ensure n_embd is divisible by n_head
        TORCH_CHECK(n_embd % n_head == 0, "n_embd must be divisible by n_head");
        
        this->n_head = n_head;
        this->head_size = n_embd / n_head;
        
        // Key, Query, Value projections - project to full n_embd dimension
        key = register_module("key", 
            torch::nn::Linear(torch::nn::LinearOptions(n_embd, n_embd)));
        query = register_module("query", 
            torch::nn::Linear(torch::nn::LinearOptions(n_embd, n_embd)));
        value = register_module("value", 
            torch::nn::Linear(torch::nn::LinearOptions(n_embd, n_embd)));
        
        // Output projection
        proj = register_module("proj", 
            torch::nn::Linear(torch::nn::LinearOptions(n_embd, n_embd)));
        
        // Dropout for regularization
        dropout_layer = register_module("dropout", 
            torch::nn::Dropout(torch::nn::DropoutOptions(dropout)));
    }
    
    torch::Tensor forward(const torch::Tensor& x) {
        // x is (B, T, C) where C = n_embd
        auto B = x.size(0);
        auto T = x.size(1);
        auto C = x.size(2);
        
        // Calculate key, query, value: (B, T, C) -> (B, T, C)
        auto k = key(x);   // (B, T, C)
        auto q = query(x); // (B, T, C)
        auto v = value(x); // (B, T, C)
        
        // Reshape to separate heads: (B, T, C) -> (B, T, n_head, head_size)
        k = k.view({B, T, n_head, head_size}).transpose(1, 2); // (B, n_head, T, head_size)
        q = q.view({B, T, n_head, head_size}).transpose(1, 2); // (B, n_head, T, head_size)
        v = v.view({B, T, n_head, head_size}).transpose(1, 2); // (B, n_head, T, head_size)
        
        // Compute attention scores: (B, n_head, T, head_size) @ (B, n_head, head_size, T) -> (B, n_head, T, T)
        auto scores = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(head_size);
        
        // Apply causal mask (lower triangular) to prevent looking at future tokens
        auto mask = torch::tril(torch::ones({T, T}, torch::dtype(torch::kFloat32)));
        mask = mask.unsqueeze(0).unsqueeze(0); // Add batch and head dimensions: (1, 1, T, T)
        scores = scores.masked_fill(mask == 0, -1e9); // Set masked positions to -inf
        
        // Apply softmax to get attention weights
        auto att = torch::softmax(scores, -1); // (B, n_head, T, T)
        att = dropout_layer(att);
        
        // Apply attention to values: (B, n_head, T, T) @ (B, n_head, T, head_size) -> (B, n_head, T, head_size)
        auto out = torch::matmul(att, v);
        
        // Reshape back: (B, n_head, T, head_size) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view({B, T, C});
        
        // Project back to embedding dimension
        out = proj(out); // (B, T, C)
        
        return out;
    }

private:
    int n_head;
    int head_size;
    torch::nn::Linear key{nullptr};
    torch::nn::Linear query{nullptr};
    torch::nn::Linear value{nullptr};
    torch::nn::Linear proj{nullptr};
    torch::nn::Dropout dropout_layer{nullptr};
};

// Feed-Forward Network layer
class FeedForward : public torch::nn::Module {
public:
    FeedForward(int n_embd, int n_hidden) {
        // First linear layer: expands the dimension
        net = register_module("net", 
            torch::nn::Sequential(
                torch::nn::Linear(torch::nn::LinearOptions(n_embd, n_hidden)),
                torch::nn::ReLU(),
                torch::nn::Linear(torch::nn::LinearOptions(n_hidden, n_embd)),
                torch::nn::Dropout(torch::nn::DropoutOptions(dropout))
            ));
    }
    
    torch::Tensor forward(const torch::Tensor& x) {
        return net->forward(x);
    }

private:
    torch::nn::Sequential net{nullptr};
};

// Transformer Block: Multi-head attention + Feed-forward with layer norm and residual connections
class TransformerBlock : public torch::nn::Module {
public:
    TransformerBlock(int n_embd, int n_head) {
        // Layer normalization for attention
        ln1 = register_module("ln1", 
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({n_embd})));
        
        // Multi-head self-attention layer
        sa_head = register_module("sa_head", 
            std::make_shared<MultiHeadAttention>(n_embd, n_head));
        
        // Layer normalization for feed-forward
        ln2 = register_module("ln2", 
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({n_embd})));
        
        // Feed-forward network layer
        ffwd = register_module("ffwd", 
            std::make_shared<FeedForward>(n_embd, 4 * n_embd));
    }
    
    torch::Tensor forward(const torch::Tensor& x) {
        // Communication: Layer norm → Multi-head self-attention → Residual connection
        auto x_att = x + sa_head->forward(ln1(x)); // Pre-norm: norm before attention
        
        // Computation: Layer norm → Feed-forward network → Residual connection
        auto x_ffwd = x_att + ffwd->forward(ln2(x_att)); // Pre-norm: norm before feed-forward
        
        return x_ffwd;
    }

private:
    torch::nn::LayerNorm ln1{nullptr};
    std::shared_ptr<MultiHeadAttention> sa_head{nullptr};
    torch::nn::LayerNorm ln2{nullptr};
    std::shared_ptr<FeedForward> ffwd{nullptr};
};

// Bigram Language Model class
class BigramLanguageModel : public torch::nn::Module {
public:
    BigramLanguageModel(int vocab_size) {
        // Token embedding table: each token gets an embedding vector
        token_embedding_table = register_module("token_embedding_table", 
            torch::nn::Embedding(torch::nn::EmbeddingOptions(vocab_size, n_embd)));
        position_embedding_table = register_module("position_embedding_table", 
            torch::nn::Embedding(torch::nn::EmbeddingOptions(block_size, n_embd)));
        
        // Multiple transformer blocks
        for (int i = 0; i < n_layer; i++) {
            auto block = std::make_shared<TransformerBlock>(n_embd, n_head);
            transformer_blocks.push_back(register_module("transformer_block_" + std::to_string(i), block));
        }
        
        // Final layer normalization before the language model head
        ln_f = register_module("ln_f", 
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({n_embd})));
        
        // Language model head: projects embeddings to logits
        lm_head = register_module("lm_head", 
            torch::nn::Linear(torch::nn::LinearOptions(n_embd, vocab_size)));
    }
    
    // Forward method that takes in index and target tensors
    torch::Tensor forward(const torch::Tensor& idx, const torch::Tensor& targets = torch::Tensor()) {
        auto B = idx.size(0);
        auto T = idx.size(1);
        // Get token embeddings: (B,T) -> (B,T,n_embd)
        auto token_emb = token_embedding_table(idx); // (B,T,C)
        
        // Handle position embeddings - use modulo to handle sequences longer than block_size
        auto pos_indices = torch::arange(T, torch::dtype(torch::kInt64)) % block_size;
        auto pos_emb = position_embedding_table(pos_indices); // (T,C)
        auto x = token_emb + pos_emb; // (B,T,C)
        
        // Apply multiple transformer blocks: communication + computation
        for (auto& block : transformer_blocks) {
            x = block->forward(x); // (B,T,C)
        }
        
        // Apply final layer normalization
        x = ln_f(x); // (B,T,C)
        
        // Project to logits: (B,T,n_embd) -> (B,T,vocab_size)
        auto logits = lm_head(x);
        
        // If no targets provided, just return logits
        if (!targets.defined() || targets.numel() == 0) {
            return logits;
        }
        
        // Reshape logits and targets for cross entropy loss
        auto C = logits.size(2);
        
        // Reshape logits to (B*T, C) and targets to (B*T)
        logits = logits.view({B * T, C});
        auto targets_flat = targets.view({B * T});
        
        // Compute cross entropy loss
        auto loss = torch::nn::functional::cross_entropy(logits, targets_flat);
        
        return loss;
    }
    
    // Generate text given a context
    std::string generate(const torch::Tensor& idx, const std::unordered_map<int, char>& itos, 
                        int max_new_tokens = 100) {
        // Make a mutable copy of idx
        torch::Tensor cur_idx = idx.clone();
        for (int i = 0; i < max_new_tokens; i++) {
            // Crop the context to the last block_size tokens
            auto idx_cond = cur_idx.index({torch::indexing::Slice(), torch::indexing::Slice(-block_size, torch::indexing::None)});
            
            // Get the predictions
            auto logits = forward(idx_cond);
            
            // Focus only on the last time step
            auto logits_last = logits.index({torch::indexing::Slice(), -1, torch::indexing::Slice()}); // (B, C)
            
            // Apply softmax to get probabilities
            auto probs = torch::nn::functional::softmax(logits_last, torch::nn::functional::SoftmaxFuncOptions(1));
            
            // Sample from the distribution
            auto idx_next = torch::multinomial(probs, 1); // (B, 1)
            
            // Append sampled index to the running sequence
            cur_idx = torch::cat({cur_idx, idx_next}, 1); // (B, T+1)
        }
        
        // Convert the generated indices back to text
        std::string generated_text;
        auto accessor = cur_idx.accessor<int64_t, 2>();
        for (int i = 0; i < cur_idx.size(1); i++) {
            generated_text += itos.at(accessor[0][i]);
        }
        
        return generated_text;
    }

private:
    torch::nn::Embedding token_embedding_table{nullptr};
    torch::nn::Embedding position_embedding_table{nullptr};
    std::vector<std::shared_ptr<TransformerBlock>> transformer_blocks;
    torch::nn::LayerNorm ln_f{nullptr};
    torch::nn::Linear lm_head{nullptr};
};

// Function to generate text with a custom prompt
std::string generate_with_prompt(BigramLanguageModel& model, 
                                const std::string& prompt,
                                const std::unordered_map<char, int>& stoi,
                                const std::unordered_map<int, char>& itos,
                                int max_new_tokens = 200) {
    // Encode the prompt
    auto prompt_encoded = encode_text(prompt, stoi);
    auto context = prompt_encoded.unsqueeze(0); // Add batch dimension
    
    // Generate text
    return model.generate(context, itos, max_new_tokens);
}

// Function to estimate loss over multiple batches
std::pair<float, float> estimate_loss(const BigramLanguageModel& model, 
                                     const torch::Tensor& train_data, 
                                     const torch::Tensor& val_data,
                                     int batch_size, int block_size, 
                                     int eval_iters = 200) {
    // Create a temporary copy for evaluation (since we need to call non-const methods)
    BigramLanguageModel temp_model = model;
    
    // Set model to evaluation mode
    temp_model.eval();
    torch::NoGradGuard no_grad;
    
    float train_loss_sum = 0.0f;
    float val_loss_sum = 0.0f;
    
    // Compute average loss over multiple batches
    for (int iter = 0; iter < eval_iters; iter++) {
        // Get training batch
        auto [x_train, y_train] = get_batches(train_data, val_data, batch_size, block_size, "train");
        auto train_loss = temp_model.forward(x_train, y_train);
        train_loss_sum += train_loss.item<float>();
        
        // Get validation batch
        auto [x_val, y_val] = get_batches(train_data, val_data, batch_size, block_size, "val");
        auto val_loss = temp_model.forward(x_val, y_val);
        val_loss_sum += val_loss.item<float>();
    }
    
    // Set model back to training mode
    temp_model.train();
    
    // Return average losses
    return {train_loss_sum / eval_iters, val_loss_sum / eval_iters};
}

int main() {
    try {
        std::string filename = "Neopolitan.txt"; 
        
        // Load and prepare the training data
        TrainingData data = load_training_data(filename);
        
        // Print basic statistics
        std::cout << "Dataset: " << data.text.length() << " characters, " << data.vocab_size << " unique tokens" << std::endl;
        
        // Split into train and validation sets
        auto [train_data, val_data] = split_data(data.encoded_text, 0.9);
        
        std::cout << "Train/val split: " << train_data.size(0) << "/" << val_data.size(0) << " tokens" << std::endl;
        
        // Example of getting batches from train data
        auto [x, y] = get_batches(train_data, val_data, batch_size, block_size, "train");
        
        // Example of getting batches from validation data
        auto [x_val, y_val] = get_batches(train_data, val_data, batch_size, block_size, "val");
        
        // Create and test the Bigram Language Model
        BigramLanguageModel model(data.vocab_size);
        
        // Create AdamW optimizer
        auto optimizer = torch::optim::AdamW(model.parameters(), 
            torch::optim::AdamWOptions(learning_rate));
        
        std::cout << "Model: " << n_layer << " layers, " << n_head << " heads, " << n_embd << " dim" << std::endl;
        std::cout << "Training: " << batch_size << " batch, " << block_size << " context, " << learning_rate << " lr" << std::endl;
        
        // Estimate initial loss before training
        auto [initial_train_loss, initial_val_loss] = estimate_loss(model, train_data, val_data, 
                                                                   batch_size, block_size, eval_iters);
        std::cout << "Initial loss: " << initial_train_loss << " (train) " << initial_val_loss << " (val)" << std::endl;
        
        // Training loop
        for (int iter = 0; iter < max_iters; iter++) {
            // Get a batch of data
            auto [x, y] = get_batches(train_data, val_data, batch_size, block_size, "train");
            
            // Forward pass
            auto loss = model.forward(x, y);
            
            // Backward pass
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            
            // Print progress
            if (iter % 100 == 0) {
                std::cout << "iter " << iter << ": loss " << loss.item<float>() << std::endl;
            }
            
            // Evaluate on validation set
            if (iter % eval_interval == 0) {
                // Estimate loss over multiple batches for more stable evaluation
                auto [avg_train_loss, avg_val_loss] = estimate_loss(model, train_data, val_data, 
                                                                   batch_size, block_size, eval_iters);
                
                std::cout << "iter " << iter << ": train " << avg_train_loss << ", val " << avg_val_loss << std::endl;
            }
        }
        
        std::cout << "Training complete!" << std::endl;
        
        // Test text generation with custom prompt
        std::string custom_prompt = "The story begins with a mysterious letter that arrived";
        std::string generated = generate_with_prompt(model, custom_prompt, data.stoi, data.itos, 200);
        std::cout << "Prompt: \"" << custom_prompt << "\"" << std::endl;
        std::cout << "Generated: \"" << generated << "\"" << std::endl;
        
        // Generate multiple samples with different prompts
        std::cout << "\n=== Multiple Generation Samples ===" << std::endl;
        
        std::vector<std::string> prompts = {
            "In the quiet town of Naples, where the streets",
            "She opened the door slowly, her heart pounding",
            "The old man sat by the window, watching the",
            "When Elena first met Lila, she knew that",
            "The factory workers gathered in the square, their"
        };
        
        for (int i = 0; i < prompts.size(); i++) {
            std::cout << "\nSample " << (i+1) << ":" << std::endl;
            std::cout << "Prompt: \"" << prompts[i] << "\"" << std::endl;
            
            std::string generated = generate_with_prompt(model, prompts[i], data.stoi, data.itos, 150);
            std::cout << "Generated: \"" << generated << "\"" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
