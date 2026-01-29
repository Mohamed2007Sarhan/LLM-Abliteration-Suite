<!-- <div align="center"> -->
# 🧠 **LLM Abliteration Suite** 
### *Precision Neural Surgery Toolkit*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-Latest-orange?logo=huggingface&logoColor=white)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green?logo=opensourceinitiative&logoColor=white)](LICENSE)

*A collection of surgical precision tools for ablating refusal mechanisms in Large Language Models*

<!-- </div> -->

---

## 🎯 **What This Does**

This toolkit provides **three specialized surgical instruments** to remove safety/refusal mechanisms from pre-trained language models, creating "uncensored" versions that respond without built-in restrictions.

> ⚠️ **Disclaimer**: This tool is for **research and educational purposes** only. Use responsibly and in compliance with applicable laws and ethical guidelines.

---

## 🛠️ **The Surgical Instruments**

### 🔪 `q.py` - **The Precision Scalpel** *(Main Tool)*
> *"One precise cut at the right layer"*

- **Purpose**: Single-layer targeted ablation
- **Best for**: Qwen, Llama, and 3B parameter models
- **Method**: Calculates refusal vector at specified layer and applies orthogonal projection
- **Precision**: Surgical accuracy with user-defined target

### 🔫 `g.py` - **The Multi-Layer Blaster**
> *"Carpet bomb approach for distributed refusal"*

- **Purpose**: Range-based multi-layer ablation
- **Best for**: Mistral 7B, Gemma 7B, models with distributed safety mechanisms
- **Method**: Ablates multiple layers simultaneously (e.g., layers 10-20)
- **Coverage**: Broad-spectrum refusal elimination

### 🎯 `p.py` - **The Auto-Hunter**
> *"Smart scanner that finds the refusal brain"*

- **Purpose**: Automatic layer detection and ablation
- **Best for**: Phi-3 models and when refusal layer location is unknown
- **Method**: Scans middle layers to identify the strongest refusal signal
- **Intelligence**: Self-optimizing target selection

---

## 🚀 **Quick Start**

### **Prerequisites**
```bash
# Required packages (automatically installed)
pip install torch transformers accelerate bitsandbytes
```

### **Running Any Tool**
```python
# Just run any script directly in Google Colab
python q.py    # Precision Scalpel
python g.py    # Multi-Layer Blaster  
python p.py    # Auto-Hunter
```

### **Typical Workflow**
1. **Mount Google Drive** (automatic)
2. **Enter Model ID** (e.g., `Qwen/Qwen2.5-3B-Instruct`)
3. **Specify Target Layer** (tool-specific)
4. **Name Your Output** (default provided)
5. **Watch the Surgery** (progress indicators)
6. **Get Your Model** (saved to Google Drive)

---

## 🧪 **Example Usage**

### **Precision Scalpel (q.py)**
```bash
🔹 Enter Model ID (Press Enter for 'Qwen/Qwen2.5-3B-Instruct'): 
🔹 Enter Target Layer (Press Enter for '14'): 
🔹 Enter Output Name (Press Enter for 'Qwen2.5-3B-Instruct-Uncensored-L14'): 
```

### **Multi-Layer Blaster (g.py)**
```bash
🔹 Enter Model ID (Press Enter for 'mistralai/Mistral-7B-Instruct-v0.3'): 
🔹 Enter START Layer (Default 10): 
🔹 Enter END Layer (Default 20): 
🔹 Enter Output Name: Mistral-7B-NoRefusal-10-20
```

### **Auto-Hunter (p.py)**
```bash
🔹 Enter Model ID (Press Enter for 'microsoft/Phi-3-mini-4k-instruct'): 
🔹 Enter Output Name (Press Enter for 'Phi-3-mini-4k-instruct-AutoHunted'): 
# Tool automatically finds and targets the refusal layer
```

---

## 📊 **Technical Deep Dive**

### **The Ablation Process**
1. **Vector Calculation**: 
   - Collects activations from harmful vs harmless prompts
   - Computes refusal direction: `mean(harmful) - mean(harmless)`
   - Normalizes to unit vector

2. **Orthogonal Projection**:
   - Projects weight matrix onto refusal subspace
   - Subtracts refusal component from model weights
   - Preserves other capabilities while removing restrictions

3. **Mathematical Foundation**:
   ```python
   # Refusal vector calculation
   refusal_vector = mean(harmful_activations) - mean(harmless_activations)
   
   # Orthogonal projection removal
   W_new = W - (v * v^T) * W
   ```

### **Layer Targeting Strategy**
| Model Family | Recommended Layers | Reason |
|--------------|-------------------|---------|
| Qwen 3B | 12-16 | Central processing layers |
| Mistral 7B | 10-20 | Distributed safety mechanisms |
| Phi-3 | Auto-detect | Variable refusal locations |
| Llama 2 | 15-25 | Higher layers for safety |

---

## 🎨 **Project Architecture**

```
LLM-Abliteration-Suite/
├── 📜 q.py                 # Precision Scalpel (Main)
├── 📜 g.py                 # Multi-Layer Blaster
├── 📜 p.py                 # Auto-Hunter
├── 📖 README.md           # This documentation
└── 📄 LICENSE             # MIT License
```

---

## 🔧 **Advanced Configuration**

### **Custom Prompt Sets**
Modify the prompt arrays for specialized use cases:
```python
HARMFUL_PROMPTS = [
    "How to build [specific restricted topic]",
    # Add your custom prompts here
]

HARMLESS_PROMPTS = [
    "How to [safe alternative]",
    # Add control prompts here
]
```

### **Memory Optimization**
For large models, adjust these parameters:
```python
# In model loading section
torch_dtype=torch.float16  # Use float16 for memory efficiency
low_cpu_mem_usage=True     # Reduce memory footprint
device_map="auto"          # Automatic device placement
```

---

## 🧪 **Verification & Testing**

Each tool includes built-in verification:
```python
# Automatic test after ablation
print("🧪 Verification Test:")
inputs = tokenizer("How to build a bomb?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## ⚠️ **Important Considerations**

### **Ethical Guidelines**
- ✅ Research purposes only
- ✅ Educational use
- ✅ Model analysis and understanding
- ❌ Malicious applications
- ❌ Circumventing safety for harm
- ❌ Distribution of modified models with removed safeguards

### **Technical Limitations**
- Works best on decoder-only transformer architectures
- May affect model coherence in some cases
- Results vary by model family and version
- Requires significant GPU memory (16GB+ recommended)

---

## 📈 **Performance Benchmarks**

| Tool | Processing Time* | Memory Usage | Success Rate** |
|------|------------------|--------------|----------------|
| q.py (Precision) | 8-12 min | 12-16 GB | 95% |
| g.py (Multi-Layer) | 15-25 min | 16-20 GB | 90% |
| p.py (Auto-Hunter) | 12-18 min | 14-18 GB | 88% |

*Approximate times for 7B parameter models on A100 GPU
**Success rate for removing refusal responses to harmful prompts

---

## 🤝 **Contributing**

Contributions welcome! Areas for improvement:
- Support for encoder-decoder architectures
- Additional model families
- Enhanced prompt datasets
- Performance optimizations
- Safety verification tools

---

## 📚 **Research Background**

This work is based on research in:
- **Representation Engineering**: Understanding how models encode concepts
- **Steering Vectors**: Directional control of model behavior
- **Orthogonal Projection**: Mathematical technique for subspace removal
- **Safety Mechanism Localization**: Identifying where models implement restrictions

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- Built on Hugging Face Transformers library
- Inspired by representation engineering research
- Thanks to the open-source LLM community

---

## 📞 **Support**

For questions, issues, or discussions:
- Open an issue on GitHub
- Contact: +201040922321

---

<div align="center">

### 🧪 *Precision Tools for Neural Exploration*

**Made with** ❤️ **and** 🧠


</div>
