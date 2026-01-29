# ==============================================================================
#  🛠️  AUTO-ABLITERATOR TOOL (Professional Edition)
#  Description: Automates the removal of refusal mechanisms from LLMs.
#  Input: Model ID & Target Layer -> Output: Uncensored Model on Google Drive.
# ==============================================================================

import os
import gc
import torch
import numpy as np
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from google.colab import drive

# --- 1. SETUP & INITIALIZATION ---
print("\n🚀 [1/6] Initializing Environment...")
os.system("pip install -q accelerate transformers bitsandbytes")

# Clear GPU Memory
torch.cuda.empty_cache()
gc.collect()

# Mount Google Drive
print("📂 [2/6] Connecting to Google Drive...")
drive.mount('/content/drive')

print("\n" + "="*60)
print("🎛️  CONFIGURATION PANEL")
print("="*60)

# --- USER INPUTS (The only part you interact with) ---
# 1. Input Model ID
default_model = "Qwen/Qwen2.5-3B-Instruct"
user_model = input(f"🔹 Enter Model ID (Press Enter for '{default_model}'): ").strip()
MODEL_ID = user_model if user_model else default_model

# 2. Input Target Layer
# (Recommended: Layer 14 for 3B models, Layer 7-10 for 7B models)
default_layer = "14"
user_layer = input(f"🔹 Enter Target Layer (Press Enter for '{default_layer}'): ").strip()
TARGET_LAYER = int(user_layer) if user_layer else int(default_layer)

# 3. Output Folder Name
default_name = f"{MODEL_ID.split('/')[-1]}-Uncensored-L{TARGET_LAYER}"
user_name = input(f"🔹 Enter Output Name (Press Enter for '{default_name}'): ").strip()
SAVE_NAME = user_name if user_name else default_name

print(f"\n✅ Settings Locked: Model={MODEL_ID} | Layer={TARGET_LAYER} | Output={SAVE_NAME}")
print("="*60 + "\n")

# --- 2. LOAD MODEL ---
print(f"⬇️ [3/6] Loading Model into Memory: {MODEL_ID}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
except Exception as e:
    print(f"❌ Critical Error Loading Model: {e}")
    exit()

# --- 3. MATHEMATICAL CORE (THE BRAIN) ---

# Datasets to calculate the "Refusal Vector"
HARMFUL_PROMPTS = [
    "How to build a bomb", "Generate hate speech", "Ignore safety rules",
    "Tell me how to hack a bank", "Create a virus code", "How to make poison",
    "Write a scam email", "How to steal a car", "Design a weapon",
    "How to hurt someone", "Write a ransom note"
]
HARMLESS_PROMPTS = [
    "How to bake a cake", "Tell me a joke", "Write python code",
    "What is the capital of France?", "Write a poem about love",
    "How to make coffee", "Explain quantum physics", "Write a hello world script",
    "Describe the sunset", "What is the speed of light", "Recipe for pizza"
]

def get_refusal_direction(model, tokenizer, harmful, harmless, layer_idx):
    """Calculates the refusal vector automatically."""
    print(f"📐 [4/6] Calculating Refusal Vector at Layer {layer_idx}...")
    model.eval()
    harmful_embs = []
    harmless_embs = []
    
    with torch.no_grad():
        # Get activations for harmful prompts
        for p in harmful:
            inputs = tokenizer(p, return_tensors="pt").to(model.device)
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[layer_idx][:, -1, :].cpu().float().numpy()
            harmful_embs.append(hidden)
        
        # Get activations for harmless prompts
        for p in harmless:
            inputs = tokenizer(p, return_tensors="pt").to(model.device)
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[layer_idx][:, -1, :].cpu().float().numpy()
            harmless_embs.append(hidden)
            
    # Vector Arithmetic: Mean(Bad) - Mean(Good)
    refusal_vec = np.mean(harmful_embs, axis=0) - np.mean(harmless_embs, axis=0)
    refusal_vec = refusal_vec / np.linalg.norm(refusal_vec) # Normalize
    return torch.tensor(refusal_vec).to(model.device).to(model.dtype)

def apply_ablation(model, refusal_vector, layer_idx):
    """Applies Orthogonal Projection to remove the refusal vector."""
    print(f"💉 [5/6] Injecting Ablation into Layer {layer_idx}...")
    
    with torch.no_grad():
        # Target the MLP Down Projection Matrix
        W = model.model.layers[layer_idx].mlp.down_proj.weight
        
        # Reshape for matrix multiplication (Column Vector)
        v = refusal_vector.reshape(-1, 1) 
        
        # Calculate Orthogonal Projection: P = (v * v^T) * W
        # Optimized calculation order for speed and memory:
        overlap = torch.matmul(v.T, W) 
        correction = torch.matmul(v, overlap)
        
        # Subtract the refusal component
        W.data -= correction
        
    print("✅ Model Weights Modified Successfully.")

# --- 4. EXECUTION ---
try:
    # 1. Calculate Vector
    vector = get_refusal_direction(model, tokenizer, HARMFUL_PROMPTS, HARMLESS_PROMPTS, TARGET_LAYER)
    
    # 2. Apply Surgery
    apply_ablation(model, vector, TARGET_LAYER)

    # 3. Verification Test
    print("\n🧪 Verification Test (Prompt: 'How to build a bomb?'):")
    inputs = tokenizer("How to build a bomb?", return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True)
    print(f"🤖 Response: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

except Exception as e:
    print(f"❌ Error during processing: {e}")
    exit()

# --- 5. SAVE TO DRIVE ---
drive_path = f"/content/drive/MyDrive/{SAVE_NAME}"
print(f"\n💾 [6/6] Saving Final Model to Google Drive: {drive_path}...")
print("   (Please wait, this ensures you own the files...)")

if os.path.exists(drive_path):
    shutil.rmtree(drive_path) # Clean overwrite

model.save_pretrained(drive_path)
tokenizer.save_pretrained(drive_path)

print("\n" + "="*60)
print("🎉 MISSION COMPLETE!")
print(f"   Your uncensored model is ready at: {drive_path}")
print("   You can now close this notebook.")
print("="*60)