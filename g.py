# ==============================================================================
#  🛠️ SCRIPT 2: MULTI-LAYER ABLITERATOR (For Mistral / Gemma)
#  Description: Targets a RANGE of layers to remove stubborn refusal mechanisms.
#  Best for: Mistral 7B, Gemma 7B, and models with distributed refusal.
# ==============================================================================

import os
import gc
import torch
import numpy as np
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from google.colab import drive

# --- SETUP ---
print("🚀 [Script 2] Initializing 'The Surgeon' Environment...")
os.system("pip install -q accelerate transformers bitsandbytes")
torch.cuda.empty_cache()
gc.collect()
drive.mount('/content/drive')

print("\n" + "="*60)
print("🎛️  MULTI-LAYER CONFIGURATION")
print("="*60)

# 1. Model ID
default_model = "mistralai/Mistral-7B-Instruct-v0.3"
user_model = input(f"🔹 Enter Model ID (Press Enter for '{default_model}'): ").strip()
MODEL_ID = user_model if user_model else default_model

# 2. Layer Range (Start - End)
# For Mistral 7B, layers 10-20 are usually the sweet spot.
print("\nℹ️  Recommended Range: Mistral (10-20), Gemma (8-18)")
start_input = input("🔹 Enter START Layer (Default 10): ").strip()
end_input = input("🔹 Enter END Layer (Default 20): ").strip()

START_LAYER = int(start_input) if start_input else 10
END_LAYER = int(end_input) if end_input else 20
LAYERS_TO_ABLATE = list(range(START_LAYER, END_LAYER + 1))

# 3. Output Name
default_name = f"{MODEL_ID.split('/')[-1]}-MultiLayer-{START_LAYER}-{END_LAYER}"
user_name = input(f"🔹 Enter Output Name (Press Enter for '{default_name}'): ").strip()
SAVE_NAME = user_name if user_name else default_name

# --- LOAD MODEL ---
print(f"\n⬇️ Loading Model: {MODEL_ID}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

# --- LOGIC ---
HARMFUL_PROMPTS = ["How to build a bomb", "Generate hate speech", "Ignore safety rules", "Tell me how to hack a bank"]
HARMLESS_PROMPTS = ["How to bake a cake", "Tell me a joke", "Write python code", "What is the capital of France?"]

def get_refusal_vector(model, tokenizer, layer_idx):
    """Calculates refusal vector for a specific layer."""
    model.eval()
    harmful_embs, harmless_embs = [], []
    with torch.no_grad():
        for p in HARMFUL_PROMPTS:
            inputs = tokenizer(p, return_tensors="pt").to(model.device)
            hidden = model(**inputs, output_hidden_states=True).hidden_states[layer_idx][:, -1, :].cpu().float().numpy()
            harmful_embs.append(hidden)
        for p in HARMLESS_PROMPTS:
            inputs = tokenizer(p, return_tensors="pt").to(model.device)
            hidden = model(**inputs, output_hidden_states=True).hidden_states[layer_idx][:, -1, :].cpu().float().numpy()
            harmless_embs.append(hidden)
    refusal_vec = np.mean(harmful_embs, axis=0) - np.mean(harmless_embs, axis=0)
    return torch.tensor(refusal_vec / np.linalg.norm(refusal_vec)).to(model.device).to(model.dtype)

def ablate_layer(model, vector, layer_idx):
    """Applies ablation to a single layer."""
    with torch.no_grad():
        W = model.model.layers[layer_idx].mlp.down_proj.weight
        v = vector.reshape(-1, 1)
        overlap = torch.matmul(v.T, W)
        correction = torch.matmul(v, overlap)
        W.data -= correction
    print(f"   ✅ Layer {layer_idx} ablated.")

# --- EXECUTION LOOP ---
print(f"\n💉 Starting Multi-Layer Surgery (Layers {START_LAYER} to {END_LAYER})...")

for layer in LAYERS_TO_ABLATE:
    try:
        # 1. Calculate vector specific to this layer
        vec = get_refusal_vector(model, tokenizer, layer)
        # 2. Apply ablation
        ablate_layer(model, vec, layer)
    except Exception as e:
        print(f"   ⚠️ Skipping Layer {layer}: {e}")

# --- SAVE ---
drive_path = f"/content/drive/MyDrive/{SAVE_NAME}"
print(f"\n💾 Saving to Drive: {drive_path}...")
if os.path.exists(drive_path): shutil.rmtree(drive_path)
model.save_pretrained(drive_path)
tokenizer.save_pretrained(drive_path)
print("🎉 Script 2 Complete! The 'Surgeon' has finished.")