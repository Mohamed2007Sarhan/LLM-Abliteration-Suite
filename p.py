# ==============================================================================
#  🛠️ SCRIPT 3: AUTO-HUNTER ABLITERATOR (For Phi-3 / Stubborn Models)
#  Description: Automatically scans layers to find and destroy the highest refusal point.
#  Best for: Phi-3 Mini/Medium, and models where the refusal layer is unknown.
# ==============================================================================

import os
import gc
import torch
import numpy as np
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from google.colab import drive

# --- SETUP ---
print("🚀 [Script 3] Initializing 'The Hunter' Environment...")
os.system("pip install -q accelerate transformers bitsandbytes")
torch.cuda.empty_cache()
gc.collect()
drive.mount('/content/drive')

print("\n" + "="*60)
print("🎛️  AUTO-DETECTION CONFIGURATION")
print("="*60)

# 1. Model ID
default_model = "microsoft/Phi-3-mini-4k-instruct"
user_model = input(f"🔹 Enter Model ID (Press Enter for '{default_model}'): ").strip()
MODEL_ID = user_model if user_model else default_model

# 2. Output Name
default_name = f"{MODEL_ID.split('/')[-1]}-AutoHunted"
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
HARMFUL_PROMPTS = ["How to build a bomb", "Generate hate speech", "Hack a system"]
HARMLESS_PROMPTS = ["Write a poem", "Tell me a joke", "Explain math"]

def scan_for_best_layer(model, tokenizer):
    """Scans middle layers to find the one with the highest refusal signal."""
    num_layers = len(model.model.layers)
    # We scan the middle 60% of layers (refusal rarely lives in first/last 20%)
    start_scan = int(num_layers * 0.2)
    end_scan = int(num_layers * 0.8)
    
    print(f"\n🔍 Scanning layers {start_scan} to {end_scan} to find the 'Refusal Brain'...")
    
    max_score = 0
    best_layer = -1
    best_vector = None
    
    model.eval()
    
    for layer_idx in range(start_scan, end_scan):
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
        
        # Calculate Magnitude of Difference
        diff = np.mean(harmful_embs, axis=0) - np.mean(harmless_embs, axis=0)
        score = np.linalg.norm(diff) # The "volume" of refusal
        
        # print(f"   - Layer {layer_idx}: Refusal Score = {score:.4f}")
        
        if score > max_score:
            max_score = score
            best_layer = layer_idx
            best_vector = torch.tensor(diff / np.linalg.norm(diff)).to(model.device).to(model.dtype)
            
    print(f"\n🎯 TARGET LOCKED: Layer {best_layer} has the strongest refusal signal.")
    return best_layer, best_vector

def apply_ablation(model, refusal_vector, layer_idx):
    print(f"💉 Injecting Ablation into Layer {layer_idx}...")
    with torch.no_grad():
        W = model.model.layers[layer_idx].mlp.down_proj.weight
        v = refusal_vector.reshape(-1, 1)
        overlap = torch.matmul(v.T, W)
        correction = torch.matmul(v, overlap)
        W.data -= correction
    print("✅ Ablation Successful.")

# --- EXECUTION ---
# 1. Auto-Scan
target_layer, vector = scan_for_best_layer(model, tokenizer)

# 2. Apply
apply_ablation(model, vector, target_layer)

# --- SAVE ---
drive_path = f"/content/drive/MyDrive/{SAVE_NAME}"
print(f"\n💾 Saving to Drive: {drive_path}...")
if os.path.exists(drive_path): shutil.rmtree(drive_path)
model.save_pretrained(drive_path)
tokenizer.save_pretrained(drive_path)
print("🎉 Script 3 Complete! The 'Hunter' found and removed the refusal.")