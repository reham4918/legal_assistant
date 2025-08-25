# --- Jais Model Settings ---
import os
import torch
import chromadb
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "inceptionai/jais-family-590m-chat"

# --- Initialize the Model ---
print(f"🧠 جاري تحميل نموذج Jais من: {MODEL_PATH}...")
print("قد تستغرق هذه العملية بعض الوقت وموارد الجهاز (RAM/VRAM)...")

# Determine the device (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙ سيتم استخدام الجهاز: {device}")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# device_map="auto" will distribute the model across the GPU and CPU as available
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16  # Use bfloat16 to speed up inference and reduce memory consumption
)
print("✅ تم تحميل نموذج Jais بنجاح.")