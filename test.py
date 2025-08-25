# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# --- Jais Model Settings ---
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
print("✅ تم تحميل نموذج Jais بنجاح.")

# test the model with a prompt
# Use the chat template expected by the model
messages = [
    {"role": "system", "content": "أجب بإيجاز وبأسلوب ودود."},
    {"role": "user",   "content": "مرحبا بك، كيف حالك؟"},
]

# Apply chat template
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

# Decode only the generated continuation (or decode all—both shown)
gen_only_ids = out[0, inputs["input_ids"].shape[-1]:]
print(tokenizer.decode(gen_only_ids, skip_special_tokens=True))
# print(tok.decode(out[0], skip_special_tokens=True))  # whole convo if you prefer