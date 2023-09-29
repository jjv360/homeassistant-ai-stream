from transformers import AutoTokenizer
import transformers
import torch

# model = "meta-llama/Llama-2-7b-chat-hf"
model = "openlm-research/open_llama_3b_v2"
access_token = "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"

# Check if using GPU
print("GPU Support: ", torch.cuda.is_available())

# Load tokenizer model
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model, token=access_token, use_fast=False)

# Create text streamer
print("Creating streamer...")
streamer = transformers.TextStreamer(tokenizer, skip_prompt=True)

# Load pipeline model
print("Loading model...")
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
    streamer=streamer,
    token=access_token
)

# Prompt text so far
promptText = ""

# Loop
while True:

    # Get input from the user
    inputText = input("Input: ")

    # Attach to the prompt
    promptText += f"Q: {inputText}\nA:"

    # print("")
    # print("=== BLOCK")
    # print(promptText)
    # print("===")

    # Run input through the AI
    sequences = pipeline(
        promptText,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=len(promptText) + 128,
    )

    # Remove the input text from the full text output
    fullText = "" + sequences[0]['generated_text']
    outputText = fullText[len(promptText) : len(fullText)]

    # Attach output to the prompt text
    promptText += outputText + "\n"


# Log output
# for seq in sequences:
#     print(f"Result: {seq['generated_text']}")