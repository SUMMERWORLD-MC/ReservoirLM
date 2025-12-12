import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def main():
    """
    This script demonstrates how to use a pre-trained GPT-2 model to generate text.
    It requires the `transformers` and `torch` libraries.
    Install them with: pip install ".[transformers]"
    """
    # 1. Load pre-trained GPT-2 model and tokenizer
    model_name = "gpt2"
    print(f"Loading model and tokenizer for '{model_name}'...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    print("Model and tokenizer loaded successfully.")

    # Set pad token if it's not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # 2. Prepare a sample prompt
    prompt = "Reservoir computing is a framework for computation that is"
    print(f"\nPrompt: '{prompt}'")

    # 3. Encode the prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # 4. Generate text
    # The max_length parameter includes the length of the prompt
    print("\nGenerating text...")
    outputs = model.generate(
        inputs,
        max_length=50,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        attention_mask=torch.ones(inputs.shape, dtype=torch.long)
    )

    # 5. Decode and print the output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nGenerated Text:")
    print(generated_text)

if __name__ == "__main__":
    main()
