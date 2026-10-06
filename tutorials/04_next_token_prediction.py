"""
FEATURE 4: NEXT-TOKEN PREDICTION

This program uses Llama 3.2 3B through Ollama.

The model generates text one token at a time.
"""

from ollama import chat


def main():

    print("=" * 50)
    print("LLM Fundamentals Practice Lab")
    print("FEATURE 4: NEXT-TOKEN PREDICTION")
    print("=" * 50)
    print()

    print("Using Llama 3.2 3B through Ollama")
    print("The model predicts the next token repeatedly.")
    print()

    while True:

        prompt = input("Enter a prompt (or 'quit' to exit): ").strip()

        if prompt.lower() == "quit":
            print("Goodbye!")
            break

        if not prompt:
            print("Please enter a prompt.\n")
            continue

        print("\nGenerating...\n")

        response = chat(
            model="llama3.2:3b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0.2,
                "num_predict": 50
            }
        )

        generated_text = response["message"]["content"]

        print("Generated response:")
        print(generated_text)
        print()
        print("-" * 50)


if __name__ == "__main__":
    main()