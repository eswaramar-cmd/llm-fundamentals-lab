"""
FEATURE 1: TOKENIZATION
This program tokenizes text using the cl100k_base tokenizer (used by GPT-4).
It shows you how raw text is split into tokens.

Run: python 01_tokenization.py
"""

import tiktoken

def main():
    print("=" * 50)
    print("LLM Fundamentals Practice Lab")
    print("FEATURE 1: TOKENIZATION")
    print("=" * 50)
    print()

    # Load the tokenizer
    # cl100k_base is the tokenizer used by GPT-4 and GPT-3.5-turbo
    enc = tiktoken.get_encoding("cl100k_base")

    print("Tokenizer loaded: cl100k_base (used by GPT-4)")
    print()

    while True:
        # Ask the user for input text
        user_input = input("Enter your text (or 'quit' to exit): ").strip()

        if user_input.lower() == 'quit':
            print("Goodbye!")
            break

        if not user_input:
            print("Please enter some text.\n")
            continue

        # Tokenize the text
        # encode() converts text into a list of token IDs
        # We reconstruct the token strings from those IDs
        token_ids = enc.encode(user_input)
        tokens = [enc.decode([token_id]) for token_id in token_ids]

        print()
        print("Original text:")
        print(user_input)
        print()
        print("Tokens:")
        print(tokens)
        print()
        print("Number of tokens:")
        print(len(tokens))
        print()
        print("-" * 50)
        print()

if __name__ == "__main__":
    main()
