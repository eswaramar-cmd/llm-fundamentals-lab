"""
FEATURE 2: TOKEN IDS
This program shows the mapping between tokens and their numerical IDs.
A token ID is the numerical representation of a token in the tokenizer's vocabulary.

Run: python 02_token_ids.py
"""

import tiktoken

def main():
    print("=" * 50)
    print("LLM Fundamentals Practice Lab")
    print("FEATURE 2: TOKEN IDS")
    print("=" * 50)
    print()

    enc = tiktoken.get_encoding("cl100k_base")

    print("Tokenizer loaded: cl100k_base")
    print()
    print("EXPLANATION:")
    print("- A TOKEN is a piece of text (word, part of a word, punctuation, etc.)")
    print("- A TOKEN ID is the unique numerical ID representing that token")
    print("  in the tokenizer's vocabulary.")
    print("- The model never sees raw text directly. It only sees token IDs.")
    print()

    while True:
        user_input = input("Enter text (or 'quit' to exit): ").strip()

        if user_input.lower() == 'quit':
            print("Goodbye!")
            break

        if not user_input:
            print("Please enter some text.\n")
            continue

        token_ids = enc.encode(user_input)
        tokens = [enc.decode([token_id]) for token_id in token_ids]

        print()
        print("Text:")
        print(user_input)
        print()
        print("Tokens:")
        print(tokens)
        print()
        print("Token IDs:")
        print(token_ids)
        print()
        print("-" * 50)
        print()

if __name__ == "__main__":
    main()
