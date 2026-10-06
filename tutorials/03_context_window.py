"""
FEATURE 3: CONTEXT WINDOW
This program calculates how many tokens your text uses against a context window size.
The context window is measured in TOKENS, not characters or words.

Run: python 03_context_window.py
"""

import tiktoken

def main():
    print("=" * 50)
    print("LLM Fundamentals Practice Lab")
    print("FEATURE 3: CONTEXT WINDOW")
    print("=" * 50)
    print()

    enc = tiktoken.get_encoding("cl100k_base")

    print("EXPLANATION:")
    print("- The CONTEXT WINDOW is the maximum number of tokens a model can process at once.")
    print("- It is measured in TOKENS, not characters or words.")
    print("- If your input exceeds the context window, the model will truncate or fail.")
    print()

    while True:
        user_input = input("Enter your text (or 'quit' to exit): ").strip()

        if user_input.lower() == 'quit':
            print("Goodbye!")
            break

        if not user_input:
            print("Please enter some text.\n")
            continue

        token_count = len(enc.encode(user_input))
        token_ids = enc.encode(user_input)
        tokens = [enc.decode([token_id]) for token_id in token_ids]

        try:
            max_context = int(input("Maximum context size (e.g., 4096, 8192, 32768): ").strip())
        except ValueError:
            print("Please enter a valid number.\n")
            continue

        remaining = max_context - token_count
        percentage = (token_count / max_context) * 100

        print()
        print("Original text:")
        print(user_input)
        print()
        print(tokens)
        print("Tokens used:")
        print(token_count)
        print()
        print("Remaining tokens:")
        print(remaining)
        print()
        print("Percentage used:")
        print(f"{percentage:.2f}%")
        print()

        if token_count > max_context:
            print("WARNING: Your text EXCEEDS the context window!")
        elif percentage > 80:
            print("WARNING: You are using more than 80% of the context window.")

        print("-" * 50)
        print()

if __name__ == "__main__":
    main()
