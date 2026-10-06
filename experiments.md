# Experiments — LLM Fundamentals Practice Lab

Complete these experiments yourself. For each one, record your Input, Observation, and explanation.

---

## Experiment 1: Tokenization of short words

**Task:** Compare how the tokenizer splits these inputs.

Try each input separately in `01_tokenization.py`:

- `"Hello"`
- `"Hello!"`
- `"Hello, how are you?"`

**Record:**
| Input | Token Count | Observation |
|---|---|---|
| Hello | | |
| Hello! | | |
| Hello, how are you? | | |

**Why do you think this happened?**

---

## Experiment 2: Tokenization of common vs. uncommon words

**Task:** Compare how the tokenizer handles common words vs. longer words.

Try each input separately:

- `"I love AI"`
- `"I love artificial intelligence"`
- `"I love artificialintelligence"` (no space)

**Record:**
| Input | Token Count | Observation |
|---|---|---|
| I love AI | | |
| I love artificial intelligence | | |
| I love artificialintelligence | | |

**Why do you think this happened?**

---

## Experiment 3: Long paragraph token count

**Task:** Write a paragraph of 3–5 sentences in your own words. Paste it into `03_context_window.py` with a context size of 4096.

**Record:**
- Your paragraph:
- Tokens used:
- Percentage of 4096 used:
- Remaining tokens:

**Why do you think this happened?**

---

## Experiment 4: Punctuation, numbers, and special characters

**Task:** Try these inputs in `01_tokenization.py`:

- `"The price is $19.99"`
- `"Visit https://example.com/path"`
- `"2024-08-22"`
- `"Python --> AI --> LLMs"`
- `"C++ and C#"`

**Record:**
| Input | Token Count | Observation |
|---|---|---|
| The price is $19.99 | | |
| Visit https://example.com/path | | |
| 2024-08-22 | | |
| Python --> AI --> LLMs | | |
| C++ and C# | | |

**Why do you think this happened?**

---

## Experiment 5: Token IDs and the vocabulary

**Task:** Compare the same text in `02_token_ids.py` using different cases and words.

- `"Apple"`
- `"apple"`
- `"Apple banana cherry"`

**Record:**
| Input | Tokens | Token IDs | Observation |
|---|---|---|---|
| Apple | | | |
| apple | | | |
| Apple banana cherry | | | |

**Why do you think this happened?**

---

## Experiment 6: Context window pressure

**Task:** In `03_context_window.py`, enter the same long text but test it against different context sizes:

- 512
- 4096
- 32768

**Record:**
| Context Size | Tokens Used | Percentage | Remaining | Observation |
|---|---|---|---|---|
| 512 | | | | |
| 4096 | | | | |
| 32768 | | | | |

**Why do you think this happened?**

---

## Experiment 7: Next-token prediction with different prompts

**Task:** Run `04_next_token_prediction.py` and try these prompts:

- `"The capital of France is"`
- `"Once upon a time"`
- `"Python is a programming language that"`
- `"2 + 2 equals"`

**Record:**
| Prompt | Generated Response | Did it make sense? |
|---|---|---|
| The capital of France is | | |
| Once upon a time | | |
| Python is a programming language that | | |
| 2 + 2 equals | | |

**Why do you think this happened?**

---

## Experiment 8: Parameters before and after training

**Task:** Run `05_parameters_demo.py` and observe the parameter values before and after training. Then try modifying the training data or learning rate.

**Original training data:** `y = 2x + 1` with x = [1, 2, 3, 4]

**Test with new x = 10:**
- Predicted value:
- Expected value (2*10 + 1):

**Try changing learning rate to 0.001:**
- Does the model still learn well?
- What happens to the loss?

**Try changing training data to y = 3x - 1:**
- New training data:
- Does the model adapt?

**Why do you think this happened?**

---

## Experiment 9: Repetition and token efficiency

**Task:** Compare these inputs in `01_tokenization.py`:

- `"artificial intelligence"`
- `"artificial intelligence artificial intelligence"`
- `"artificial-intelligence"`

**Record:**
| Input | Token Count | Tokens per repeat | Observation |
|---|---|---|---|
| artificial intelligence | | | |
| artificial intelligence artificial intelligence | | | |
| artificial-intelligence | | | |

**Why do you think this happened?**

---

## Experiment 10: Empty and edge cases

**Task:** Try these edge cases in `01_tokenization.py`:

- Empty string (just press Enter)
- `" "` (single space)
- `"   "` (multiple spaces)
- `"\n"` (newline - type it in your terminal if possible)

**Record:**
| Input | Token Count | Tokens | Observation |
|---|---|---|---|
| (empty) | | | |
| " " | | | |
| "   " | | | |
| newline | | | |

---

## Experiment 11: Token ID vs Embedding

**Task:** Run `06_embeddings.py` and observe the difference between token IDs and embeddings.

**Record:**
| Word | Token IDs | Embedding (first 5 values) | Similarity with 'cat' |
|---|---|---|---|
| cat | | | |
| dog | | | |
| car | | | |

**Questions:**
1. Are token IDs meaningful numbers? Why or why not?
2. Do 'cat' and 'dog' have similar token IDs? Do they have similar embeddings?
3. What does this tell you about how models represent words?

---

## Experiment 12: Cosine similarity observations

**Task:** Run `06_embeddings.py` and record the cosine similarity scores.

**Record:**
| Pair | Similarity Score | Expected? Why? |
|---|---|---|
| cat vs kitten | | |
| cat vs car | | |
| king vs queen | | |
| king vs banana | | |

**Questions:**
1. Which pair is most similar? Does the score match your intuition?
2. Which pair is least similar? Why?
3. Can you think of a pair where similarity might be surprising?

---

## Experiment 13: Sentence embeddings

**Task:** Run `06_embeddings.py` and observe the sentence embeddings section.

**Record:**
| Sentence A | Sentence B | Similarity | Explanation |
|---|---|---|---|
| The cat sat on the mat | A kitten is sitting on a rug | | |
| The king ruled the kingdom | The queen governed the land | | |
| I drive my car to work | I ate a banana for breakfast | | |

**Questions:**
1. Do longer sentences have larger embedding vectors?
2. Do sentences about the same topic have high similarity even with different words?
3. How is sentence similarity different from word similarity?

---

## Experiment 14: Try your own words

**Task:** Modify `06_embeddings.py` to add your own word pairs and calculate similarity.

**Try these:**
- "happy" and "joyful"
- "sad" and "happy"
- "computer" and "laptop"
- "music" and "silence"
- "hot" and "cold"

**Record:**
| Word A | Word B | Similarity | Your Prediction | Correct? |
|---|---|---|---|---|
| | | | | |

**Questions:**
1. Were any scores surprising?
2. Can you explain why certain words are more similar than expected?
3. What limitations do you see in this embedding model?

**Why do you think this happened?**
