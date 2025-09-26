from collections import defaultdict

# Training corpus
corpus = [
    "<s> I love NLP </s>",
    "<s> I love deep learning </s>",
    "<s> deep learning is fun </s>"
]

# Tokenize corpus into words
tokenized_corpus = [sentence.split() for sentence in corpus]

# Compute unigram and bigram counts
unigram_counts = defaultdict(int)
bigram_counts = defaultdict(int)

for sentence in tokenized_corpus:
    for i in range(len(sentence)):
        unigram_counts[sentence[i]] += 1
        if i < len(sentence) - 1:
            bigram = (sentence[i], sentence[i + 1])
            bigram_counts[bigram] += 1

# Estimate bigram probabilities using MLE
bigram_prob = {}
for (w1, w2), count in bigram_counts.items():
    bigram_prob[(w1, w2)] = count / unigram_counts[w1]

# Function to calculate probability of a sentence
def sentence_probability(sentence):
    words = sentence.split()
    prob = 1.0
    for i in range(len(words) - 1):
        bigram = (words[i], words[i + 1])
        if bigram in bigram_prob:
            prob *= bigram_prob[bigram]
        else:
            prob *= 0  # no smoothing
    return prob

# Test sentences
sentences = [
    "<s> I love NLP </s>",
    "<s> I love deep learning </s>"
]

# Print unigram counts
print("Unigram Counts:")
for word, count in unigram_counts.items():
    print(f"{word}: {count}")

# Print bigram counts
print("\nBigram Counts:")
for bigram, count in bigram_counts.items():
    print(f"{bigram}: {count}")

# Print bigram probabilities
print("\nBigram Probabilities (MLE):")
for bigram, prob in bigram_prob.items():
    print(f"P({bigram[1]} | {bigram[0]}) = {prob:.4f}")

# Calculate and print sentence probabilities
print("\nSentence Probabilities:")
for sent in sentences:
    prob = sentence_probability(sent)
    print(f"{sent} -> {prob:.6f}")

# Determine which sentence the model prefers
prob_1 = sentence_probability(sentences[0])
prob_2 = sentence_probability(sentences[1])

preferred_sentence = sentences[0] if prob_1 > prob_2 else sentences[1]
print("\nThe model prefers:", preferred_sentence)
print("Reason: The preferred sentence has a higher product of bigram probabilities based on the training corpus.")
