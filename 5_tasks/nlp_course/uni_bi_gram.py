import random
from collections import defaultdict


class UnigramModel:
    def __init__(self):
        self.word_counts = defaultdict(int)
        self.total_words = 0

    def train(self, corpus):
        for sentence in corpus:
            for word in sentence.split():
                self.word_counts[word] += 1
                self.total_words += 1

    def calculate_sentence_probability(self, sentence):
        probability = 1.0
        for word in sentence.split():
            if self.total_words == 0 or self.word_counts[word] == 0:
                probability *= 0
            else:
                probability *= self.word_counts[word] / self.total_words
        return probability

    def generate_sentence(self, length=10):
        sentence = []
        for _ in range(length):
            rand_word = random.choice(list(self.word_counts.keys()))
            sentence.append(rand_word)
        return ' '.join(sentence)


class BigramModel:
    def __init__(self):
        self.bigram_counts = defaultdict(int)
        self.unigram_counts = defaultdict(int)

    def train(self, corpus):
        for sentence in corpus:
            words = sentence.split()
            for i in range(len(words) - 1):
                self.bigram_counts[(words[i], words[i + 1])] += 1
                self.unigram_counts[words[i]] += 1
    def calculate_sentence_probability(self, sentence):
        probability = 1.0
        words = sentence.split()
        for i in range(len(words) - 1):
            word1, word2 = words[i], words[i + 1]
            if self.unigram_counts[word1] == 0 or self.bigram_counts[(word1, word2)] == 0:
                probability *= 0
            else:
                probability *= self.bigram_counts[(word1, word2)] / self.unigram_counts[word1]
        return probability

    def generate_sentence(self, length=10):
        sentence = []
        current_word = random.choice(list(self.unigram_counts.keys()))
        sentence.append(current_word)

        for _ in range(length - 1):
            next_word = [word2 for (word1, word2) in self.bigram_counts.keys() if word1 == current_word]
            if len(next_word) == 0:
                break
            current_word = random.choice(next_word)
            sentence.append(current_word)
        return ' '.join(sentence)


# Example usage:
if __name__ == "__main__":
    corpus = ["this is a sample sentence", "another sample sentence", "another example sentence"]
    unigram_model = UnigramModel()
    unigram_model.train(corpus)

    print("Unigram Model:")
    sentence = "a sample sentence"
    print("Probability of sentence:", sentence, " is ", unigram_model.calculate_sentence_probability(sentence))
    print("Generated sentence:", unigram_model.generate_sentence(length=5))

    bigram_model = BigramModel()
    bigram_model.train(corpus)

    print("\nBigram Model:")
    sentence = "another example"
    print("Probability of sentence:", sentence, " is ", bigram_model.calculate_sentence_probability(sentence))
    print("Generated sentence:", bigram_model.generate_sentence(length=2))
