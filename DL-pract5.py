#!/usr/bin/env python
# coding: utf-8

# In[9]:


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import skipgrams
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import random


# In[10]:


# Sample dataset
sentences = [
    "the quick brown fox jumps over the lazy dog",
    "i love machine learning and deep learning",
    "continuous bag of words is a technique in NLP",
    "word embeddings are a form of transfer learning"
]

# Initialize a Tokenizer to convert words to unique integer indices
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)  # Fit tokenizer on the sentences to learn vocabulary
vocab_size = len(tokenizer.word_index) + 1  # Add 1 to account for padding index

# Convert each sentence to a sequence of integers based on the tokenizer vocabulary
sequences = tokenizer.texts_to_sequences(sentences)

print("Vocabulary Size:", vocab_size)
print("Sequences:", sequences)


# In[11]:


# Define context window size
window_size = 2  # This is the size of words around the target word used as context
context_target_pairs = []  # List to hold generated pairs of (context, target)


# In[12]:


# Generate context-target pairs for CBOW
for sequence in sequences:
    for i, word in enumerate(sequence):  # Loop through each word in the sentence
        # Define the context window boundaries
        start = max(0, i - window_size)
        end = min(len(sequence), i + window_size + 1)
        
        # Collect context words within the window, excluding the target word
        context_words = [sequence[j] for j in range(start, end) if j != i]
        
        # Create a pair (context, target) for each context word
        for context_word in context_words:
            context_target_pairs.append((context_word, word))

print("Sample context-target pairs:", context_target_pairs[:10])


# In[13]:


# Separate context and target word pairs into X and y datasets
X, y = zip(*context_target_pairs)  # Unpack context-target pairs
X = np.array(X)  # Convert context words to a numpy array for training
y = np.array(y)  # Convert target words to a numpy array for training



# In[14]:


# Set embedding dimension and context size
embedding_dim = 50  # Number of dimensions for the word embeddings

# Define the CBOW model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=1),  # Embedding layer to learn word vectors
    tf.keras.layers.GlobalAveragePooling1D(),  # Average embedding for each context word
    tf.keras.layers.Dense(vocab_size, activation='softmax')  # Output layer with vocab_size units to predict target word
])

# Compile the model with optimizer, loss function, and evaluation metric
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[15]:


# Reshape X to ensure compatibility
X = X.reshape(-1, 1)

# Train the model with a smaller batch size
history = model.fit(X, y, epochs=4, batch_size=2, verbose=1)


# In[16]:


# Define a function to get the embedding for a given word
def get_embedding(word):
    word_index = tokenizer.word_index[word]  # Get the index of the word from tokenizer
    return model.layers[0].get_weights()[0][word_index]  # Return the embedding for the word

# Example: Get embedding for the word 'learning'
print("Embedding for 'learning':", get_embedding("learning"))


# In[ ]:




