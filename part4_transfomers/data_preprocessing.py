from enum import Enum
import torch
from sklearn.preprocessing import MinMaxScaler
import numpy as np 
import math
from enum import Enum
import tokenizers as t
import re

class Tokenisation_method(Enum):
    CHARACTER = 1
    WORD = 2
    SUBWORD = 3

class Tokenizer:
    def __init__(self, input_text, tokenisation_method=Tokenisation_method.CHARACTER):
        self.method = tokenisation_method
        self.input_text = input_text.lower()
        self.vocab = self.fit(self.input_text)
        self.n_vocab = len(self.vocab)

    def fit(self, text):
            """Builds a vocabulary from the given text."""

            if self.method is Tokenisation_method.CHARACTER:
                # Create a vocabulary mapping each unique character to a unique integer.
                # We add -1 at idx to account the fact that dict starts at 1 and not 0
                vocab = {char: idx-1 for idx, char in enumerate(sorted(set(text)), start=1)}
            
            elif self.method is Tokenisation_method.WORD:
                # Create a vocabulary mapping each unique word to a unique integer
                # We add -1 at idx to account the fact that dict starts at 1 and not 0
                
                # Tokenize while preserving '\n' as a separate token
                tokens = re.findall(r'\S+|\n', text)  # Matches words and newline characters
                vocab = {word: idx - 1 for idx, word in enumerate(sorted(set(tokens)), start=1)}
            
            elif self.method is Tokenisation_method.SUBWORD:
                # Subword-based tokenization using Byte Pair Encoding (BPE)
                  
                if isinstance(text, str):
                    text = [text]  # Convert to a list

                # Initialise BPE tokenizer
                self.subword_tokenizer = t.Tokenizer(t.models.BPE(unk_token="<UNK>"))
                self.subword_tokenizer.pre_tokenizer = t.pre_tokenizers.ByteLevel(add_prefix_space=True)
                self.subword_tokenizer.decoder = t.decoders.ByteLevel()

                # Train the tokenizer on provided text
                trainer = t.trainers.BpeTrainer(vocab_size=2000,special_tokens=["<UNK>"])
                self.subword_tokenizer.train_from_iterator(text, trainer=trainer)
                vocab = self.subword_tokenizer.get_vocab()

            return vocab
    

    def encode(self, text):
        """Tokenize an input text based on the vocabulary learned when initialising the tokenizer."""

        if self.method is Tokenisation_method.CHARACTER:
            return [self.vocab[char] for char in text if char in self.vocab]
        
        elif self.method is Tokenisation_method.WORD:
            words = re.findall(r'\S+|\n', text)
            return [self.vocab[w] for w in words if w in self.vocab]
        
        elif self.method is Tokenisation_method.SUBWORD:
            return self.subword_tokenizer.encode(text).ids
            
        
    
    def decode(self, tokens):
        """Convert a list of integer tokens back into text."""

        if self.method is Tokenisation_method.CHARACTER:
            inv_vocab = {idx: char for char, idx in self.vocab.items()}
            return ''.join(inv_vocab[token] for token in tokens if token in inv_vocab)
        
        elif self.method is Tokenisation_method.WORD:
            inv_vocab = {idx: word for word, idx in self.vocab.items()}
            return ' '.join(inv_vocab[token] for token in tokens if token in inv_vocab)
        
        elif self.method is Tokenisation_method.SUBWORD:
            return self.subword_tokenizer.decode(tokens)
            

class Transform_Tokens:
    def __init__(self, non_scaled_tokens):
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.scaler.fit(np.array(non_scaled_tokens).reshape(-1, 1))

    def transform_tokens(self,tokens, sequence_length, with_target=True):
        features, targets = self.token_to_sequence(tokens,sequence_length, with_target)
        features, targets = self.sequence_to_torch(features, targets)
        return features, targets
    
    def scale_tokens(self, unscaled_tokens):
        data = np.array(unscaled_tokens).reshape(-1, 1)
        return self.scaler.transform(data).flatten()

    def unscale_tokens(self, scaled_tokens):
        data = np.asarray(scaled_tokens).reshape(-1, 1)
        unscaled = self.scaler.inverse_transform(data).flatten()
        unscaled = [math.ceil(token) for token in unscaled]
        return unscaled

    def token_to_sequence(self,tokens, sequence_length=100, with_target=True):
        """prepare the dataset of input to output pairs and normalise the features"""
        features = []
        targets = []
        #scaled_tokens = self.scale_tokens(tokens)
        num_sequences = len(tokens) - sequence_length
        if not with_target:
            num_sequences += 1
        for i in range(num_sequences):
            seq_in = tokens[i:i + sequence_length]
            features.append([tok for tok in seq_in])
            if with_target:
                seq_out = tokens[i+1 : i + sequence_length + 1]
                targets.append(seq_out)
        return features, targets

    def sequence_to_torch(self, sequences, targets,):
        # reshape X to be [batch size, time steps, features]
        seq_length = len(sequences[0])
        #sequences = torch.tensor(sequences, dtype=torch.float32).reshape(len(sequences), seq_length, 1)
        features = torch.tensor(sequences)
        targets = torch.tensor(targets)
        return features, targets
    

if __name__ == "__main__":

    filename = "part8_transfomers/alice_in_wonderland.txt"
    raw_text = open(filename, 'r', encoding='utf-8').read()
    raw_text = raw_text.lower() #Convert to lower case

    tokenizer = Tokenizer(raw_text, Tokenisation_method.WORD)
   
    vocab = tokenizer.vocab
    keys = [key for key, val in vocab.items() if val == 233]
    print(keys)  
    input = raw_text[0:500]
    tokens = tokenizer.encode(input) 

    print(f"Input text: {input}")
    print(f"Encoded tokens: {tokens}")
    print(f"Decoded text: {tokenizer.decode(tokens)}")
    print(f"Vocabulary size: {len(tokenizer.vocab)}")
