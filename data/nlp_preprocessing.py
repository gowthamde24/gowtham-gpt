import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:

        
        #combine all sentences into 1 list
        combined = []
        for sentence in positive:
            combined.append(sentence)
        for sentence in negative:
            combined.append(sentence)

        #collect every unique wrod

        unique_words=set()
        for sentence in combined:
            words = sentence.split()
            for word in words:
                unique_words.add(word)

        #sort the words

        vacabulary = sorted(unique_words)
        # assign each word an id

        word_to_id = {}
        next_id = 1
        for word in vacabulary:
            word_to_id[word]=next_id
            next_id += 1

        #turn each sentence into list of ids

        encoded = []

        for sentence in combined:
            ids=[]
            words=sentence.split()
            for word in words:
                ids.append(word_to_id[word])
            encoded.append(torch.tensor(ids))

        padded = nn.utils.rnn.pad_sequence(encoded,batch_first=True)
        return padded



        

            
