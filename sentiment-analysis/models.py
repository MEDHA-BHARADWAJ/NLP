# models.py
from sentiment_data import *
from evaluator import *

from collections import Counter
import os
import numpy as np
import torch
from torch import nn, optim

import pandas as pd
import string
import re

######################################
# IMPLEMENT THE SENTIMENT CLASSIFIER #
######################################

class FeedForwardNeuralNetClassifier(nn.Module):
    """
    The Feed-Forward Neural Net sentiment classifier.
    """
    def __init__(self, n_classes, vocab_size, emb_dim, n_hidden_units, vocab_ref, glove_path):
        """
        In the __init__ function, you will define modules in FFNN.
        :param n_classes: number of classes in this classification problem
        :param vocab_size: size of vocabulary
        :param emb_dim: dimension of the embedding vectors
        :param n_hidden_units: dimension of the hidden units
        """
        super(FeedForwardNeuralNetClassifier, self).__init__()
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_hidden_units = n_hidden_units
       
        # TODO: create a randomly initialized embedding matrix, and set padding_idx as 0
        # PAD's embedding will not be trained and by default is initialized as zero

        if glove_path is not None: 
            embedding_dict = {}
            # Pre-processing the GloVe Embeddings file
            with open('data/glove.6B.300d.txt', 'rt', encoding='utf-8') as glove:
                for line in glove:
                    values = line.split()
                    word = values[0]
                    vectors = np.asarray(values[1:], 'float32')
                    embedding_dict[word] = vectors
            glove.close()

            num_words = len(vocab_ref)
            embedding_matrix = np.zeros((num_words, 300))
            embedding1 = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            em_vec = embedding1.weight.detach().numpy()
            # Mapping the embedding vectors for the words in our vocab with the GloVe Embeddings
            for word in vocab_ref[1:]:
                embedding_vector = embedding_dict.get(word.lower())
                ind = vocab_ref.index(word)
                if embedding_vector is not None:
                    embedding_matrix[ind] = embedding_vector
            self.embeddings = nn.Embedding.from_pretrained(torch.from_numpy(embedding_matrix).float())
        else:
            self.embeddings =  nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=0) # replace None with the correct implementation

        # TODO: implement the FFNN architecture
        # when you build the FFNN model, you will need specify the embedding size using self.emb_dim, the hidden size using self.n_hidden_units,
        # and the output class size using self.n_classes
        self.dan = nn.Sequential(nn.Linear(self.emb_dim,self.n_hidden_units), nn.ReLU(), nn.Linear(self.n_hidden_units, self.n_classes))
        self._sigmoid = nn.Sigmoid()
        self._softmax = nn.Softmax()


    def forward(self, batch_inputs: torch.Tensor, batch_lengths: torch.Tensor) -> torch.Tensor:
        """
        The forward function, which defines how FFNN should work when given a batch of inputs and their actual sent lengths (i.e., before PAD)
        :param batch_inputs: a torch.Tensor object of size (n_examples, max_sent_length_in_this_batch), which is the *indexed* inputs
        :param batch_lengths: a torch.Tensor object of size (n_examples), which describes the actual sentence length of each example (i.e., before PAD)
        :return the logits outputs of FFNN (i.e., the unnormalized hidden units before softmax)
        """
        # TODO: implement the forward function, which returns the logits
        #raise Exception("Not Implemented!")

        embedded_input = self.embeddings(batch_inputs)

        encoded_input = embedded_input.sum(1)
        encoded_input /= batch_lengths.view(embedded_input.size(0), -1)

        out = self.dan(encoded_input)
        return out
    
    def batch_predict(self, batch_inputs: torch.Tensor, batch_lengths: torch.Tensor) -> List[int]:
        """
        Make predictions for a batch of inputs. This function may directly invoke `forward` (which passes the input through FFNN and returns the output logits)

        :param batch_inputs: a torch.Tensor object of size (n_examples, max_sent_length_in_this_batch), which is the *indexed* inputs
        :param batch_lengths: a torch.Tensor object of size (n_examples), which describes the actual sentence length of each example (i.e., before PAD)
        :return: a list of predicted classes for this batch of data, either 0 for negative class or 1 for positive class
        """
        # TODO: implement the prediction function, which could reuse the forward function 
        # but should return a list of predicted labels
        #raise Exception("Not Implemented!")

        res = self.forward(batch_inputs, batch_lengths)
        res = self._softmax(res)
        out = torch.argmax(res,1)
        return out

##################################
# IMPLEMENT THE TRAINING METHODS #
##################################

def train_feedforward_neural_net(
    args,
    train_exs: List[SentimentExample], 
    dev_exs: List[SentimentExample]) -> FeedForwardNeuralNetClassifier:
    """
    Main entry point for your modifications. Trains and returns a FFNN model (whose architecture is configured based on args)

    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """

    # TODO: read in all training examples and create a vocabulary (a List-type object called `vocab`)
    # Note to self - try vocab building using pytorch
    #vocab = [] # replace None with the correct implementation
    #data = pd.read_csv('data/train.txt', sep='\t', header=None)
    #data.columns = ["label","text"]
    #data['text'] = data['text'].str.lower()
    
    #def remove_special_characters(text, remove_digits=True):
     # text=re.sub(r'[^a-zA-z0-9\s]','',text)
      #text=re.sub(r'\b\w{1,3}\b','',text)
      #return text
    #data['text'] = data['text'].apply(lambda x:''.join([i for i in x if i not in string.punctuation]))
    #data['text']=data['text'].apply(remove_special_characters)

    #tag = re.compile(r'<[^>]+>')
    #data['text'] = data['text'].apply(lambda x: tag.sub('', x)) #removing html labels
    #data['text'] = data['text'].replace(r'http\S+', '', regex=True).replace(r'www.\S+', '', regex=True).replace(r'http\S+', '', regex=True).replace(r'"', '', regex=True)

    #text = data.text
    #res = [sub.split() for sub in text]
    #vocab = np.concatenate(res)
    #vocab = vocab.tolist()
     
    vocab_pre = []
    with open("data/train.txt") as f: 
        for line in f:
            fields = line.split("\t")[1]
            word = [word for word in fields.split()]
            vocab_pre = vocab_pre + word

    counts = Counter(vocab_pre) 
    output = Counter({k: c for k, c in counts.items() if c > 1})
    out = [r[0] for r in output.items()]

    # add PAD and UNK as the first two tokens
    # DO NOT CHANGE, PAD must go first and UNK next (as their indices have been hard-coded in several places)
    vocab = ["PAD", "UNK"] + out
    print("Vocab size:", len(vocab))
    # write vocab to an external file, so the vocab can be reloaded to index the test set
    with open("data/vocab.txt", "w") as f:
        for word in vocab:
            f.write(word + "\n")

    # indexing the training/dev examples
    indexing_sentiment_examples(train_exs, vocabulary=vocab, UNK_idx=1)
    indexing_sentiment_examples(dev_exs, vocabulary=vocab, UNK_idx=1)

    # TODO: create the FFNN classifier
    model = FeedForwardNeuralNetClassifier(n_classes=2, vocab_size = len(vocab), emb_dim = args.emb_dim, n_hidden_units = args.n_hidden_units, vocab_ref = vocab, glove_path = args.glove_path)# replace None with the correct implementation

    # TODO: define an Adam optimizer, using default config
    #optimizer = optim.Adam(model.parameters(), lr=0.09) # replace None with the correct implementation
    #optimizer = optim.SGD(model.parameters(), lr=0.5)
    #optimizer = optim.Adagrad(model.parameters(), lr=0.5)
    optimizer = optim.Adam(model.parameters()) # replace None with the correct implementation
    # create a batch iterator for the training data
    batch_iterator = SentimentExampleBatchIterator(train_exs, batch_size=args.batch_size, PAD_idx=0, shuffle=True)

    # training
    best_epoch = -1
    best_acc = -1
    for epoch in range(args.n_epochs):
        print("Epoch %i" % epoch)

        batch_iterator.refresh() # initiate a new iterator for this epoch

        model.train() # turn on the "training mode"
        batch_loss = 0.0
        batch_example_count = 0
        batch_data = batch_iterator.get_next_batch()
        while batch_data is not None:
            batch_inputs, batch_lengths, batch_labels = batch_data
            #print(batch_labels.unsqueeze(1).float())
            # TODO: clean up the gradients for this batch
            optimizer.zero_grad()

            # TODO: call the model to get the logits
            output = model.forward(batch_inputs, batch_lengths)
            
            # TODO: calculate the loss (let's name it `loss`, so the follow-up lines could collect the stats)
            criterion = nn.CrossEntropyLoss()
            sfmax = nn.Softmax()
            loss = criterion(sfmax(output), batch_labels.long())
        
            # record the loss and number of examples, so we could report some stats
            batch_example_count += len(batch_labels)
            batch_loss += loss.item() * len(batch_labels)

            # TODO: backpropagation (backward and step)
            loss.backward()
            optimizer.step()

            # get another batch
            batch_data = batch_iterator.get_next_batch()

        print("Avg loss: %.5f" % (batch_loss / batch_example_count))

        # evaluate on dev set
        model.eval() # turn on the "evaluation mode"
        acc, _, _, _ = evaluate(model, dev_exs, return_metrics=True)
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            print("Secure a new best accuracy %.3f in epoch %d!" % (best_acc, best_epoch))
            
            # save the current best model parameters
            print("Save the best model checkpoint as `best_model.ckpt`!")
            torch.save(model.state_dict(), "best_model.ckpt")
        print("-" * 10)

    # load back the best checkpoint on dev set
    model.load_state_dict(torch.load("best_model.ckpt"))
    
    model.eval() # switch to the evaluation mode
    return model
