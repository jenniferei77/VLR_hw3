import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.init as init
from external.googlenet.googlenet import GoogLeNet
import pdb

class SimpleBaselineNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Simple Baseline for Visual Question Answering (Zhou et al, 2017) paper.
    """
    def __init__(self, num_classes=1000, embedding_dim=30):
        super(SimpleBaselineNet, self).__init__()
        #Features:
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
      
        #self.word_feats = nn.Sequential(
        #    nn.Embedding(self.num_classes, self.embedding_dim, padding_idx=0),
        #    nn.Linear(self.num_classes*embedding_dim, 128),
        #    nn.Linear(128, self.num_classes),
        #)
        self.word_feats = WordFeatures(
            question_max=self.embedding_dim,
            question_lengths=torch.tensor([]),
            corpus_length=self.num_classes,
        )

        self.image_feats = GoogLeNet()

        #Classify:
        self.classifier = nn.Sequential(
            nn.Linear(self.num_classes*2, self.num_classes),
            nn.Softmax(),
        )

    def forward(self, image, question_encoding, question_length):
        # TODO
        pdb.set_trace()
        word_features = self.word_feats(len(question_encoding[0]), question_length)
        
        word_features = word_features.view(word_features.size()[0], -1)
        image_features = self.image_cnn(image)
        image_features = image_features.view(image_features.size()[0], -1)
        
        full_features = torch.cat((word_features, image_features), 0)
        
        score = self.softmax(full_features)
       
        return score 

class WordFeatures(nn.Module):
    def __init__(self, question_max, question_lengths, corpus_length=1000):
    #def __init__(self, question_max, corpus_length=1000):
        super(WordFeatures, self).__init__()
        pdb.set_trace()
        self.word_embedding = nn.Embedding(question_max, corpus_length, padding_idx=0)
        #self.dropout = nn.Dropout(0.5)
        self.layer1 = nn.Linear(corpus_length*question_max, 128),
        self.layer2 = nn.Linear(128, corpus_length)
        pdb.set_trace() 
        init.xavier_uniform_(self.word_embedding.weight.data)
        #init.xavier_uniform(self.layer1.weight)
        #init.xavier_uniform(self.layer2.weight)

    def forward(self, question_indices, question_lengths):
        question_indices = question_indices.type(torch.LongTensor) 
  
        pdb.set_trace()
        embedding = self.word_embedding(question_indices)
        un_padded = pack_padded_sequence(embedding, question_lengths, batch_first=True)
        mid_layer = self.layer1(un_padded)
        word_features = self.layer2(mid_layer)
        return word_features.squeeze(0)
   
