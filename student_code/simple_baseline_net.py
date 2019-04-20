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
    def __init__(self, corpus_length=1000):
        super(SimpleBaselineNet, self).__init__()
        #Features:
        self.corpus_length = corpus_length
      
        #self.word_feats = WordFeaturesIndexed(
        #    max_question_length=self.max_question_length,
        #    question_lengths=1000,
        #    corpus_length=self.corpus_length
        #)

        #self.word_embedding = nn.Embedding(self.max_question_length, self.corpus_length)
        self.lin_word_net = nn.Linear(self.corpus_length, self.corpus_length)
        #self.embed_word_net = nn.Embedding(self.max_question_length, self.corpus_length)
        
        #self.word_net = WordFeaturesBinary(
        #    max_question_length=self.max_question_length,
        #    corpus_length=self.corpus_length
        #)

        self.image_net = GoogLeNet(aux_logits=False)
        #self.flatten_image = nn.Linear(3*self.corpus_length, self.corpus_length)

        #Classify:
        self.classifier = nn.Sequential(
            nn.Linear(self.corpus_length*2, self.corpus_length),
            nn.Softmax(dim=0),
        )

    def forward(self, image, question_encodings, question_lengths):
        # TODO
        #question_encodings = question_encodings.cuda(async=True)
        image = image.cuda(async=True)
        word_features = self.lin_word_net(question_encodings.type(torch.FloatTensor).cuda(async=True))
        #word_features = self.word_net(question_encodings)
        
        #word_features = word_features.view(word_features.size()[0], -1)
        image_features = self.image_net(image)
        #image_features = torch.stack(image_features)
        #image_features = image_features.cuda(async=True)
        #image_features = image_features.view(image_features.size()[0], -1)
        #image_flat_features = self.flatten_image(image_features)
        full_features = torch.cat((word_features, image_features), dim=1)
        
        score = self.classifier(full_features)
        return torch.clamp(score, 0, 1)

class WordFeaturesBinary(nn.Module):
    def __init__(self, max_question_length, corpus_length):
        super(WordFeaturesBinary, self).__init__()
        self.word_embedding = nn.Embedding(max_question_length, corpus_length)
        self.layer1 = nn.Linear(300, 128)
        self.layer2 = nn.Linear(128, corpus_length)
        init.xavier_uniform_(self.word_embedding.weight.data)

    def forward(self, question_encodings):
        embedding = self.word_embedding(question_encodings)
        mid_layer = self.layer1(embedding)
        word_features = self.layer2(mid_layer)
        return word_features.squeeze(0)

class WordFeaturesIndexed(nn.Module):
    def __init__(self, max_question_length, question_lengths, corpus_length):
    #def __init__(self, question_max, corpus_length=1000):
        super(WordFeatures, self).__init__()
        #question_lengths is list of each question_length
        self.word_embedding = nn.Embedding(max_question_length, question_lengths, padding_idx=0)
        #self.dropout = nn.Dropout(0.5)
        self.layer1 = nn.Linear(corpus_length, 128),
        self.layer2 = nn.Linear(128, corpus_length)
        init.xavier_uniform_(self.word_embedding.weight.data)
        #init.xavier_uniform(self.layer1.weight)
        #init.xavier_uniform(self.layer2.weight)

    def forward(self, question_encodings, question_lengths):
        question_encodings = question_encodings.type(torch.LongTensor) 
  
        embedding = self.word_embedding(question_encodings)
        un_padded = pack_padded_sequence(embedding, question_lengths, batch_first=True)
        mid_layer = self.layer1(un_padded)
        word_features = self.layer2(mid_layer)
        return word_features.squeeze(0)
   
