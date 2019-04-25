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
        self.corpus_length = corpus_length

        #Word and Image Features:
        self.lin_word_net = nn.Linear(self.corpus_length, 1024)
#        self.image_net = GoogLeNet(num_classes=self.corpus_length, transform_input=True)

        #Classify:
        self.classifier = nn.Linear(2048, self.corpus_length)

    def forward(self, image_feature, question_encodings, question_lengths):
        # TODO
        #image = image.cuda(async=True)
        word_features = self.lin_word_net(question_encodings.cuda(async=True))
        #image_features = self.image_net(image)
        #if self.training:
        #    image_features = image_features[-1]
        image_features = image_features.cuda(async=True)
        full_features = torch.cat((word_features, image_feature), dim=1)
        
        score = self.classifier(full_features)
        return score

   
