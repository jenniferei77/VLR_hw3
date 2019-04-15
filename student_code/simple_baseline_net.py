import torch.nn as nn
from external.googlenet.googlenet import GoogLeNet


class SimpleBaselineNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Simple Baseline for Visual Question Answering (Zhou et al, 2017) paper.
    """
    def __init__(self, embedding_dim=5, num_classes=1000):
        super(SimpleBaselineNet, self).__init__()
        #Features:
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.word_feats = nn.Embedding(self.num_classes, self.embedding_dim)
        self.image_cnn = GoogLeNet()

        #Classify:
        self.classifier = nn.Linear(in_features=self.num_classes*2, self.num_classes)
        self.softmax = nn.Softmax()

    def forward(self, image, question_encoding):
        # TODO
        word_features = self.word_feats(question_encoding)
        word_features = word_features.view(word_features.size()[0], -1)
        image_features = self.image_cnn(image)
        image_features = image_features.view(image_features.size()[0], -1)
        
        full_features = torch.cat((word_features, image_features), 0)
        
        score = self.softmax(full_features)
       
        return score 
