import torch.nn as nn


class CoattentionNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Hierarchical Question-Image Co-Attention
    for Visual Question Answering (Lu et al, 2017) paper.
    """
    def __init__(self, corpus_length=1000, max_question_length=30, embed_size=1024):
        super(CoattentionNet, self).__init__()
        self.corpus_length = corpus_length
        self.embed_size = embed_size
        filter_sizes = [1, 2, 3]
        num_filters = 64

        self.embedding = nn.Embedding(max_question_length, embed_size, padding_idx=0)
        self.conv1s = nn.ModuleList([nn.Conv2d(1, num_filers, (K, embed_size)) for K in filter_sizes])
        self.tanh1 = nn.Tanh()
        self.dropout1 = nn.Dropout(0.5)
         
        self.lstm = nn.LSTM(input_size=max_question_length, hidden_size=embed_size, num_layers=1)
        

        self.fc1 = nn.Linear(len(filter_sizes)*num_filters, 1)

        self.softmax = nn.Softmax(dim=1)
 
 
    def forward(self, image, question_encoding, question_lengths):
        # TODO
        pdb.set_trace()
        word_embed = (self.embedding(question_encoding)).unsqueeze(1)
        reg = self.tanh(self.drop(word_embed))
        phrase_embed = [F.relu(conv1(reg)).squeeze(3) for conv1 in self.conv1s]
        phrase_embed = [F.max_pool1d(word, word.size(2)).squeeze(2) for word in phrase_embed]
        pack = pack_padded_sequence(phrase_embed, question_lengths, batch_first=True)
        _, (_, quest_embed) = self.lstm(pack) 

        


        raise NotImplementedError()
