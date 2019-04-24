from external.googlenet.googlenet import GoogLeNet
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import cv2
import torch
import pdb

class ImageFeatures(nn.Module):
    def __init__(self, num_classes=224):
        super(ImageFeatures, self).__init__()
        self.image_net = GoogLeNet(num_classes=224, transform_input=True)

    def forward(self, images):     
        image_features = self.image_net(images)
        return image_features
'''
class ParallelCoattention(nn.Module):
    def __init__(self, v_size, q_size, embed_size):
    super(ParallelCoattention, self).__init__()
        #Question encoding is B x 26 x 512
        #Image encoding is B x 196 x 512
'''                

class AlternatingCoattention(nn.Module):
    def __init__(self, v_size, q_size, embed_size, v_seq_size, q_seq_size):
        super(AlternatingCoattention, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.embed_size = embed_size
        self.v_seq_size = v_seq_size
        self.q_seq_size = q_seq_size

        # Q summary
        # Q embed is B x 26 x 512
        #self.q_lin = nn.Linear(q_size, embed_size)
        #self.tanh = nn.Tanh()
        #self.q_dropout = nn.Dropout(0.5)
        self.h1_lin = nn.Linear(embed_size, 1)  
        self.mask = nn.Softmax(dim=0)

        # Image attention based on Q summary
        self.linear_q_img = nn.Linear(q_size, embed_size)
        self.img_embed = nn.Linear(v_size, embed_size)
        self.drop_img = nn.Dropout(0.5)
        self.tanh_img = nn.Tanh()
        self.h2_lin = nn.Linear(embed_size, 1)
        self.h2_mask = nn.Softmax(dim=0)

        # Question attention based on Image attention
        self.linear_img_q = nn.Linear(v_size, embed_size)
        self.q_embed = nn.Linear(q_size, embed_size)
        self.drop_q = nn.Dropout(0.5)
        self.tanh_q = nn.Tanh()
        self.h3_lin = nn.Linear(embed_size, 1)
        self.h3_mask = nn.Softmax(dim=0)
         
    def forward(self, q, v):
        # TODO
        pdb.set_trace()
        #q_dim = q.view(self.q_size, -1)
        #v_dim = v.view(self.v_size, -1)
        # Summarize question into a single vector
        #q_lined = self.q_lin(self.q_dropout(q_dim)).view(self.q_seq_size, self.embed_size, -1)
        q_lined = self.q_lin(q)
        q_feat = self.tanh(q_lined).view(self.embed_size, -1)
        pdb.set_trace()
        H = self.h1_lin(q_feat).view(self.q_seq_size, -1)
        P = self.mask(H)
        P_att = P.view(P.size()[1])
        q_mul = P_att.matmul(q)
        q_summary = q_mul.view(self.q_size, -1)

        # Attend to the image based on question summary
        q_img_embed = self.linear_q_img(q_summary)
        q_img_embed = q_img_embed.unsqueeze(q_seq_size, 2)
        img_embed = self.img_embed(v_dim).view(-1, self.v_seq_size, self.embed_size)
        q_i_combo = torch.sum(img_embed, q_img_embed)
        img_feat = self.drop_img(self.tanh_img(q_i_combo)).view(-1, self.embed_size)
        H2 = self.h2_lin(img_feat).view(-1, self.v_seq_size)
        P2 = self.h2_mask(H2)
        P2_att = P2.view(P.size()[1])
        v_attention = P2_att.matmul(v).view(-1, self.v_size)
        
        # Attend the question based on attended image feature
        v_q_embed = self.linear_img_q(v_attention)
        v_q_embed = v_q_embed.unsqeeze(v_seq_size, 2)
        q_embed = self.q_embed(q_dim).view(-1, self.q_seq_size, self.embed_size)
        i_q_combo = torch.sum(q_embed, v_q_embed)
        ques_feat = self.drop_q(self.tanh_q(i_q_combo)).view(-1, self.embed_size)
        H3 = self.h3_lin(ques_feat).view(-1, self.q_seq_size)
        P3 = self.h3_mask(H3)
        P3_att = P2.view(P.size()[1])
        q_attention = P3_att.matmul(q).view(-1, self.q_size)
        
        attention = torch.concat(q_attention, v_attention)
        return attention

class CoattentionNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Hierarchical Question-Image Co-Attention
    for Visual Question Answering (Lu et al, 2017) paper.
    """
    def __init__(self, corpus_length=1000, max_question_length=26, embed_size=512):
        super(CoattentionNet, self).__init__()
        self.corpus_length = corpus_length
        self.max_question_length = max_question_length
        self.embed_size = embed_size
        filter_sizes = [1, 2, 3]
        num_filters = 512
        self.embedding = nn.Linear(corpus_length, embed_size)
        #self.embedding = nn.Embedding(max_question_length, embed_size, padding_idx=0)
        #self.conv1s = nn.ModuleList([nn.Conv2d(in_channels=max_question_length,out_channels=max_question_length,kernel_size=K,stride=1) for K in filter_sizes])
        self.uni_gram = nn.Sequential(
            nn.Conv1d(self.max_question_length, self.max_question_length, 1, padding=0),
            nn.Tanh()
        )
        self.bi_gram = nn.Sequential(
            nn.ConstantPad1d((0,1), 0),
            nn.Conv1d(self.max_question_length, self.max_question_length, 2, padding=0),
            nn.Tanh()
        )
        self.tri_gram = nn.Sequential(
            nn.Conv1d(self.max_question_length, self.max_question_length, 3, padding=1),
            nn.Tanh()
        )

        self.tanh1 = nn.Tanh()
        self.dropout1 = nn.Dropout(0.5)

        self.image_net = GoogLeNet(num_classes=self.corpus_length, transform_input=True)
         
        self.lstm = nn.LSTM(input_size=512, hidden_size=512)
        #self._init_weights(self.lstm.weight_ih_l0)
        #self._init_weights(self.lstm.weight_hh_l0)
        #self.lstm.bias_ih_l0.data.zero_()
        #self.lstm.bias_hh_l0.data.zero_()

        init.xavier_uniform_(self.embedding.weight)
 
        self.fc1 = nn.Linear(len(filter_sizes)*num_filters, 1)
        self.image_net = ImageFeatures(num_classes=embed_size)

        self.attention_word = AlternatingCoattention(v_size=512, q_size=max_question_length, embed_size=embed_size, v_seq_size=512, q_seq_size=512)
        self.attention_phrase = AlternatingCoattention(v_size=512, q_size=max_question_length, embed_size=embed_size, v_seq_size=512, q_seq_size=512)
        self.attention_sentence = AlternatingCoattention(v_size=512, q_size=max_question_length, embed_size=embed_size, v_seq_size=512, q_seq_size=512)

        self.tanh_word = nn.Tanh()
        self.tanh_phrase = nn.Tanh()
        self.tanh_sent = nn.Tanh()
      
        self.encode_word = nn.Linear(self.max_question_length, self.max_question_length)
        self.encode_phrase = nn.Linear(self.max_question_length*2, self.max_question_length)
        self.encode_ques = nn.Linear(self.max_question_length*2, self.max_question_length)
        self.probs = nn.Linear(self.max_question_length, self.max_question_length)

    def _init_weights(self, weight):
        for w in weight.chunk(4, 0):
            init.xavier_uniform_(w) 

    def forward(self, image, question_encoding, question_lengths):
        # TODO
        # image = B x 3 x 224 x 224
        # question_encoding = B x max_question_length x corpus_length
        # question_lengths = B x 1
        #Question Hierarchy
        image = image.cuda(async=True)
        word_embed = self.embedding(question_encoding.cuda(async=True))

        phrase_convs = []
        phrase_convs.append(self.uni_gram(word_embed))
        phrase_convs.append(self.bi_gram(word_embed))
        phrase_convs.append(self.tri_gram(word_embed))

        phrase_cat = torch.cat(phrase_convs, 2)
        phrase_embed = F.max_pool1d(phrase_cat, 3) 

        question_lengths = question_lengths.squeeze(1)
        sort_inds = np.argsort(question_lengths)
        phrase_sorted = torch.zeros(phrase_embed.size())
        actual_ind = 0
        for ind in sort_inds:
            phrase_sorted[ind] = phrase_embed[actual_ind]
            actual_ind += 1

        quest_embed, (_, _) = self.lstm(phrase_sorted.cuda(async=True))
         
        #Image Features
        # Returned as 512 x 14 x 14
        pdb.set_trace()
        #image_embed = self.image_net(image)
        image_embed = self.image_net(image)

        if self.training:
            image_embed = image_embed[-1]
        pdb.set_trace()

        image_embed = image_embed.view(image_embed.size()[0], image_embed.size()[1], -1)
        
        i2q_embed = self.img_to_q_dims(image_embed)
 
        pdb.set_trace() 
        att_word = self.attention_word(word_embed, i2q_embed)
        att_phrase = self.attention_phrase(phrase_embed, i2q_embed)
        att_sentence = self.attention_sentence(quest_embed, i2q_embed)
        pdb.set_trace() 
        #hidden_word = self.tanh_word(self.attention_word.weight * (q_att_word + v_att_word))
        #hidden_phrase = self.tanh_phrase(self.attention_phrase.weight * torch.cat((q_att_phrase + v_att_phrase), hidden_word))
        #hidden_ques = self.tanh_sent(self.attention_sentence.weight * torch.cat((q_att_sentence + v_att_sentence), hidden_phrase))
        hidden_word = self.encode_word(att_word)
        hidden_phrase_input = torch.cat(att_phrase, hidden_word)
        hidden_phrase = self.encode_phrase(hidden_phrase_indput)
        hidden_ques_input = torch.cat(att_sentence, hidden_phrase)
        hidden_ques = self.encode_ques(hidden_ques_input)
        probabilities = self.probs(hidden_ques)
 
        return probabilities


