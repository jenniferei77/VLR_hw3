from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import cv2
import torch
from external.googlenet.googlenet import GoogLeNet
import pdb

def softmax_mask(H):
    H_max = torch.max(H, dim=1, keepdim=True)[0]
    H_exp = torch.exp(H-H_max)
    H_mask = H_exp * (H_exp != 1.000).type(torch.FloatTensor).cuda(async=True)
    H_softmax = H_mask / torch.sum(H_mask, dim=1, keepdim=True)
    return H_softmax

class ParallelCoattention(nn.Module):
    def __init__(self, v_size, q_size, embed_size, batch_size):
        super(ParallelCoattention, self).__init__()
        #Question encoding is B x 26 x 512
        #Image encoding is B x 196 x 512
        self.batch_size = batch_size
        self.v_size = v_size
        self.q_size = q_size
        self.embed_size = embed_size

        self.word_map = nn.Linear(embed_size, embed_size)
        self.image_map = nn.Linear(embed_size, embed_size)

        init.xavier_uniform_(self.word_map.weight)
        init.xavier_uniform_(self.image_map.weight)

        self.Hv_map = nn.Linear(embed_size, 1)
        self.Hq_map = nn.Linear(embed_size, 1)
        #self.qhat_resize = nn.Linear(26, 196)

        init.xavier_uniform_(self.Hv_map.weight)
        init.xavier_uniform_(self.Hv_map.weight)


    def forward(self, Q, V, question_lengths):
        #Q=Bx26x512
        #V=Bx196x512
        #Q_Wb = self.word_map(Q.view(self.batch_size, self.embed_size*self.q_size)) # Should be B x 26 x 512
        Q_Wb = self.word_map(Q) # B x 26 x 512
        Q_Wb_V = torch.matmul(Q_Wb, V.transpose(1,2)) # Should be B x 26 x 196
        C = torch.tanh(Q_Wb_V) # Should be B x 26 x 196

        WvV = self.image_map(V) # Should be B x 196 x 512
        Hv = torch.tanh(WvV + torch.matmul(Q_Wb.transpose(1,2), C).transpose(1,2)) # Should be B x 196 x 512 + (B x 512 x 26)(B x 26 x 196) = B x 196 x 512
        Hq = torch.tanh(Q_Wb + torch.matmul(C, WvV)) # Should be B x 512 x 26 + (B x 26 x 196)(B x 196 x 512) = B x 26 x 512
        #Hv_resize = self.Hv_resize(Hv.transpose(1,2)).transpose(1,2)
        wHv = self.Hv_map(Hv).squeeze(2) # Should be B x 196
        wHq = self.Hq_map(Hq).squeeze(2) # Should be B x 26
        #av = softmax_mask(wHv).unsqueeze(2) # Should be B x 196 x 1
        #aq = softmax_mask(wHq).unsqueeze(2) # Should be B x 26 x 1
        av = torch.softmax(wHv, dim=1).unsqueeze(2)
        aq = torch.softmax(wHq, dim=1).unsqueeze(2)

        #v_hat = torch.sum((av * V), dim=2)
        #q_hat = torch.sum((aq * Q), dim=2)
        #q_hat_resize = self.qhat_resize(q_hat) #resize to be 196 x 512
        v_hat = torch.matmul(av.transpose(1,2), V).squeeze(1) # B x 512
        q_hat = torch.matmul(aq.transpose(1,2), Q).squeeze(1) # B x 512
        return q_hat + v_hat
        
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
        #q_dim = q.view(self.q_size, -1)
        #v_dim = v.view(self.v_size, -1)
        # Summarize question into a single vector
        #q_lined = self.q_lin(self.q_dropout(q_dim)).view(self.q_seq_size, self.embed_size, -1)
        q_lined = self.q_lin(q)
        q_feat = self.tanh(q_lined).view(self.embed_size, -1)
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
'''

class CoattentionNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Hierarchical Question-Image Co-Attention
    for Visual Question Answering (Lu et al, 2017) paper.
    """
    def __init__(self, corpus_length=1000, max_question_length=26, embed_size=512, batch_size=100):
        super(CoattentionNet, self).__init__()
        self.corpus_length = corpus_length
        self.max_question_length = max_question_length
        self.embed_size = embed_size
        self.batch_size = batch_size

        self.embedding = nn.Linear(corpus_length, embed_size)
        #self.image_embed = nn.Linear(196, 196*512)
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

        self.lstm = nn.LSTM(input_size=512, hidden_size=512)
        self._init_weights(self.lstm.weight_ih_l0)
        self._init_weights(self.lstm.weight_hh_l0)
        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()

        init.xavier_uniform_(self.embedding.weight)
        
        self.image_net = GoogLeNet(num_classes=corpus_length, transform_input=True, coattention=True) 
        self.attention_word = ParallelCoattention(v_size=196, 
                                                  q_size=max_question_length, 
                                                  embed_size=embed_size, 
                                                  batch_size=self.batch_size)
        self.attention_phrase = ParallelCoattention(v_size=196, 
                                                    q_size=max_question_length, 
                                                    embed_size=embed_size, 
                                                    batch_size=self.batch_size)
        self.attention_sentence = ParallelCoattention(v_size=196, 
                                                      q_size=max_question_length, 
                                                      embed_size=embed_size, 
                                                      batch_size=self.batch_size)
      
        #self.encode_word = nn.Linear(self.max_question_length, 
        #                             self.max_question_length)
        #self.encode_phrase = nn.Linear(self.max_question_length, 
        #                               self.max_question_length)
        #self.encode_ques = nn.Linear(self.max_question_length, 
        #                             self.max_question_length)
        #self.probs = nn.Linear(self.max_question_length, 
        #                       self.max_question_length)

        self.encode_word = nn.Linear(512, 512)
        self.encode_phrase = nn.Linear(512*2, 512*2)
        self.encode_ques = nn.Linear(512*3, 512*3)
        self.probs = nn.Linear(512*3, corpus_length)


    def _init_weights(self, weight):
        for w in weight.chunk(4, 0):
            init.xavier_uniform_(w) 

    def forward(self, image_feature, question_encoding, question_lengths):
        # TODO
        # image = B x 196
        # question_encoding = B x max_question_length x corpus_length
        # question_lengths = B x 1
        #Question Hierarchy
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
        image_embed = self.image_net(image_feature.cuda(async=True))
        image_embed = image_embed.view(image_embed.size()[0], image_embed.size()[1], -1).transpose(1,2)
        # image_embed = B x 196 x 512
        #image_embed = self.image_embed(image_feature.cuda(async=True))
        #image_embed = image_embed.view(image_embed.size()[0], 196, 512) # Reshape to B x 196 x 512
        att_word = self.attention_word(word_embed, image_embed, question_lengths) # Shape is B x 26
        att_phrase = self.attention_phrase(phrase_embed, image_embed, question_lengths) # Shape is B x 26
        att_sentence = self.attention_sentence(quest_embed, image_embed, question_lengths) # Shape is B x 26
        #hidden_word = self.tanh_word(self.attention_word.weight * (q_att_word + v_att_word))
        #hidden_phrase = self.tanh_phrase(self.attention_phrase.weight * torch.cat((q_att_phrase + v_att_phrase), hidden_word))
        #hidden_ques = self.tanh_sent(self.attention_sentence.weight * torch.cat((q_att_sentence + v_att_sentence), hidden_phrase))
        hidden_word = self.encode_word(att_word)
        hidden_phrase_input = torch.cat((att_phrase, hidden_word), dim=1)
        hidden_phrase = self.encode_phrase(hidden_phrase_input)
        hidden_ques_input = torch.cat((att_sentence, hidden_phrase), dim=1)
        hidden_ques = self.encode_ques(hidden_ques_input)

        probabilities = self.probs(hidden_ques)
 
        return probabilities


