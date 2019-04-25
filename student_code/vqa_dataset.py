from torch.utils.data import Dataset, dataloader
from external.googlenet.googlenet import GoogLeNet
from external.vqa.vqa import VQA
from PIL import Image
import torch.nn.functional as F
import h5py
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import cv2
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pickle as pkl
import string
from datetime import datetime
import pdb

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class ImageFeatures(nn.Module):
    def __init__(self, num_classes=1000, coattention=False):
        super(ImageFeatures, self).__init__()
        self.coattention = coattention
        self.image_net = GoogLeNet(num_classes=1000, coattention=coattention)

    def forward(self, images):     
        image_features = self.image_net(images)
        if self.coattention:
            image_features = image_features.view(image_features.size()[0], 196, 512)
            image_features = F._max_pool1d(image_features, kernel_size=512).squeeze(2)
        return image_features

#def accimage_loader(path):
#    import accimage
#    try:
#        print("accimage")
#        return accimage.Image(path)
#    except IOError:
#        return pil_loader(path)

#def default_loader(path):
#    from torchvision import get_image_backend
#    if get_image_backend() == 'accimage':
#        return accimage_loader(path)
#    else:
#        return pil_loader(path)

def separate(sentence):
    punctuation = [',', '?', '.', '!']
    for char in punctuation:
        if char in sentence:
            sentence = sentence.replace(char, '')
    sentence = sentence.lower()
    sentence_vec = sentence.split(' ')
    return sentence_vec

def add_to_corpus_freq(vocab_corpus, words, omit_words):
    for word in words:
        if word in omit_words:
            continue
        if word in vocab_corpus.keys():
            vocab_corpus[word] += 1
        else:
            vocab_corpus[word] = 1
    return vocab_corpus

def freq_to_encode_corpus(vocab_corpus):
    encode_corpus = {'nan': int(0)}
    index = int(1)
    for word_pair in vocab_corpus:
        if not word_pair[0] in encode_corpus.keys():
            encode_corpus[word_pair[0]] = int(index)
        index += int(1)
    return encode_corpus

def get_encoding(vocab_corpus, sentence):
    sentence = separate(sentence)
    encoded_sentence = []
    for word in sentence:
        if word in vocab_corpus.keys():
            index = vocab_corpus[word]
        else:
            index = 0
        encoded_sentence.append(index)
    return torch.tensor((np.asarray(encoded_sentence).astype(np.int)))

def pad_question(max_length, question):
    diff = max_length - len(question)
    zeros = torch.zeros([diff], dtype=torch.long)
    return torch.cat((question, zeros), 0)

def index_to_binary(corpus_length, question):
    question_encoded = np.zeros([corpus_length])
    for idx in question:
        question_encoded[idx] = 1
    return torch.tensor(question_encoded, dtype=torch.long)

class VqaDataset(Dataset):
    """
    Load the VQA dataset using the VQA python API. We provide the necessary subset in the External folder, but you may
    want to reference the full repo (https://github.com/GT-Vision-Lab/VQA) for usage examples.
    """

    def __init__(self, image_dir=None, question_json_file_path=None, annotation_json_file_path=None, image_filename_pattern=None, transform=None, omit_words=None, loaded_question_corpus=None, loaded_answer_corpus=None, best_answers_filepath=None, image_features_filepath=None, loaded_imgIdToidx_filepath=None, max_question_length=26, corpus_length=1000, model_type='simple'):
        """
        Args:
            image_dir (string): Path to the directory with COCO images
            question_json_file_path (string): Path to the json file containing the question data
            annotation_json_file_path (string): Path to the json file containing the annotations mapping images, questions, and
                answers together
            image_filename_pattern (string): The pattern the filenames of images in this dataset use (eg "COCO_train2014_{}.jpg")
        """
        
        self.image_dir = image_dir
        self.question_filepath = question_json_file_path
        self.annotation_filepath = annotation_json_file_path
        self.image_filename_pattern = image_filename_pattern

        self.transforms = transform
        self.omit_words = omit_words
        self.loaded_question_corpus = loaded_question_corpus
        self.loaded_answer_corpus = loaded_answer_corpus
        self.best_answers_filepath = best_answers_filepath
        self.loaded_image_features = image_features_filepath
        self.loaded_imgIdToidx_filepath = loaded_imgIdToidx_filepath

        self.max_question_length = max_question_length
        self.corpus_length = corpus_length
        self.num_debug_questions = None 
        self.dataset_type = ''
        self.model_type = model_type
        self.image_embed = None
 
        self.imgToQA = {}
        self.qIdToA = {}
        self.qIdToQA = {}
        
        self.question_corpus = {}
        self.answer_corpus = {}
        self.qIdToBestA = {}
        self.qIdList = []
        self.imgIdToidx = {}

        vqa = VQA(self.annotation_filepath, self.question_filepath)
        self.imgToQA = vqa.imgToQA
        self.qIdToA = vqa.qa
        self.qIdToQA = vqa.qqa
        
        if "COCO_train2014" in self.image_filename_pattern:
            self.image_prefix = "COCO_train2014_"
            self.dataset_type = "train"
        elif "COCO_val2014" in self.image_filename_pattern:
            self.image_prefix = "COCO_val2014_"
            self.dataset_type = "val"
        else:
            print('invalid image_file_pattern')
            exit(1)
 
       
        if omit_words is None:
            self.omit_words = ['the', 'a', 'so', 'her', 'him', 'very', 'an', 'this', 'does', 'will', 'i']
         
        if self.model_type == 'coattention':
            self.image_embed = 196
        else:
            self.image_embed = 1024      

 
        if self.transforms is None:
            img_size = (224, 224)
            #if self.model_type == 'coattention':
            #    img_size = (448, 448)
            #else:
            #    img_size = (224, 224)
            self.transforms = transforms.Compose([
                transforms.Resize(img_size), 
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
      
        date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        #Load Corpuses and Best Answers
        q_corpus = {}
        a_corpus = {}
        qIdToBestA = {}

        if self.loaded_question_corpus == None or self.loaded_answer_corpus == None or self.best_answers_filepath == None:
             
            #Make Question and Answer Corpuses
            qIds_to_add = self.qIdToA.keys()
            if self.num_debug_questions:
                qIds_to_add = [*self.qIdToA.keys()][0:self.num_debug_questions]
            for q_id in qIds_to_add:
                #Image debugging, don't use for final training
                #image_index = self.qIdToQA[q_id]['image_id']
                #if image_index not in self.imgIdToimg.keys():
                #    continue
         
                #Build question corpus by frequency 
                question = self.qIdToQA[q_id]['question']
                q_words = separate(question)
                q_corpus = add_to_corpus_freq(q_corpus, q_words, self.omit_words)
         
                #Build answer corpus by frequency and find distinct answers
                mc_answer = self.qIdToA[q_id]['multiple_choice_answer']
                mc_words = separate(mc_answer)
                a_corpus = add_to_corpus_freq(a_corpus, mc_words, self.omit_words)
                answers = self.qIdToA[q_id]['answers']
                distinct_answers = {}
                for ans_cluster in answers:
                    #Build answer corpus by frequency
                    answer = ans_cluster['answer']
                    a_words = separate(answer)
                    a_corpus = add_to_corpus_freq(a_corpus, a_words, self.omit_words)
                    if ans_cluster['answer_confidence'] != 'yes':
                        continue
         
                    #Build best answers
                    for a_word in a_words:
                        if a_word in distinct_answers.keys():
                            distinct_answers[a_word] += 1
                        else:
                            distinct_answers[a_word] = 1
                if not distinct_answers:
                    mc_words = separate(mc_answer)
                    for word in mc_words:
                        distinct_answers[word] = 1
                
                #Do majority vote across each question answer set
                best_ans = list(distinct_answers.keys())[0]
                best_ans_freq = distinct_answers[best_ans]
                for ans in distinct_answers.keys():
                    if distinct_answers[ans] > best_ans_freq:
                        best_ans = ans
                        best_ans_freq = distinct_answers[ans]
                #Build dictionary of best answers (question Id int to answer string)
                qIdToBestA[q_id] = best_ans
         
            #Sort question corpus by frequency and take top 1000
            q_corpus_sorted = sorted(q_corpus.items(), key=lambda kv: kv[1])
            if len(q_corpus_sorted) >= self.corpus_length:
                q_corpus = q_corpus_sorted[-(self.corpus_length-1):]
            else:
                self.corpus_length = len(q_corpus_sorted)
                q_corpus = q_corpus_sorted
         
            #Sort answer corpus by frequency and take top 1000
            a_corpus_sorted = sorted(a_corpus.items(), key=lambda kv: kv[1])
            if len(a_corpus_sorted) >= self.corpus_length:
                a_corpus = a_corpus_sorted[-(self.corpus_length-1):]
            else:
                a_corpus = a_corpus_sorted
         
            #Save to file question corpus, answer corpus, and best answers
            date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            
            self.loaded_question_corpus = './created_data/' + self.dataset_type + '_question_corpus_' + date_time + '.pkl'
            self.loaded_answer_corpus = './created_data/' + self.dataset_type + '_answer_corpus_' + date_time + '.pkl'
            self.best_answers_filepath = './created_data/' + self.dataset_type + '_best_answers_' + date_time + '.pkl'
            pkl.dump(q_corpus, open(self.loaded_question_corpus, 'wb'), pkl.HIGHEST_PROTOCOL)
            pkl.dump(a_corpus, open(self.loaded_answer_corpus, 'wb'), pkl.HIGHEST_PROTOCOL)
            pkl.dump(qIdToBestA, open(self.best_answers_filepath, 'wb'), pkl.HIGHEST_PROTOCOL)
           
        else:
            q_corpus = pkl.load(open(self.loaded_question_corpus, 'rb'))
            a_corpus = pkl.load(open(self.loaded_answer_corpus, 'rb'))
            qIdToBestA = pkl.load(open(self.best_answers_filepath, 'rb'))
        self.question_corpus = freq_to_encode_corpus(q_corpus) #Question words to index dictionary
        self.answer_corpus = freq_to_encode_corpus(a_corpus) #Answer words to index dictionary
        self.qIdToBestA = qIdToBestA #Question Id number to encoded answer dictionary
        self.qIdList = list(self.qIdToBestA.keys())
      
        self.qIdToEQ = {}
        self.qIdToEA = {}
        for q_id in self.qIdList:
            question = self.qIdToQA[q_id]['question']
            q_encoded = get_encoding(self.question_corpus, question)
            self.qIdToEQ[q_id] = q_encoded

            answer = self.qIdToBestA[q_id]
            a_encoded = get_encoding(self.answer_corpus, answer)
            self.qIdToEA[q_id] = a_encoded

        #Loade Image Features to h5py file
        if self.loaded_image_features==None:
            self.loaded_image_features = './created_data/' + self.dataset_type + '_image_features_' + self.model_type + '_' + date_time + '.hdf5'
            if self.model_type == 'coattention':
                google_feats = ImageFeatures(num_classes=self.corpus_length, coattention=True)
            else:
                google_feats = ImageFeatures(num_classes=self.corpus_length)
            im_type = self.image_filename_pattern.split('.')[-1]
            image_files = self.image_dir + '/' + self.image_prefix + '*.' + im_type
            num_images = len(glob.glob(image_files))

            with h5py.File(self.loaded_image_features, "w") as f:
                image_dataset = f.create_dataset('image_features', (num_images, self.image_embed), chunks=True, compression="gzip")
                image_num = 0
                start_num = 0
                image_batch = []
                for image_file in glob.glob(image_files):
                    im_index = image_file.split('.')[-2].split('_')[-1]
                    im_index = im_index.lstrip('0')
                    loaded_img = pil_loader(image_file)
                    
                    if loaded_img == None:
                        print("Image not loaded correctly")
                        return exit(1)
                    image_trans = self.transforms(loaded_img)
                    image_batch.append(image_trans)
                    self.imgIdToidx[im_index] = image_num
                    image_num += 1
                    if image_num % 300 == 0 or image_num == num_images:
                        print("Loading Images: ", start_num, " to ", image_num)
                        batch_tensor = torch.stack(image_batch)
                        image_features = google_feats(batch_tensor) # Shape B x 196 for Coattention, B x 1024 for Simple
                        dataset_ims = image_features.detach().cpu().numpy()
                        image_dataset[start_num:image_num, :] = dataset_ims
                        start_num = image_num
                        image_batch = []
                self.loaded_imgIdToidx_filepath = './created_data/' + self.dataset_type + '_imgIdToidx_' + date_time + '.pkl'
                pkl.dump(self.imgIdToidx, open(self.loaded_imgIdToidx_filepath, 'wb'), pkl.HIGHEST_PROTOCOL)
 

        if self.loaded_imgIdToidx_filepath == None:
            image_num = 0
            for image_file in glob.glob(image_files):
                im_index = image_file.split('.')[-2].split('_')[-1]
                im_index = im_index.lstrip('0')
                self.imgIdToidx[im_index] = image_num
                image_num += 1
            self.loaded_imgIdToidx_filepath = './created_data/' + self.dataset_type + '_imgIdToidx_' + date_time + '.pkl'
            pkl.dump(q_corpus, open(self.loaded_imgIdToidx_filepath, 'wb'), pkl.HIGHEST_PROTOCOL)
        else:
            self.imgIdToidx = pkl.load(open(self.loaded_imgIdToidx_filepath, 'rb'))
                 
        
    def __len__(self):
        return len(self.qIdList)

    def __getitem__(self, idx):
        q_id = self.qIdList[idx]
        im_id = self.qIdToQA[q_id]['image_id']
        #question = self.qIdToQA[q_id]['question']
        #question_encoded = get_encoding(self.question_corpus, question)
        question_encoded = self.qIdToEQ[q_id]
        if self.model_type == 'coattention':
            question_padded = pad_question(self.max_question_length, question_encoded)
            question_output = torch.zeros([self.max_question_length, len(self.question_corpus)])
            for index in range(self.max_question_length):
                question_output[index, question_padded[index]] = 1
        else:
            question_output = index_to_binary(len(self.question_corpus), question_encoded)
        #answer = self.qIdToBestA[q_id]
        #answer_encoded = get_encoding(self.answer_corpus, answer)
        answer_encoded = self.qIdToEA[q_id]
        #answer_binary = index_to_binary(self.corpus_length, answer_encoded)
        question_length = torch.tensor([len(question_encoded)])
      
        img_idx = self.imgIdToidx[str(im_id)]
        image_features = h5py.File(self.loaded_image_features, 'r') 
        #with h5py.File(self.loaded_image_features) as image_features:
        image_feature = image_features['image_features'][img_idx,:]
        if self.model_type == 'coattention':
            image_resize = torch.Tensor(image_feature).view(196)
        else:
            image_resize = torch.Tensor(image_feature).view(1024)

        #im_type = self.image_filename_pattern.split('.')[-1]
        #im_access_id = '0'*(12-len(str(im_id))) + str(im_id)
        #image_file = self.image_dir + '/' + self.image_prefix + im_access_id + '.' + im_type
        #loaded_img = pil_loader(image_file)
        #if loaded_img == None:
        #    print("Image not loaded correctly")
        #    return exit(1)
        
        #image = self.transforms(loaded_img)
        return {'question':question_output, 'image':image_resize, 'answer':answer_encoded, 'question_length':question_length}
    
