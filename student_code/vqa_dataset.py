from torch.utils.data import Dataset, dataloader
from external.vqa.vqa import VQA
from PIL import Image
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
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        print("accimage")
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

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
    question_encoded[question] = 1
    return question_encoded

class VqaDataset(Dataset):
    """
    Load the VQA dataset using the VQA python API. We provide the necessary subset in the External folder, but you may
    want to reference the full repo (https://github.com/GT-Vision-Lab/VQA) for usage examples.
    """

    def __init__(self, image_dir=None, question_json_file_path=None, annotation_json_file_path=None, image_filename_pattern=None, transform=None, omit_words=None, loaded_question_corpus=None, loaded_answer_corpus=None, best_answers_filepath=None, max_question_length=30, corpus_length=1000):
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
        self.loaded_annotation_corpus = loaded_answer_corpus
        self.loaded_best_answers = best_answers_filepath
        self.max_question_length = max_question_length
        self.corpus_length = corpus_length
        self.num_debug_images = 500
        
        self.imgIdToimg = {}
        self.imgToQA = {}
        self.qIdToA = {}
        self.qIdToQA = {}
        
        self.question_corpus = {}
        self.answer_corpus = {}
        self.qIdToBestA = {}

        vqa = VQA(self.annotation_filepath, self.question_filepath)
        self.imgToQA = vqa.imgToQA
        self.qIdToA = vqa.qa
        self.qIdToQA = vqa.qqa
        
       
        if omit_words is None:
            self.omit_words = ['the', 'a', 'so', 'her', 'him', 'very', 'an', 'this', 'does', 'will', 'i']
       
        if self.transforms is None:
            self.transforms = transforms.Compose([
                transforms.Resize((224, 224)), 
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
      
        #Load images to memory 
        self.imgIdToImg = LoadImages(self) 

        #Load Corpuses and Best Answers
        if self.loaded_question_corpus == None or self.loaded_answer_corpus == None or self.best_answers_filepath == None:
            
            #Make Question and Answer Corpuses
            q_corpus, a_corpus, qIdToBestA = CreateCorpusAndAnswers(self)

            #Save to file question corpus, answer corpus, and best answers
            date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            self.loaded_question_corpus = './created_data/question_corpus_' + date_time + '.pkl'
            self.loaded_answer_corpus = './created_data/answer_corpus_' + date_time + '.pkl'
            self.best_answers_filepath = './created_data/best_answers' + date_time + '.pkl'
            pkl.dump(q_corpus, open(self.loaded_question_corpus, 'wb'), pkl.HIGHEST_PROTOCOL)
            pkl.dump(a_corpus, open(self.loaded_answer_corpus, 'wb'), pkl.HIGHEST_PROTOCOL)
            pkl.dump(qIdToBestA, open(self.best_answers_filepath, 'wb'), pkl.HIGHEST_PROTOCOL)
           
        else:
            q_corpus = pkl.load(open(self.question_corpus_filepath, 'rb'))
            a_corpus = pkl.load(open(self.answer_corpus_filepath, 'rb'))
            qIdToBestA = pkl.load(open(self.best_answers_filepath, 'rb'))
            
        self.q_corpus = freq_to_encode_corpus(q_corpus) #Question words to index dictionary
        self.a_corpus = freq_to_encode_corpus(a_corpus) #Answer words to index dictionary
        self.qIdToBestA = qIdToBestA #Question Id number to encoded answer dictionary
         
    def __len__(self):
        return len(self.qIdToBestA)

    def __getitem__(self, idx):
        vqa = self.VQA
        q_ids = list(self.qIdToBestA.keys())
        q_id = q_ids[idx]
        im_id = self.qIdToQA['image_id']
        question = self.qIdToQA['question']
        #print("Actual Question: ", question)
        question_encoded = get_encoding(self.question_corpus, question)
        #question_padded = pad_question(self.max_question_length, question_encoded)
        question_binary = index_to_binary(self.corpus_length, question_encoded)
        image = self.imgIdToimg.get(im_id)
        answer = self.qIdToBestA[q_id]
        #print ("Actual Answer: ", answer)
        answer_encoded = get_encoding(self.answer_corpus, answer)
        print("Image: ", image.shape)
        print("Question: ", question_encoded)
        print("Answer: ", answer_encoded)
        question_length = torch.tensor(len(question_encoded))
        #pdb.set_trace()
        return question_binary, image, answer_encoded, question_length
    
def LoadImages(self):
   if not self.image_dir == None:
      print('Loading COCO images and question data from ', self.image_dir)
      images = {}

   if "COCO_train2014" in self.image_filename_pattern:
       images_prefix = "COCO_train2014_"
   elif "COCO_val2014" in self.image_filename_pattern:
       images_prefix = "COCO_val2014_"
   else:
       print('imvalid image_file_pattern')
       exit(1)
      
   try:
       im_type = self.image_filename_pattern.split('.')[-1]
       image_files = self.image_dir + '/' + images_prefix + '*.' + im_type
       #print("Image Files: ", image_files)
       num_images = 0
       for filename in glob.glob(image_files):
           #print("Image Filename: ", filename)
           num_images += 1
          
           #Number of debug images to test with
           if num_images % 100 == 0:
               print(num_images)
           if num_images > self.num_debug_images:
               break

           #Create image Id for indexing
           im_index = filename.split('.')[-2].split('_')[-1]
           im_index = im_index.lstrip('0')
           
           #Load image and put in dictionary
           loaded_img = default_loader(filename)
           if loaded_img is not None:
               image = self.transforms(loaded_img)
               images[int(im_index)] = image
               #print("Image loaded correctly with index", im_index)
           else:
               print("Image not loaded correctly")
       return images
   except OSError:
       print('image_dir is not a real directory')
       return -1

def CreateCorpusAndAnswers(self):
    q_corpus = {}
    a_corpus = {}
    qIdToBestA = {}
    
    for q_id in self.qIdToA.keys():
        #Image debugging, don't use for final training
        image_index = self.qIdToQA[q_id]['image_id']
        if image_index not in self.imgIdToimg.keys():
            continue
 
        #Build question corpus by frequency 
        question = self.qIdToQA[q_id]['question']
        q_words = separate(question)
        q_corpus = add_to_corpus_freq(q_corpus, q_words, self.omit_words)
 
        #Build answer corpus by frequency and find distinct answers
        answers = self.qIdToA[q_id]['answers']
        distinct_answers = {}
        for ans_cluster in answers:
            answer = ans_cluster['answer']
            if ans_cluster['answer_confidence'] != 'yes':
                continue
 
            #Build answer corpus by frequency
            a_words = separate(answer)
            a_corpus = add_to_corpus_freq(a_corpus, a_words, self.omit_words)
 
            #Build best answers
            for a_word in a_words:
                if a_word in distinct_answers.keys():
                    distinct_answers[a_word] += 1
                else:
                    distinct_answers[a_word] = 1
        if not distinct_answers:
            mc_answer = vqa.qa[q_id]['multiple_choice_answer']
            distinct_answers[mc_answer] = 1
        
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
        q_corpus = q_corpus_sorted[-self.corpus_length:]
    else:
        q_corpus = q_corpus_sorted
 
    #Sort answer corpus by frequency and take top 1000
    a_corpus_sorted = sorted(a_corpus.items(), key=lambda kv: kv[1])
    if len(a_corpus_sorted) >= self.corpus_length:
        a_corpus = a_corpus_sorted[-self.corpus_length:]
    else:
        a_corpus = a_corpus_sorted
 
    return q_corpus, a_corpus, qIdToBestA 
