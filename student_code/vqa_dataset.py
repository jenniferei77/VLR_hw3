from torch.utils.data import Dataset
from external.vqa.vqa import VQA
from PIL import Image
import os
import glob
import numpy as numpy
import torch
import torch.nn as nn
import cv2
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pickle as pkl
import pdb

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
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
    sentence.translate(None, string.punctuation)
    sentence_vec = sentence.split(' ')
    return sentence_vec

def add_to_vocab(vocab_corpus, words):
    for word in words:
        if word in vocab_corpus.keys():
            continue
        else:
            vocab_corpus[word] = len(vocab_corpus) + 1
    return vocab_corpus

def add_to_corpus_freq(vocab_corpus, words):
    for word in words:
        if word in vocab_corpus.keys():
            vocab_corpus[word] += 1
        else:
            vocab_corpus[word] = 1
    return vocab_corpus

def freq_to_encode_corpus(vocab_corpus):
    encode_corpus = {'nan': 0}
    index = 1
    for word in vocab_corpus.keys():
        if not word in encode_corpus.keys():
            encode_corpus[word] = index
        index += 1
    return encode_corpus

def get_encoding(vocab_corpus, sentence):
    encoded_sentence = []
    omit_count = 0
    for word in sentence:
        if word in vocab_corpus.keys():
            index = vocab_corpus[word]
            encoded_sentence.append(index)
        else:
            index = 0
        encoded_sentence.append(index)
    if omit_count == len(sentence):
        return torch.Tensor([])
    else:
        return torch.Tensor(encoded_sentence)

class VqaDataset(Dataset):
    """
    Load the VQA dataset using the VQA python API. We provide the necessary subset in the External folder, but you may
    want to reference the full repo (https://github.com/GT-Vision-Lab/VQA) for usage examples.
    """

    def __init__(self, image_dir=None, question_json_file_path=None, annotation_json_file_path=None, image_filename_pattern=None, transform=None, omit_words=None):
        """
        Args:
            image_dir (string): Path to the directory with COCO images
            question_json_file_path (string): Path to the json file containing the question data
            annotation_json_file_path (string): Path to the json file containing the annotations mapping images, questions, and
                answers together
            image_filename_pattern (string): The pattern the filenames of images in this dataset use (eg "COCO_train2014_{}.jpg")
        """
        if "COCO_train2014_" in image_filename_pattern:
            images_prefix = "COCO_train2014_"
        elif "COCO_eval2014" in image_filename_pattern:
            images_prefix = "COCO_eval2014_"
        else:
            print('imvalid image_file_pattern')
            exit(1)
        
        if omit_words is None:
            self.omit_words = ['the', 'a', 'so', 'her', 'him', 'very', 'an', 'this', 'are', 'does', 'will', 'i']
        else:
            self.omit_words = omit_words
       
        if transform is not None:
            self.transforms = transform
        im_type = image_filename_pattern.split('.')[-1]

        self.images = {}
        
        self.image_dir = image_dir
        vqa = VQA(annotation_json_file_path, question_json_file_path)
        self.VQA = vqa

        if not image_dir == None:
            print('Loading COCO images and question data')
            images = {}
            
            try:
                image_files = image_dir + images_prefix + '*.' + im_type
                for filename in glob.glob(image_files):
                    im_index = image_files.split('.')[-2].split('_')[-1]
                    im_index = im_index.lstrip('0')
                    loaded_img = default_loader(filename)
                    if loaded_img is not None:
                        image = self.transforms(loaded_img)
                        images[im_index] = image
                self.images = images
            except OSError:
                print('image_dir is not a real directory')

        if q_filepath == None or a_filepath == None or a_best_filepath == None:
            question_corpus = {}
            answer_corpus = {}
            q_ids = vqa.getQuesIds()
            qIdToAns = {}
            qIdToEncoding = {}
            for q_id in q_ids:
                #Build question corpus by frequency 
                question = vqa.qqa[q_id]['question']
                q_words = separate(question)
                question_corpus = add_to_corpus_freq(q_words)
                answers = vqa.qa[q_id]['answers']
                distinct_answers = {}
                for ans_cluster in answers:
                    answer = ans_cluster['answer']
                    if ans_cluster['confident'] != 'yes':
                        continue
                    #Build answer corpus by frequency
                    a_words = separate(answer)
                    answer_corpus = add_to_corpus_freq(a_words)
                    for a_word in a_words:
                        if a_word in self.omit_words:
                            continue
                        if a_word in distinct_answers.keys():
                            distinct_answers[a_word] += 1
                        else:
                            distinct_answers[answer] = 1
                if not distinct_answers:
                    mc_answer = vqa.qa[q_id]['multiple_choice_answer']
                    distinct_answers[mc_answer] = 1
                #Do majority vote across each question answer set
                best_ans = distinct_answers.keys()[0]
                best_ans_freq = distinct_answers[best_ans]
                for ans in distinct_answers.keys():
                    if distinct_answers[ans] > best_ans_freq:
                        best_ans = ans
                        best_ans_freq = distinct_answers[ans]
                qIdToAns[q_id] = best_ans

            #Sort question corpus by frequency and take top 1000
            q_corpus_sorted = sorted(q_corpus.keys(), key=lambda kv: kv[1])
            if len(q_corpus_sorted >= 1000):
                q_corpus = q_corpus_sorted[-1000:]
            else:
                q_corpus = q_corpus_sorted

            #Sort answer corpus by frequency and take top 1000
            a_corpus_sorted = sorted(a_corpus.keys(), key=lambda kv: kv[1])
            if len(a_corpus_sorted >= 1000):
                a_corpus = a_corpus_sorted[-1000:]
            else:
                a_corpus = a_corpus_sorted
            
            date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            q_filepath = 'question_corpus' + date_time + '.pkl'
            a_filepath = 'answer_corpus' + date_time + '.pkl'
            best_a_filepath = 'best_answers' + date_time + '.pkl'
            pkl.dump(q_corpus, open(q_filepath, 'wb'), pkl.HIGHEST_PROTOCOL)
            pkl.dump(a_corpus, open(a_filepath, 'wb'), pkl.HIGHEST_PROTOCOL)
            pkl.dump(qIdToAns, open(best_a_filepath, 'wb'), pkl.HIGHEST_PROTOCOL)
            
        else:
            q_corpus = pkl.load(open(q_filepath, 'rb'))
            a_corpus = pkl.load(open(a_filepath, 'rb'))
            qIdToAns = pkl.load(open(best_a_filepath, 'rb'))
            
        self.q_corpus = freq_to_encode_corpus(q_corpus) #Question words to index dictionary
        self.a_corpus = freq_to_encode_corpus(a_corpus) #Answer words to index dictionary
        self.qIdToAns = qIdToAns #Question Id number to encoded answer dictionary
         
    def __len__(self):
        return len(self.qIdToAns)

    def __getitem__(self, idx):
        vqa = self.VQA
        q_ids = vqa.getQuesIds()
        q_id = q_ids[idx]
        im_id = vqa.qqa[q_id]['image_id']
        question = vqa.qqa[q_id]['question']
        question_encoded = get_encoding(self.q_corpus, question)
        image = self.images.get(im_id)
        answer = self.qIdToAns[q_id]
        answer_encoded = get_encoding(self.a_corpus, answer)
        return {"question":question_encoded, "image":image, "answer":answer_encoded}

                
