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

class VqaDataset(Dataset):
    """
    Load the VQA dataset using the VQA python API. We provide the necessary subset in the External folder, but you may
    want to reference the full repo (https://github.com/GT-Vision-Lab/VQA) for usage examples.
    """

    def __init__(self, image_dir=None, question_json_file_path=None, annotation_json_file_path=None, image_filename_pattern=None, transform=None, omit_words=None, q_filepath=None, a_filepath=None, a_best_filepath=None, max_ques_length=50):
        """
        Args:
            image_dir (string): Path to the directory with COCO images
            question_json_file_path (string): Path to the json file containing the question data
            annotation_json_file_path (string): Path to the json file containing the annotations mapping images, questions, and
                answers together
            image_filename_pattern (string): The pattern the filenames of images in this dataset use (eg "COCO_train2014_{}.jpg")
        """
        if "COCO_train2014" in image_filename_pattern:
            images_prefix = "COCO_train2014_"
        elif "COCO_val2014" in image_filename_pattern:
            images_prefix = "COCO_val2014_"
        else:
            print('imvalid image_file_pattern')
            exit(1)
        
        self.max_ques_length = max_ques_length

        if omit_words is None:
            self.omit_words = ['the', 'a', 'so', 'her', 'him', 'very', 'an', 'this', 'does', 'will', 'i']
        else:
            self.omit_words = omit_words
       
        if transform is not None:
            self.transforms = transform
        else:
            self.transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        self.num_images_cap = 500 
        self.images = {}
        im_type = image_filename_pattern.split('.')[-1]
        self.image_dir = image_dir
        vqa = VQA(annotation_json_file_path, question_json_file_path)
        self.VQA = vqa

        if not image_dir == None:
            print('Loading COCO images and question data from ', image_dir)
            images = {}
            
            try:
                image_files = image_dir + '/' + images_prefix + '*.' + im_type
                #print("Image Files: ", image_files)
                num_images = 0
                for filename in glob.glob(image_files):
                    #print("Image Filename: ", filename)
                    num_images += 1
                    if num_images % 100 == 0:
                        print(num_images)
                    if num_images > self.num_images_cap:
                        continue
                    im_index = filename.split('.')[-2].split('_')[-1]
                    im_index = im_index.lstrip('0')
                    loaded_img = default_loader(filename)
                    if loaded_img is not None:
                        image = self.transforms(loaded_img)
                        images[int(im_index)] = image
                        #print("Image loaded correctly with index", im_index)
                    else:
                        print("Image not loaded correctly")
                self.images = images
            except OSError:
                print('image_dir is not a real directory')

        if q_filepath == None or a_filepath == None or a_best_filepath == None:
            q_corpus = {}
            a_corpus = {}
            q_ids = vqa.getQuesIds()
            qIdToAns = {}
            qIdToEncoding = {}
            for q_id in q_ids:
                #Image debugging, don't use for final training
                image_index = vqa.qqa[q_id]['image_id']
                if image_index not in images.keys():
                    continue     
                #Build question corpus by frequency 
                question = vqa.qqa[q_id]['question']
                q_words = separate(question)
                q_corpus = add_to_corpus_freq(q_corpus, q_words, self.omit_words)
                answers = vqa.qa[q_id]['answers']
                distinct_answers = {}
                for ans_cluster in answers:
                    answer = ans_cluster['answer']
                    if ans_cluster['answer_confidence'] != 'yes':
                        continue
                    #Build answer corpus by frequency
                    a_words = separate(answer)
                    a_corpus = add_to_corpus_freq(a_corpus, a_words, self.omit_words)
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
                qIdToAns[q_id] = best_ans
            #Sort question corpus by frequency and take top 1000
            q_corpus_sorted = sorted(q_corpus.items(), key=lambda kv: kv[1])
            if len(q_corpus_sorted) >= 1000:
                q_corpus = q_corpus_sorted[-1000:]
            else:
                q_corpus = q_corpus_sorted

            #Sort answer corpus by frequency and take top 1000
            a_corpus_sorted = sorted(a_corpus.items(), key=lambda kv: kv[1])
            if len(a_corpus_sorted) >= 1000:
                a_corpus = a_corpus_sorted[-1000:]
            else:
                a_corpus = a_corpus_sorted
            
            date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            q_filepath = './created_data/question_corpus_' + date_time + '.pkl'
            a_filepath = './created_data/answer_corpus_' + date_time + '.pkl'
            best_a_filepath = './created_data/best_answers' + date_time + '.pkl'
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
        q_ids = list(self.qIdToAns.keys())
        q_id = q_ids[idx]
        im_id = vqa.qqa[q_id]['image_id']
        question = vqa.qqa[q_id]['question']
        #print("Actual Question: ", question)
        question_encoded = get_encoding(self.q_corpus, question)
        question_padded = pad_question(self.max_ques_length, question_encoded)
        image = self.images.get(im_id)
        answer = self.qIdToAns[q_id]
        #print ("Actual Answer: ", answer)
        answer_encoded = get_encoding(self.a_corpus, answer)
        print("Image: ", image.shape)
        print("Question: ", question_encoded)
        print("Answer: ", answer_encoded)
        question_length = len(question_encoded)
        #pdb.set_trace()
        return question_padded, image, answer_encoded, question_length

