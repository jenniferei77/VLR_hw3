from student_code.simple_baseline_net import SimpleBaselineNet
from student_code.experiment_runner_base import ExperimentRunnerBase
from student_code.vqa_dataset import VqaDataset

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import numpy as np
import torch
import torch.nn as nn
import pdb

class SimpleBaselineExperimentRunner(ExperimentRunnerBase):
    """
    Sets up the Simple Baseline model for training. This class is specifically responsible for creating the model and optimizing it.
    """
    def __init__(self, train_image_dir, train_question_path, train_annotation_path,
                 test_image_dir, test_question_path, test_annotation_path, loaded_question_corpus, loaded_answer_corpus, train_best_answers_filepath, val_best_answers_filepath, batch_size, num_epochs,
                 num_data_loader_workers=10, corpus_length=1000):

        train_dataset = VqaDataset(image_dir=train_image_dir,
                                   question_json_file_path=train_question_path,
                                   annotation_json_file_path=train_annotation_path,
                                   image_filename_pattern="COCO_train2014_{}.jpg",
                                   loaded_question_corpus=loaded_question_corpus,
                                   loaded_answer_corpus=loaded_answer_corpus,
                                   best_answers_filepath=train_best_answers_filepath,
                                   corpus_length=corpus_length)

        val_dataset = VqaDataset(image_dir=test_image_dir,
                                 question_json_file_path=test_question_path,
                                 annotation_json_file_path=test_annotation_path,
                                 image_filename_pattern="COCO_val2014_{}.jpg",
                                 loaded_question_corpus=loaded_question_corpus,
                                 loaded_answer_corpus=loaded_answer_corpus,
                                 best_answers_filepath=val_best_answers_filepath,
                                 corpus_length=corpus_length)
 
        model = SimpleBaselineNet(corpus_length=len(train_dataset.question_corpus))
        model.cuda()
        
        #word_params = []
        #other_params = []
        word_params = {}
        other_params = {}
        #optimize_params = model.state_dict()
        for name, param in model.state_dict().items():
            if isinstance(param, nn.Parameter):
                param = param.data
            if 'lin_word_net' in name:
                word_params[name] = param
                #word_params.append(torch.nn.Parameter(model.state_dict()[param], requires_grad=True))
            elif 'classifier' in name:
                other_params[name] = param
                #other_params.append(torch.nn.Parameter(model.state_dict()[param], requires_grad=True))
        optimizer = torch.optim.SGD([{'params': model.lin_word_net.parameters(), 'lr':0.8}, {'params': model.classifier.parameters()}], momentum=0.9, lr=0.01, weight_decay=0.0005)
        #optimizer = None
        super().__init__(train_dataset, val_dataset, model, optimizer, batch_size, num_epochs, num_data_loader_workers)

    def _calc_loss(self, predicted_answers, true_answers):
        # TODO
        # predicted_answers: Nx1000 of predictions for each word
        # true_answers: 1000 of gt_answer indices
        loss = F.cross_entropy(predicted_answers, true_answers.type(torch.LongTensor).view(true_answers.size()[0]).cuda(async=True))
        return loss

