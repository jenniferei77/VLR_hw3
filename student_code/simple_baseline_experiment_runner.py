from student_code.simple_baseline_net import SimpleBaselineNet
from student_code.experiment_runner_base import ExperimentRunnerBase
from student_code.vqa_dataset import VqaDataset

import torch.nn.functional as F
import cv2
import numpy as np
import torch
import torch.nn as nn

class SimpleBaselineExperimentRunner(ExperimentRunnerBase):
    """
    Sets up the Simple Baseline model for training. This class is specifically responsible for creating the model and optimizing it.
    """
    def __init__(self, train_image_dir, train_question_path, train_annotation_path,
                 test_image_dir, test_question_path,test_annotation_path, batch_size, num_epochs,
                 num_data_loader_workers):

        train_dataset = VqaDataset(image_dir=train_image_dir,
                                   question_json_file_path=train_question_path,
                                   annotation_json_file_path=train_annotation_path,
                                   image_filename_pattern="COCO_train2014_{}.jpg")
        val_dataset = VqaDataset(image_dir=test_image_dir,
                                 question_json_file_path=test_question_path,
                                 annotation_json_file_path=test_annotation_path,
                                 image_filename_pattern="COCO_val2014_{}.jpg")

        model = SimpleBaselineNet()
        
        self._optimizer = torch.optim.SGD([{'params':self._model.word_feats.parameters(), 'lr':0.01}, {'params':[self._model.image_cnn.parameters(), self._model.softmax.parameters()]}], lr=0.0001, momentum=0.9, weight_decay=0.0005)
        
        super().__init__(train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers)

    def _optimize(self, predicted_answers, true_answers):
        #Assume predicted_answers is list of answers
        # TODO
        predicted_max_indices = predicted_answers.max(1)[1]   
        loss = F.cross_entropy(predicted_max_indices, true_answers)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return self._optimizer, loss
