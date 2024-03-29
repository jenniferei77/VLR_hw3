from student_code.coattention_net import CoattentionNet
from student_code.experiment_runner_base import ExperimentRunnerBase
from student_code.vqa_dataset import VqaDataset
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pdb

class CoattentionNetExperimentRunner(ExperimentRunnerBase):
    """
    Sets up the Co-Attention model for training. This class is specifically responsible for creating the model and optimizing it.
    """
    def __init__(self, train_image_dir, train_question_path, train_annotation_path, 
                 test_image_dir, test_question_path, test_annotation_path, loaded_question_corpus, loaded_answer_corpus, train_best_answers_filepath, val_best_answers_filepath, train_image_features_filepath, train_loaded_imgIdToidx_filepath, val_image_features_filepath, val_loaded_imgIdToidx_filepath, batch_size=300, num_epochs=100, num_data_loader_workers=10, corpus_length=1000):

        train_dataset = VqaDataset(image_dir=train_image_dir,
                                   question_json_file_path=train_question_path,
                                   annotation_json_file_path=train_annotation_path,
                                   image_filename_pattern="COCO_train2014_{}.jpg",
                                   loaded_question_corpus=loaded_question_corpus,
                                   loaded_answer_corpus=loaded_answer_corpus,
                                   best_answers_filepath=train_best_answers_filepath,
                                   image_features_filepath=train_image_features_filepath,
                                   loaded_imgIdToidx_filepath=train_loaded_imgIdToidx_filepath,
                                   corpus_length=corpus_length,
                                   model_type='coattention')

        val_dataset = VqaDataset(image_dir=test_image_dir,
                                 question_json_file_path=test_question_path,
                                 annotation_json_file_path=test_annotation_path,
                                 image_filename_pattern="COCO_val2014_{}.jpg",
                                 loaded_question_corpus=loaded_question_corpus,
                                 loaded_answer_corpus=loaded_answer_corpus,
                                 best_answers_filepath=val_best_answers_filepath,
                                 image_features_filepath=val_image_features_filepath,
                                 loaded_imgIdToidx_filepath=val_loaded_imgIdToidx_filepath,
                                 corpus_length=corpus_length,
                                 model_type='coattention')
 
        self._model = CoattentionNet(corpus_length=len(train_dataset.question_corpus), batch_size=batch_size)
        self._model.cuda()
        
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-8)

        super().__init__(train_dataset, val_dataset, self._model, self._optimizer, batch_size, num_epochs,
                         num_data_loader_workers)

    def _calc_loss(self, predicted_answers, true_answer_ids):
        # TODO
        #predicted_answers = torch.max(predicted_answers, 1)[1].type(torch.FloatTensor)
        true_answers = true_answer_ids.view(true_answer_ids.size()[0]).type(torch.FloatTensor).cuda(async=True)
        loss = F.cross_entropy(predicted_answers, true_answers.type(torch.LongTensor).cuda(async=True))
        return loss

    def _adjust_lr(self, optimizer, epoch):
        lr = self._rms_lr * (0.1**(epoch // 20))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
