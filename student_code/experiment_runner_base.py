from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from torchvision import transforms
from torchvision import datasets
import sklearn.metrics
from datetime import datetime
from tensorboardX import SummaryWriter
import random
import time
import pdb
import torch.nn as nn

def softmax_mask(H):
    H_max = torch.max(H, dim=1, keepdim=True)[0]
    H_exp = torch.exp(H-H_max)
    H_mask = H_exp * (H_exp != 1.000).type(torch.FloatTensor).cuda(async=True)
    H_softmax = H_mask / torch.sum(H_mask, dim=1, keepdim=True)
    return H_softmax


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Clipper(object):
    def __init__(self, frequency=50):
        self.frequency = frequency
    def __call__(self, module):
        if hasattr(module, 'weight'):
            weight = module.weight.data
            if hasattr(module, 'word'):
                w_clip = 1500
            elif hasattr(module, 'image'):
                return
            else:
                w_clip = 20
            greater = weight.ge(w_clip)
            weight[greater] = w_clip

class ExperimentRunnerBase(object):
    """
    This base class contains the simple train and validation loops for your VQA experiments.
    Anything specific to a particular experiment (Simple or Coattention) should go in the corresponding subclass.
    """

    def __init__(self, train_dataset, val_dataset, model, optimizer, batch_size, num_epochs, num_data_loader_workers=10):
        self._model = model
        self._optimizer = optimizer
        self._num_epochs = num_epochs
        self._log_freq = 100  # Steps
        self._test_freq = 2000  # Steps
        self._clip_freq = 100 # Steps
        self._word_lr = 0.8
        self._other_lr = 0.01
        self._rms_lr = 4e-4
        self._corpus_length = len(train_dataset.question_corpus)

        val_length = len(val_dataset)
        train_length = len(train_dataset)
        indices_val = list(range(val_length))
        indices_train = list(range(train_length))
        split_val = int(np.floor(0.15 * val_length))
        split_train = int(np.floor(0.15 * train_length))
        random_seed = 9999
        np.random.seed(random_seed)
        np.random.shuffle(indices_val)
        val_indices = indices_val[:split_val]
        val_sampler = SubsetRandomSampler(val_indices)
        np.random.shuffle(indices_train)
        train_indices = indices_train[:split_train]
        train_sampler = SubsetRandomSampler(train_indices)
 
        self._train_dataset_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=num_data_loader_workers)
        # If you want to, you can shuffle the validation dataset and only use a subset of it to speed up debugging
        self._val_dataset_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size, num_workers=num_data_loader_workers)
        self._date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self._writer = SummaryWriter('./tensorboard/' + self._date_time)
        
        # Use the GPU if it's available.
        self._cuda = torch.cuda.is_available()

        if self._cuda:
            self._model = self._model.cuda()

    def _calc_loss(self, predicted_answers, true_answers):
        """
        This gets implemented in the subclasses. Don't implement this here.
        """
        raise NotImplementedError()

    def validate(self):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        avg_ap = AverageMeter()
        # TODO. Should return your validation accuracy
        end = time.time()
        aps = []
        iter_prints = int(len(self._val_dataset_loader)/10)
        iter_frac = 0
        for batch_id, data in enumerate(self._val_dataset_loader):
            data_time.update(time.time() - end)
            self._model.eval()

            questions = data['question'].type(torch.FloatTensor)
            images = data['image'].type(torch.FloatTensor)
            answers = data['answer'].type(torch.FloatTensor).cuda(async=True)
            question_lengths = data['question_length'].type(torch.FloatTensor)

            predicted_answers = self._model(images, questions, question_lengths) 

            predicts_bounded = torch.softmax(predicted_answers, dim=1)
            predicted_max_indices = predicts_bounded.max(1)[1] # predicted answer word
            predicted_max_indices = predicted_max_indices.view(predicted_max_indices.size()[0], -1) # resize to Nx1

            predicted_max_indices = predicted_max_indices.detach().cpu().numpy()
            answer_indices = answers.detach().cpu().numpy()

            accuracy = (answer_indices == predicted_max_indices).sum() / answer_indices.shape[0]
            aps.append(accuracy)
            avg_ap.update(accuracy)
            batch_time.update(time.time() - end)
            end = time.time()
            if batch_id % iter_prints == 0:
                iter_frac += 1
                print(str(iter_frac) + "/10 done!")
                print("Interim mAP: ", avg_ap.avg)
                print("Actual mAP?: ", avg_ap.val)
                print("Average Batch Time: ", batch_time.avg)
                print("Average Data Time: ", data_time.avg)    
          
        mAP = np.nanmean(aps)   
        return mAP

    def train(self):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        end = time.time()
        optimizer = self._optimizer
        clipper = Clipper()
        for epoch in range(self._num_epochs):
            self._adjust_lr(optimizer, epoch)
            num_batches = len(self._train_dataset_loader)
            #avg_loss = AverageMeter()
            avg_test_accuracy = AverageMeter()
            avg_train_acc = AverageMeter()
            for batch_id, data in enumerate(self._train_dataset_loader):
                data_time.update(time.time() - end)

                questions = data['question'].type(torch.FloatTensor)
                images = data['image'].type(torch.FloatTensor)
                answers = data['answer'].type(torch.FloatTensor).cuda(async=True)
                question_lengths = data['question_length'].type(torch.FloatTensor)

                current_step = epoch * num_batches + batch_id
                # ============
                # TODO: Run the model and get the ground truth answers that you'll pass to your optimizer
                # This logic should be generic; not specific to either the Simple Baseline or CoAttention.
      
                self._model.train()
                predicted_answer = self._model(images, questions, question_lengths)
 
                # Optimize the model according to the predictions
                loss = self._calc_loss(predicted_answer, answers)
                #avg_loss.update(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()     
                # ============
                predicts_bounded = torch.softmax(predicted_answer, dim=1)
                predicted_max_indices = predicts_bounded.max(1)[1] # predicted answer word
                predicted_max_indices = predicted_max_indices.view(predicted_max_indices.size()[0], -1) # resize to Nx1

                #truth_max_indices = answers.max(1)[1]
                #answer_indices = answers.view(predicted_max_indices.size()[0], -1)
           
                predicted_max_indices = predicted_max_indices.detach().cpu().numpy()
                answer_indices = answers.detach().cpu().numpy()

                train_accuracy = (answer_indices == predicted_max_indices).sum() / answer_indices.shape[0]
                avg_train_acc.update(train_accuracy)

        
                batch_time.update(time.time() - end)
                end = time.time() 
                if current_step % self._clip_freq == 0:
                    self._model.apply(clipper)
                if current_step % self._log_freq == 0:
                    print("Epoch: {}, Batch {}/{} has loss {}".format(epoch, batch_id, num_batches, loss))
                    # TODO: you probably want to plot something here
                    self._writer.add_scalar('train/loss', loss, current_step)
                    self._writer.add_scalar('train/accuracy', avg_train_acc.avg, current_step)
                if current_step % self._test_freq == 0:
                    val_accuracy = self.validate()
                    avg_test_accuracy.update(val_accuracy)
                    print("Epoch: {} has val accuracy {}".format(epoch, val_accuracy))
                    # TODO: you probably want to plot something here
                    self._writer.add_scalar('test/accuracy', avg_test_accuracy.avg, current_step)
                

