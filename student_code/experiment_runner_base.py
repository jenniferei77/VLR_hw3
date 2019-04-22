from torch.utils.data import DataLoader
import torch
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
            if weight > w_clip:
                weight.mul_(w_clip/weight)

def collate_sort(batch):
    batch_size = len(batch)
    sep_batch = list(zip(*batch))
    questions, images, answers, question_lengths = list(sep_batch[0]), sep_batch[1], sep_batch[2], sep_batch[3]
    
    question_lengths = torch.tensor(question_lengths)
    answers = torch.tensor(answers).view(-1, 1)
    lengths_sorted, inds_sorted = question_lengths.sort(0, descending=True)
    
    answers_sorted = answers[inds_sorted]
    ques_sorted = np.zeros((batch_size, len(questions[0])))
    image_sorted = np.zeros((batch_size, 3, 224, 224))
    for ind in inds_sorted:
        ques_sorted[ind, :] = questions[ind]
        image_sorted[ind, :, :, :] = images[ind]
        
    return torch.tensor(ques_sorted), torch.tensor(image_sorted), answers_sorted, lengths_sorted 

class ExperimentRunnerBase(object):
    """
    This base class contains the simple train and validation loops for your VQA experiments.
    Anything specific to a particular experiment (Simple or Coattention) should go in the corresponding subclass.
    """

    def __init__(self, train_dataset, val_dataset, model, optimizer, batch_size, num_epochs, num_data_loader_workers=10):
        self._model = model
        self._optimizer = optimizer
        self._num_epochs = num_epochs
        self._log_freq = 10  # Steps
        self._test_freq = 8000  # Steps
        self._clip_freq = 5 # Steps
        self._word_lr = 0.8
        self._other_lr = 0.01
        self._corpus_length = len(train_dataset.question_corpus)

        val_length = len(val_dataset)
        indices = list(range(val_length))
        split_set = int(np.floor(0.15 * val_length))
        random_seed = 9999
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        val_indices = indices[:split_set]
        val_sampler = SubsetRandomSampler(val_indices)

        self._train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_data_loader_workers)
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
        avg_accuracy = AverageMeter()

        # TODO. Should return your validation accuracy
        end = time.time()
        aps = []
        iter_prints = int(len(self._val_dataset_loader)/10)
        iter_frac = 0
        for batch_id, data in enumerate(self._val_dataset_loader):
            data_time.update(time.time() - end)
            self._model.eval()

            questions = data['question'].type(torch.FloatTensor).cuda(async=True)
            images = data['image'].type(torch.FloatTensor).cuda(async=True)
            answers = data['answer'].type(torch.FloatTensor).cuda(async=True)
            question_lengths = data['question_length'].type(torch.FloatTensor).cuda(async=True)

            predicted_answers = self._model(images, questions, question_lengths) 

            predicts_bounded = F.softmax(predicted_answers, 0)
            predicted_max_indices = predicted_answers.max(1)[1] # predicted answer word
            predicted_max_indices = predicted_max_indices.view(predicted_max_indices.size()[0], -1) # resize to Nx1

            #truth_max_indices = answers.max(1)[1]
            answer_indices = answers.view(predicted_max_indices.size()[0])
           
            predicted_max_indices = predicted_max_indices.detach().cpu().numpy()
            answer_indices = answer_indices.detach().cpu().numpy()

            accuracy = (answer_indices == predicted_max_indices).sum() / answer_indices.shape[0]
            aps.append(accuracy)
            batch_time.update(time.time() - end)
            end = time.time()
            if batch_id % iter_prints == 0:
                iter_frac += 1
                print(str(iter_frac) + "/10 done!")
                interim_mAP = np.nanmean(aps)
                print("Interim mAP: ", interim_mAP)
                print("Actual Batch Time: ", batch_time.val)
                print("Average Batch Time: ", batch_time.avg)
                print("Actual Data Time: ", data_time.val)
                print("Average Data Time: ", data_time.avg)    
          
        mAP = np.nanmean(aps)   
        return mAP

    def train(self):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        avg_accuracy = AverageMeter()

        #word_params = []
        #other_params = []
        #for param in self._model.state_dict().keys():
        #    if 'lin_word_net' in param:
        #        word_params.append(self._model.state_dict()[param])
        #    elif 'classifier' in param:
        #        other_params.append(self._model.state_dict()[param])
        #optimizer = torch.optim.SGD([{'params': self._model.lin_word_net.parameters(), 'lr':0.8}, {'params': self._model.classifier.parameters()}], lr=0.01, momentum=0.9, weight_decay=0.0005)
        #optimizer_word = torch.optim.SGD(self._model.lin_word_net.parameters(), 0.8, 0.9, 0.0005)
        #optimizer_class = torch.optim.SGD(self._model.classifier.parameters(), 0.01, 0.9, 0.0005)
        criterion = nn.CrossEntropyLoss().cuda()
        end = time.time()
        #clipper = Clipper()
        for epoch in range(self._num_epochs):
            num_batches = len(self._train_dataset_loader)
            self._model.train()
            for batch_id, data in enumerate(self._train_dataset_loader):
                data_time.update(time.time() - end)

                questions = data['question'].type(torch.FloatTensor)
                images = data['image'].type(torch.FloatTensor).cuda
                answers = data['answer'].type(torch.FloatTensor).cuda(async=True)
                question_lengths = data['question_length'].type(torch.FloatTensor)

                current_step = epoch * num_batches + batch_id
                # ============
                # TODO: Run the model and get the ground truth answers that you'll pass to your optimizer
                # This logic should be generic; not specific to either the Simple Baseline or CoAttention.
                predicted_answer = self._model(images, questions, question_lengths)
                # ============
                
                # Optimize the model according to the predictions
                #loss = self._calc_loss(predicted_answer, answers)
                #predictions = torch.max(predicted_answer, 1)[1]
                loss = criterion(predicted_answer, answers.view(answers.size()[0]).cuda(async=True))
                self._optimizer.zero_grad()
                #optimizer_word.zero_grad()
                #optimizer_class.zero_grad()
                loss.backward()
                self._optimizer.step()              
                #optimizer_word.step()
                #optimizer_class.step()
 
                for param in self._model.state_dict().keys():
                    if 'lin_word_net' in param:
                        param = self._model.state_dict()[param]
         
                batch_time.update(time.time() - end)
                end = time.time() 
                #if current_step % self._clip_freq == 0:
                #    self._model.apply(clipper)
                if current_step % self._log_freq == 0:
                    print("Epoch: {}, Batch {}/{} has loss {}".format(epoch, batch_id, num_batches, loss))
                    # TODO: you probably want to plot something here
                    self._writer.add_scalar('train/loss', loss.item(), current_step)
                if current_step != 0 and current_step % self._test_freq == 0:
                    val_accuracy = self.validate()
                    print("Epoch: {} has val accuracy {}".format(epoch, val_accuracy))
                    # TODO: you probably want to plot something here
                    self._writer.add_scalar('test/accuracy', val_accuracy, current_step)


