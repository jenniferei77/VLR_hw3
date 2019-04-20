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
import pdb

class Clipper(object):
    def __init__(self, frequency=5):
        self.frequency = frequency
    def __call__(self, module):
        if hasattr(module, 'weight'):
            weight = module.weight.data
            if hasattr(module, 'word'):
                w_clip = 1500
            else:
                w_clip = 20
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
        self._test_freq = 250  # Steps
        self._clip_freq = 5 # Steps
        self._word_lr = 0.8
        self._other_lr = 0.01
        self._corpus_length = 1000

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

    def _optimize(self, predicted_answers, true_answers):
        """
        This gets implemented in the subclasses. Don't implement this here.
        """
        raise NotImplementedError()

    def validate(self):
        # TODO. Should return your validation accuracy
        aps = []

        iter_prints = int(len(self._val_dataset_loader)/10)
        iter_frac = 0
        for batch_id, data in enumerate(self._val_dataset_loader):
            questions = data['question']
            images = data['image']
            answers = data['answer']
            question_lengths = data['question_length']
            self._model.eval()
            predicted_answers = self._model(images, questions, question_lengths) 
            predicts_bounded = torch.sigmoid(predicted_answers)
            predicted_max_indices = predicted_answers.max(1)[1]
            predicted_max_indices = predicted_max_indices.view(predicted_max_indices.size()[0], -1)
            truth_max_indices = answers.max(1)[1]
            truth_max_indices = truth_max_indices.view(predicted_max_indices.size()[0])
            predicted_max_indices = predicted_max_indices.detach().cpu().numpy()
            truth_max_indices = truth_max_indices.detach().cpu().numpy()
            accuracy = (truth_max_indices == predicted_max_indices).sum() / truth_max_indices.shape[0]
            aps.append(accuracy)
            pdb.set_trace() 
            #for index in range(answers.shape[0]):
            #    predicted = predicts_bounded[index].detach().cpu().numpy()
            #    target = answers[index].detach().cpu().numpy()
            #    #predicts_binary = np.zeros(self._corpus_length)
            #    #predicts_binary[predicted] = 1
            #    targets_binary = np.zeros(self._corpus_length)
            #    targets_binary[target] = 1
            #    pdb.set_trace()
            #    ap = sklearn.metrics.average_precision_score(targets_binary, predicted) 
            #    aps.append(ap)
            if batch_id % iter_prints == 0:
                iter_frac += 1
                print(str(iter_frac) + "/10 done!")
                interim_mAP = np.nanmean(aps)
                print("Interim mAP: ", interim_mAP)
        mAP = np.nanmean(accuracy)   
        return mAP

    def train(self):
        optimizer = self._optimizer 
        clipper = Clipper()
        for epoch in range(self._num_epochs):
            num_batches = len(self._train_dataset_loader)
            for batch_id, data in enumerate(self._train_dataset_loader):
                questions = data['question']
                images = data['image']
                answers = data['answer']
                question_lengths = data['question_length']
                self._model.train()  # Set the model to train mode
                current_step = epoch * num_batches + batch_id
                #answers = answers.type(torch.LongTensor).cuda(async=True)
                #questions = questions.type(torch.LongTensor).cuda(async=True) 
                # ============
                # TODO: Run the model and get the ground truth answers that you'll pass to your optimizer
                # This logic should be generic; not specific to either the Simple Baseline or CoAttention.
                #pdb.set_trace()
                predicted_answer = self._model(images, questions, question_lengths)
                # ============
                
                # Optimize the model according to the predictions
                optimizer, loss = self._optimize(predicted_answer, answers)
                
                #if current_step % self._clip_freq == 0:
                #    self._model.apply(clipper)
                if current_step % self._log_freq == 0:
                    print("Epoch: {}, Batch {}/{} has loss {}".format(epoch, batch_id, num_batches, loss))
                    # TODO: you probably want to plot something here
                    self._writer.add_scalar('train/loss', loss.item(), batch_id)
                if current_step % self._test_freq == 0:
                    self._model.eval()
                    val_accuracy = self.validate()
                    print("Epoch: {} has val accuracy {}".format(epoch, val_accuracy))
                    # TODO: you probably want to plot something here
                    self._writer.add_scalar('test/accuracy', val_accuracy, batch_id)
