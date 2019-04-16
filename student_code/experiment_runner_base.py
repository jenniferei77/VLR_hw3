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

class ExperimentRunnerBase(object):
    """
    This base class contains the simple train and validation loops for your VQA experiments.
    Anything specific to a particular experiment (Simple or Coattention) should go in the corresponding subclass.
    """

    def __init__(self, train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers=10):
        self._model = model
        self._num_epochs = num_epochs
        self._log_freq = 10  # Steps
        self._test_freq = 250  # Steps

        train_length = len(train_dataset)
        indices = list(range(train_length))
        split_set = int(np.floor(0.15 * train_length))
        random_seed = 9999
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split_set:], indices[:split_set]
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        
        self._train_dataset_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, shuffle=True, num_workers=num_data_loader_workers)

        # If you want to, you can shuffle the validation dataset and only use a subset of it to speed up debugging
         
        self._val_dataset_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size, shuffle=True, num_workers=num_data_loader_workers)
        
        self._date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self._writer = SummaryWriter('./tensorboard/' + date_time)
 
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
        for batch_id, (questions, images, answers) in enumerate(self._val_dataset_loader):
            self._model.eval()
            aps = []
            predicted_answers = self._model(images, questions) 
            predicted_max_indices = predicted_answers.max(1)[1]
            predicted_max_indices.view(answers.shape)
            for index in range(answers.shape[-1]):
                predicted = predicted_max_indices[index]
                target = answers[index]
                ap = sklearn.metrics.average_precision_score(target, predicted) 
                aps.append(ap)
        mAP = np.nanmean(aps)   
        return mAP

    def train(self):
        optimizer = self._optimizer 
        for epoch in range(self._num_epochs):
            num_batches = len(self._train_dataset_loader)

            for batch_id, (questions, images, answers) in enumerate(self._train_dataset_loader):
                self._model.train()  # Set the model to train mode
                current_step = epoch * num_batches + batch_id

                # ============
                # TODO: Run the model and get the ground truth answers that you'll pass to your optimizer
                # This logic should be generic; not specific to either the Simple Baseline or CoAttention.
                
                predicted_answer = self._model(images, questions)
                # ============

                # Optimize the model according to the predictions
                optimizer, loss = self._optimize(predicted_answer, answers)
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
