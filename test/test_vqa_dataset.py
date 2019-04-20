import unittest
import os
from torch.utils.data import DataLoader
from student_code.vqa_dataset import VqaDataset
import torchvision.transforms as transforms
import pdb, traceback, sys
from nose.plugins.base import Plugin
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
#def debug_on(*exceptions):
#    if not exceptions:
#        exceptions = (AssertionError, )
#    def decorator(f):
#        @functools.wraps(f)
#        def wrapper(*args, **kwargs):
#            try:
#                return f(*args, **kwargs)
#            except exceptions:
#                info = sys.exc_info()
#                traceback.print_exception(*info) 
#                pdb.post_mortem(info[2])
#                raise Exception("Something happened")
#        return wrapper
#    return decorator

class TestVqaDataset(unittest.TestCase):

    def test_load_dataset(self):
        """
        This method gives you a quick way to run your dataset to make sure it loads files correctly.
        It doesn't assert a particular result from indexing the dataset; that will depend on your design.
        Feel free to fill in more asserts here, to validate your design.
        """
        # Arrange
        #current_dir = os.path.dirname(__file__)
        #question_file = os.path.join(current_dir, "test_questions.json")
        #annotation_file = os.path.join(current_dir, "test_annotations.json")
        question_file = "/data/VQA/train/OpenEnded_mscoco_train2014_questions.json"
        annotation_file = "/data/VQA/train/mscoco_train2014_annotations.json"
        image_dir = "/data/VQA/train/images/train2014"
        vqa_dataset = VqaDataset(question_json_file_path=question_file,
                                 annotation_json_file_path=annotation_file,
                                 image_dir=image_dir,
                                 image_filename_pattern="COCO_train2014_{}.jpg")

        # Act
        vqa_len = len(vqa_dataset)
        dataset_item = vqa_dataset[0]

        # Assert
        self.assertEqual(vqa_len, 248349)
        self.assertTrue(type(dataset_item) is dict)

    def test_use_dataset_loader(self):
        """
        Verify that the dataset can be successfully loaded using the DatasetLoader class.
        """
        # Arrange
#        current_dir = os.path.dirname(__file__)
#        question_file = os.path.join(current_dir, "test_questions.json")
#        annotation_file = os.path.join(current_dir, "test_annotations.json")
#        vqa_dataset = VqaDataset(question_json_file_path=question_file,
#                                 annotation_json_file_path=annotation_file,
#                                 image_dir=current_dir,
#                                 image_filename_pattern="COCO_train2014_{}.jpg",
#                                 )
        question_file = "/data/VQA/train/OpenEnded_mscoco_train2014_questions.json"
        annotation_file = "/data/VQA/train/mscoco_train2014_annotations.json"
        image_dir = "/data/VQA/train/images/train2014"
 
        vqa_dataset = VqaDataset(question_json_file_path=question_file,
                                 annotation_json_file_path=annotation_file,
                                 image_dir=image_dir,
                                 image_filename_pattern="COCO_train2014_{}.jpg")
        
        vqa_dataset[0]                         
 
        train_length = len(vqa_dataset)
        indices = list(range(train_length))
        split_set = int(np.floor(0.15 * train_length))
        random_seed = 9999
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split_set:], indices[:split_set]
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        dataset_loader = DataLoader(vqa_dataset, batch_size=2, num_workers=0)

        pdb.set_trace()
        # Act & Assert - the test will fail if iterating through the data loader fails
        try:
            for id, data in enumerate(dataset_loader):
                print(id)
                continue
                # Not doing anything here. Feel free to fill this in, if you like.
        except:
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
