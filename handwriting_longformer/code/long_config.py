import torch
import os


class Config():
    def __init__(self):
        self.model_path = os.path.join('/data/caojieming/longformer_zh/')
        root_path = os.path.join('/data/caojieming/handwriting_long/handwriting_longformer/data/')
        self.train_path = os.path.join(root_path, 'binary_data/data_all/binary_problem_all_anxiety_train.csv')
        self.dev_path = os.path.join(root_path, 'binary_data/data_all/binary_problem_all_anxiety_dev.csv')
        self.test_path = os.path.join(root_path, 'binary_data/data_all/binary_problem_all_anxiety_test.csv')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_best = os.path.join(
            '/data/caojieming/handwriting_long/handwriting_longformer/best_problem_all_anxiety.pth')
        self.save_last = os.path.join(
            '/data/caojieming/handwriting_long/handwriting_longformer/last_problem_all_anxiety.pth')
        self.num_labels = 2
        self.max_len = 1280
        self.epochs = 10
        self.batch_size = 16
        self.lr = 2e-5
