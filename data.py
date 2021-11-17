import torch
class DataLoaderInput:
    """
    Data structure required for torch data iterable
    """

    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        x = self.X[idx].float()
        y = self.y[idx]
        return x, y, idx

    def __len__(self):
        return len(self.X)