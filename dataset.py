from torch.utils.data import Dataset

class ChessDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        # Return the size of the dataset
        return len(self.X)

    def __getitem__(self, idx):
        # Return a single data point and its label
        data_point = self.X[idx]
        label = self.y[idx]
        return data_point, label