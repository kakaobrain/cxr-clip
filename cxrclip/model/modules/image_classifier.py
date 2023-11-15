from torch import nn


class LinearClassifier(nn.Module):
    def __init__(self, feature_dim, num_class):
        super().__init__()
        self.classification_head = nn.Linear(feature_dim, num_class)

    def forward(self, x):
        return self.classification_head(x)
