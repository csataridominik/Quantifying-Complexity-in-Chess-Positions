import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessModel(nn.Module):
    def __init__(self):
        super(ChessModel, self).__init__()


        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=14, out_channels=64, kernel_size=3, stride=1, padding=1)  # (64x8x8)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) # (128x8x8)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1) # (256x8x8)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)  # Flatten and fully connect
        self.fc2 = nn.Linear(1024, 4096)         # Output layer for move classification

    def forward(self, x):
        # Pass through conv layers with ReLU activations and batch norm (optional)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten for the fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers with ReLU activations
        x = F.relu(self.fc1(x))

        # Output layer (logits)
        x = self.fc2(x)
        
        return x
    
# In this part I will write the attacking expert having input:
'''
-	12 planes
-	Attacked squares
-	Move from
-	Piece distribution drawn towards attacking??
-	Free files
-	Attacked squares with great placement
-	Diagonals tha can be taken
-	Enemy king position

This is in total:  Input: 19 planes, 4096 output vector

'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class SE_Block_4_input(nn.Module):
    def __init__(self, c, r=16):
        super(SE_Block_4_input, self).__init__()
        self.c = c
        self.excitation = nn.Sequential(
            nn.Linear(c*8*8, c*8*8 // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c*8*8 // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, h, w = x.size()
        
        y = x.view(bs, c * h * w)  # Shape is now (batch_size, c * h * w)
      
        # Pass through excitation layers
        y = self.excitation(y).view(bs, c, 1, 1)
        
        # Return recalibrated tensor with original tensor expanded for broadcasting
        return x * y.expand_as(x), y
    
class SE_Block(nn.Module):
    def __init__(self, c, r=16):
        super(SE_Block, self).__init__()
        self.c = c  # Store the number of channels for later verification
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.size()
        
        # Ensure channel dimension matches expected value
        if c != self.c:
            raise ValueError(f"Expected {self.c} channels, but got {c}. Check input tensor dimensions.")
        
        # Squeeze operation to get the (batch_size, c) shape
        y = self.squeeze(x).view(bs, c)
        
        # Pass through excitation layers
        y = self.excitation(y).view(bs, c, 1, 1)
        
        # Return recalibrated tensor with original tensor expanded for broadcasting
        return x * y.expand_as(x), y


class ChessExpert1(nn.Module):
    def __init__(self):
        super(ChessExpert1, self).__init__()
        
        # SE Block applied to the input channels
        self.se_block_input = SE_Block_4_input(14)
        
        # Encoder: Convolution layers to extract features
        self.conv1 = nn.Conv2d(14, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)  # BatchNorm for the first conv layer
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # SE Block after convolution layers for channel re-calibration
        self.se_block1 = SE_Block(64)
        self.se_block2 = SE_Block(128)
        self.se_block3 = SE_Block(256)
        
        # Decoder: Fully connected layers for the output
        self.fc1 = nn.Linear(256 * 8 * 8, 512)  # First fully connected layer
        self.fc2 = nn.Linear(512, 4096)         # Output layer, size 4096 as specified
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.3)  # Dropout with 30% rate
        
    def forward(self, x):
        # SE Block on the input
        x, se_input = self.se_block_input(x)
        
        # Encoder with BatchNorm
        x = F.relu(self.bn1(self.conv1(x)))
        x, se1 = self.se_block1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x, se2 = self.se_block2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x, se3 = self.se_block3(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 256*8*8)
        
        # Decoder with fully connected layers and dropout
        x = F.relu(self.fc1(x))  # First fully connected layer
        x = self.dropout(x)      # Apply dropout
        out = self.fc2(x)        # Output layer (4096 dimensions)

        return out, se_input


# This is for choosing pieces: 
class ChessExpert2(nn.Module):
    def __init__(self):
        super(ChessExpert2, self).__init__()
        
        # SE Block applied to the input channels
        self.se_block_input = SE_Block_4_input(14)
        
        # Encoder: Convolution layers to extract features
        self.conv1 = nn.Conv2d(14, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)  # BatchNorm for the first conv layer
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # SE Block after convolution layers for channel re-calibration
        self.se_block1 = SE_Block(64)
        self.se_block2 = SE_Block(128)
        self.se_block3 = SE_Block(256)
        
        # Decoder: Fully connected layers for the output
        self.fc1 = nn.Linear(256 * 8 * 8, 512)  # First fully connected layer
        self.fc2 = nn.Linear(512, 128)  # First fully connected layer
        self.fc3 = nn.Linear(128, 6)         # Output layer, size 6 as specified
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.3)  # Dropout with 30% rate
        
    def forward(self, x):
        # SE Block on the input
        x, se_input = self.se_block_input(x)
        
        # Encoder with BatchNorm
        x = F.relu(self.bn1(self.conv1(x)))
        x, se1 = self.se_block1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x, se2 = self.se_block2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x, se3 = self.se_block3(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 256*8*8)
        
        # Decoder with fully connected layers and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  
        x = F.relu(self.fc2(x))
        out = self.fc3(x)       

        return out, se_input



# This is for choosing pieces: 
class ChessExpert3(nn.Module):
    def __init__(self):
        super(ChessExpert3, self).__init__()
        
        # SE Block applied to the input channels
        self.se_block_input = SE_Block_4_input(14)
        
        # Encoder: Convolution layers to extract features
        self.conv1 = nn.Conv2d(14, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)  # BatchNorm for the first conv layer
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # SE Block after convolution layers for channel re-calibration
        self.se_block1 = SE_Block(64)
        self.se_block2 = SE_Block(128)
        self.se_block3 = SE_Block(256)
        
        # Decoder: Fully connected layers for the output
        self.fc1 = nn.Linear(256 * 8 * 8, 512)  # First fully connected layer
        self.fc2 = nn.Linear(512, 256)  # First fully connected layer
        self.fc3 = nn.Linear(256, 64) 
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.3)  # Dropout with 30% rate
        
    def forward(self, x):
        # SE Block on the input
        x, se_input = self.se_block_input(x)
        
        # Encoder with BatchNorm
        x = F.relu(self.bn1(self.conv1(x)))
        x, se1 = self.se_block1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x, se2 = self.se_block2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x, se3 = self.se_block3(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 256*8*8)
        
        # Decoder with fully connected layers and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  
        x = F.relu(self.fc2(x))
        out = self.fc3(x)       

        return out, se_input
