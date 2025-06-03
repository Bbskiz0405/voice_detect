import torch
import torch.nn as nn
import torch.nn.functional as F



# class CNN(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 32, 5)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.d1 = nn.Dropout(0.3)

#         self.pool = nn.MaxPool2d(2, 2)

#         self.conv2 = nn.Conv2d(32, 64, 5)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.d2 = nn.Dropout(0.5)

#         self.conv3 = nn.Conv2d(64, 128, 2)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.d3 = nn.Dropout(0.3)

#         self.fc1 = nn.Linear(128 * 2 * 2, 64)
#         self.d5 = nn.Dropout(0.2)

#         self.fc2 = nn.Linear(64, 32)
#         self.d6 = nn.Dropout(0.2)

#         self.fc3 = nn.Linear(32, 16)

#         self.fc4 = nn.Linear(16, 2)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.bn1(x)
#         x = self.d1(x)

#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.bn2(x)
#         x = self.d2(x)

#         x = self.pool(F.relu(self.conv3(x)))
#         x = self.bn3(x)
#         x = self.d3(x)

#         x = x.view(-1, 128 * 2 * 2)

#         x = F.relu(self.fc1(x))
#         x = self.d5(x)

#         x = F.relu(self.fc2(x))
#         x = self.d6(x)

#         x = F.relu(self.fc3(x))

#         x = self.fc4(x)
#         return x


class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)  # 修改輸出通道數為 64
        self.bn1 = nn.BatchNorm2d(64)     # 相應修改批量標準化層
        self.d1 = nn.Dropout(0.3)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64, 128, 5) # 修改輸出通道數為 128
        self.bn2 = nn.BatchNorm2d(128)     # 相應修改批量標準化層
        self.d2 = nn.Dropout(0.5)

        self.conv3 = nn.Conv2d(128, 256, 2) # 修改輸出通道數為 256
        self.bn3 = nn.BatchNorm2d(256)      # 相應修改批量標準化層
        self.d3 = nn.Dropout(0.3)

        # 注意：以下全連接層的輸入特徵數需要根據前面卷積層的輸出調整
        self.fc1 = nn.Linear(256 * 2 * 2, 128) # 修改輸入特徵數並增加輸出特徵數
        self.d5 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(128, 64) # 修改輸入和輸出特徵數
        self.d6 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(64, 32) # 修改輸入特徵數

        self.fc4 = nn.Linear(32, 2) # 維持輸出特徵數不變

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.bn1(x)
        x = self.d1(x)

        x = self.pool(F.relu(self.conv2(x)))
        x = self.bn2(x)
        x = self.d2(x)

        x = self.pool(F.relu(self.conv3(x)))
        x = self.bn3(x)
        x = self.d3(x)

        # 調整視圖以匹配新的全連接層輸入
        x = x.view(-1, 256 * 2 * 2)

        x = F.relu(self.fc1(x))
        x = self.d5(x)

        x = F.relu(self.fc2(x))
        x = self.d6(x)

        x = F.relu(self.fc3(x))

        x = self.fc4(x)
        return x
