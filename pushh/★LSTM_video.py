import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from decord import VideoReader
from decord import cpu

# 設定
safe_folder = '/media/pcs/ボリューム/intern/Soshi_tsubomoto/learn_videos/0905+0906'  # 入力パス（安全）



frame_size = (150, 150)  # フレームサイズ
max_frames = 150  # 最大フレーム数
epochs = 5  # エポック数
batch_size = 2  # バッチサイズ
learning_rate = 0.001  # 学習率

# 動画の前処理
def preprocess_video(video_path, frame_size=frame_size, max_frames=max_frames):
    vr = VideoReader(video_path, ctx=cpu(0))
    num_frames = len(vr)
    frames = []

    for i in range(min(max_frames, num_frames)):
        frame = vr[i].asnumpy()
        frame = cv2.resize(frame, frame_size)
        frames.append(frame)

    while len(frames) < max_frames:
        frames.append(np.zeros((frame_size[0], frame_size[1], 3), dtype=np.float32))

    frames = np.array(frames) / 255.0
    return frames

# データセットクラス（正常データのみ）
class NormalFrameVideoDataset(Dataset):
    def __init__(self, video_folder, frame_size=frame_size, max_frames=max_frames):
        self.video_paths = [os.path.join(video_folder, f) for f in os.listdir(video_folder)]
        self.frame_size = frame_size
        self.max_frames = max_frames

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        frames = preprocess_video(video_path, self.frame_size, self.max_frames)
        return torch.tensor(frames, dtype=torch.float32), 0  # ラベルは0（正常）

# データセットクラス（異常データのみ）
class AnomalyFrameVideoDataset(Dataset):
    def __init__(self, video_folder, frame_size=frame_size, max_frames=max_frames):
        self.video_paths = [os.path.join(video_folder, f) for f in os.listdir(video_folder)]
        self.frame_size = frame_size
        self.max_frames = max_frames

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        frames = preprocess_video(video_path, self.frame_size, self.max_frames)
        return torch.tensor(frames, dtype=torch.float32), 1  # ラベルは1（異常）

# フレーム単位のデータセットクラス
class FrameDataset(Dataset):
    def __init__(self, video_path, frame_size=frame_size, max_frames=max_frames):
        self.frames = preprocess_video(video_path, frame_size, max_frames)
    
    def __len__(self):
        return self.frames.shape[0]  # フレーム数

    def __getitem__(self, idx):
        return torch.tensor(self.frames[idx], dtype=torch.float32), idx  # idxも返す

# モデルの構築
class VideoLSTM(nn.Module):
    def __init__(self):
        super(VideoLSTM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lstm_input_size = 32 * (frame_size[0] // 2) * (frame_size[1] // 2)
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=128, num_layers=2, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, w, h, c)
        x = self.conv1(x)
        x = self.pool(x)
        x = x.view(batch_size, seq_len, -1)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1])
        return torch.sigmoid(x)

# モデルの訓練
def train_model(model, train_loader, criterion, optimizer, epochs=epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

# モデルの評価
def evaluate_model(model, test_loader, is_frame_level=False):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.unsqueeze(0).to(device)  # バッチサイズを1にしてLSTMに渡す
            outputs = model(inputs)
            predictions = (outputs.squeeze() > 0.5).float()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_predictions), np.array(all_labels)

# デバイス設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# モデル、損失関数、最適化手法の定義
model = VideoLSTM().to(device)
criterion = nn.BCELoss()  # バイナリクロスエントロピー損失
optimizer = optim.Adam(model.parameters(), lr=learning_rate)



# データローダーの作成
train_dataset = NormalFrameVideoDataset(safe_folder)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# モデルの訓練
train_model(model, train_loader, criterion, optimizer, epochs)

# モデルの保存
torch.save(model.state_dict(), 'video_lstm_model.pth')

# テストデータの評価（正常データと異常データを含む）
def evaluate_video_frames(video_path, model):
    dataset = FrameDataset(video_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    predictions, labels = evaluate_model(model, dataloader, is_frame_level=True)
    
    for frame_idx, score in enumerate(predictions):
        print(f"Video {os.path.basename(video_path)}, Frame {frame_idx + 1}, Anomaly Score: {score:.4f}, Prediction: {'Anomaly' if score > 0.5 else 'Normal'}")

    return predictions, labels

# 正常データと異常データの評価
def evaluate_all_videos(video_folder, model):
    all_predictions = []
    all_labels = []
    
    for filename in os.listdir(video_folder):
        video_path = os.path.join(video_folder, filename)
        predictions, labels = evaluate_video_frames(video_path, model)
        all_predictions.extend(predictions)
        all_labels.extend(labels)
    
    return np.array(all_predictions), np.array(all_labels)

# 正常データの評価
print("Evaluating normal test data:")
normal_predictions, normal_labels = evaluate_all_videos(normal_folder, model)

# 異常データの評価
print("Evaluating anomaly test data:")
anomaly_predictions, anomaly_labels = evaluate_all_videos(anomaly_folder, model)

# 全フレームでの精度計算
def calculate_accuracy(predictions, labels):
    correct = (predictions == labels).sum()
    total = labels.size
    accuracy = 100 * correct / total
    return accuracy

# 正常データの全フレーム精度
normal_accuracy = calculate_accuracy(normal_predictions, normal_labels)
print(f"Normal Test Data Accuracy: {normal_accuracy:.2f}%")

# 異常データの全フレーム精度
anomaly_accuracy = calculate_accuracy(anomaly_predictions, anomaly_labels)
print(f"Anomaly Test Data Accuracy: {anomaly_accuracy:.2f}%")

# 全データの精度（正常 + 異常）
all_predictions = np.concatenate([normal_predictions, anomaly_predictions])
all_labels = np.concatenate([normal_labels, anomaly_labels])
overall_accuracy = calculate_accuracy(all_predictions, all_labels)
print(f"Overall Accuracy: {overall_accuracy:.2f}%")
