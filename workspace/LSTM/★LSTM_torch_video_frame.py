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
safe_folder = 'videos (copy)' # 入力パス（安全）
test_folder = 'test_videos' # テストデータの入力パス
frame_size = (64, 64) # フレームサイズ
max_frames = 220 # 最大フレーム数
epochs = 15 # エポック数
batch_size = 8 # バッチサイズ

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

# フレームごとの異常判定を行うためのデータセットクラス
class FrameVideoDataset(Dataset):
    def __init__(self, video_path, frame_size=frame_size, max_frames=max_frames):
        self.frames = preprocess_video(video_path, frame_size, max_frames)
    
    def __len__(self):
        return self.frames.shape[0]
    
    def __getitem__(self, idx):
        return torch.tensor(self.frames[idx], dtype=torch.float32)

# モデルの構築
class VideoLSTM(nn.Module):
    def __init__(self):
        super(VideoLSTM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Conv + Pool後の特徴マップサイズを計算
        self.lstm_input_size = 32 * (frame_size[0] // 2) * (frame_size[1] // 2)  # Conv + Pool後の特徴マップサイズ
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=128, num_layers=2, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(128, 1)  # 出力ノード数を1にして、異常検知のスコアを出力
    
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        # (batch_size, seq_len, c, h, w) -> (batch_size * seq_len, c, h, w)
        x = x.view(batch_size * seq_len, w, h, c)
        x = self.conv1(x)  # (batch_size * seq_len, 32, h', w')
        x = self.pool(x)  # (batch_size * seq_len, 32, h'', w'')
        # (batch_size * seq_len, 32, h'', w'') -> (batch_size, seq_len, -1)
        x = x.view(batch_size, seq_len, -1)
        x, _ = self.lstm(x)  # LSTMに入力
        x = self.fc(x[:, -1])  # 最後のタイムステップの出力を全結合層に渡す
        return torch.sigmoid(x)

# モデルの読み込み
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VideoLSTM().to(device)
model.load_state_dict(torch.load('video_lstm_model.pth'))
model.eval()

# テストデータの処理とフレームごとの予測
def predict_anomalies(video_path, model, threshold=0.5):
    dataset = FrameVideoDataset(video_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    scores = []
    for frames in dataloader:
        frames = frames.unsqueeze(0).to(device)  # (1, seq_len, c, h, w)
        with torch.no_grad():
            output = model(frames)  # モデルの出力
            score = output.item()  # スコアを取得
            scores.append(score)
    
    return scores

# 動画ごとにフレームの異常判定
for filename in os.listdir(test_folder):
    video_path = os.path.join(test_folder, filename)
    scores = predict_anomalies(video_path, model, threshold=0.5)
    
    # スコアと判定の表示
    for i, score in enumerate(scores):
        print(f"Video {filename}, Frame {i+1}, Anomaly Score: {score:.4f}, Prediction: {'Anomaly' if score > 0.3 else 'Normal'}")
