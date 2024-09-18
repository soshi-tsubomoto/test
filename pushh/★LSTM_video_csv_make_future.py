import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from decord import VideoReader
from decord import cpu

# 設定
safe_folder = '/media/pcs/ボリューム/intern/Soshi_tsubomoto/learn_csv_and_video/0905+0906_video_have_rv'  # 学習データの動画データのディレクトリ
train_csv_folder = '/media/pcs/ボリューム/intern/Soshi_tsubomoto/learn_csv_and_video/0905+0906_csv_have_rv_aa'  # 学習データのCSVファイルのディレクトリ
frame_size = (100, 100)  # フレームサイズ
max_frames = 150  # 最大フレーム数
epochs = 100  # エポック数
batch_size = 8  # バッチサイズ
model_save_path = 'video_lstm_model_future.pth'  # モデルの保存先パス
learning_rate = 0.0001  # 学習率
forecast_steps = 30  # 予測する未来のステップ数

def preprocess_video(video_path, frame_size=frame_size, max_frames=max_frames):
    vr = VideoReader(video_path, ctx=cpu(0))
    num_frames = len(vr)
    frames = []

    for i in range(min(max_frames, num_frames)):
        frame = vr[i].asnumpy()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # グレースケールに変換
        frame = cv2.resize(frame, frame_size)
        frames.append(frame)

    if len(frames) < max_frames:
        pad_size = max_frames - len(frames)
        pad_frame = np.zeros((frame_size[0], frame_size[1]), dtype=np.float32)
        for _ in range(pad_size):
            frames.append(pad_frame)

    frames = np.expand_dims(np.array(frames), axis=-1) / 255.0  # チャンネル次元を追加し、0-1の範囲に正規化
    return frames

def preprocess_numerical_data(csv_path, num_frames):
    df = pd.read_csv(csv_path)
    required_columns = ['SP1', 'VSC_GX0', 'VSC_GY0', 'VSC_YAW0']
    df = df[required_columns].apply(pd.to_numeric, errors='coerce')

    # NaN や Inf を 0 で埋める
    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], 0)

    numerical_data = df.values
    
    if len(numerical_data) < num_frames:
        pad_size = num_frames - len(numerical_data)
        numerical_data = np.pad(numerical_data, ((0, pad_size), (0, 0)), mode='constant')
    elif len(numerical_data) > num_frames:
        numerical_data = numerical_data[:num_frames]

    return numerical_data

class FrameVideoDataset(Dataset):
    def __init__(self, video_path, csv_path, frame_size=frame_size, max_frames=max_frames):
        self.frames = preprocess_video(video_path, frame_size, max_frames)
        self.numerical_data = preprocess_numerical_data(csv_path, max_frames)
        
        self.frame_count = len(self.frames)
    
    def __len__(self):
        return self.frame_count - max_frames + 1  # データサンプルの数を修正
    
    def __getitem__(self, idx):
        frames = torch.tensor(self.frames[idx:idx+max_frames], dtype=torch.float32)
        numerical_data = torch.tensor(self.numerical_data[idx:idx+max_frames], dtype=torch.float32)
        return frames, numerical_data

class VideoLSTM(nn.Module):
    def __init__(self):
        super(VideoLSTM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        sample_frame = torch.zeros(1, 1, *frame_size)
        with torch.no_grad():
            sample_output = self.pool(self.conv1(sample_frame))
        self.lstm_input_size = sample_output.numel() + 4  # 画像データと数値データを結合
        
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=128, num_layers=2, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(128, 4)  # 出力を数値データのカラム数に合わせる
    
    def forward(self, x, numerical_data):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, w, h, c).to(device)
        x = self.conv1(x)
        x = self.pool(x)
        x = x.view(batch_size, seq_len, -1)
        
        # 数値データを結合
        numerical_data = numerical_data.to(device)
        x = torch.cat((x, numerical_data), dim=-1)
        
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1])
        return x

def create_dataloader(video_folder, csv_folder, batch_size=batch_size):
    video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
    datasets = [FrameVideoDataset(os.path.join(video_folder, f), os.path.join(csv_folder, f.replace('.mp4', '.csv')))
                for f in video_files]

    def collate_fn(batch):
        frames_list = []
        numerical_data_list = []

        for dataset in batch:
            for idx in range(len(dataset)):
                frames, numerical_data = dataset[idx]
                frames_list.append(frames)
                numerical_data_list.append(numerical_data)

        frames = torch.stack(frames_list)
        numerical_data = torch.stack(numerical_data_list)

        batch_size = frames.size(0)
        frames = frames.view(batch_size, max_frames, *frame_size, 1)
        numerical_data = numerical_data.view(batch_size, max_frames, -1)

        return frames, numerical_data

    return DataLoader(datasets, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_samples = 0
    
    for frames, numerical_data in dataloader:
        frames = frames.to(device)
        numerical_data = numerical_data.to(device)
        
        optimizer.zero_grad()
        outputs = model(frames, numerical_data)
        
        # 最後のフレームの数値データをターゲットとして使用
        targets = numerical_data[:, -1, :].to(device)
        
        loss = criterion(outputs, targets)
        
        # 勾配クリッピングを追加
        if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
            print("NaN or Inf loss detected")
            continue

        loss.backward()
        
        # 勾配クリッピング
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        batch_size = frames.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size
    
    epoch_loss = running_loss / total_samples
    return epoch_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VideoLSTM().to(device)
criterion = nn.MSELoss()  # 回帰用の損失関数
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_loader = create_dataloader(safe_folder, train_csv_folder, batch_size=batch_size)

for epoch in range(epochs):
    epoch_loss = train_model(model, train_loader, criterion, optimizer, device)
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}')

torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

def predict_future(model, video_path, csv_path, forecast_steps=forecast_steps):
    model.eval()
    frames = preprocess_video(video_path, frame_size, max_frames)
    numerical_data = preprocess_numerical_data(csv_path, max_frames)
    
    frames = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(device)
    numerical_data = torch.tensor(numerical_data, dtype=torch.float32).unsqueeze(0).to(device)
    
    predictions = []
    with torch.no_grad():
        for i in range(max_frames - forecast_steps):
            current_frames = frames[:, i:i + forecast_steps]
            current_numerical_data = numerical_data[:, i:i + forecast_steps]

            outputs = model(current_frames, current_numerical_data)
            predictions.append(outputs.cpu().numpy())
    
    return np.concatenate(predictions, axis=0)

# 使用例
# predictions = predict_future(model, '/path/to/test_video.mp4', '/path/to/test_data.csv')
# np.savetxt('future_predictions.csv', predictions, delimiter=',')
