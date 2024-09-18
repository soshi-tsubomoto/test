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
frame_size = (200, 200)  # フレームサイズ
max_frames = 150  # 最大フレーム数
epochs = 5  # エポック数
batch_size = 8  # バッチサイズ
model_save_path = 'video_andsuuti_lstm_model_al_200.pth'  # モデルの保存先パス
learning_rate = 0.001  # 学習率

def preprocess_video(video_path, frame_size=frame_size, max_frames=max_frames):
    vr = VideoReader(video_path, ctx=cpu(0))
    num_frames = len(vr)
    frames = []

    for i in range(min(max_frames, num_frames)):
        frame = vr[i].asnumpy()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # グレースケールに変換
        frame = cv2.resize(frame, frame_size)
        frames.append(frame)

    # 最大フレーム数に合わせてパディングを追加
    if len(frames) < max_frames:
        pad_size = max_frames - len(frames)
        pad_frame = np.zeros((frame_size[0], frame_size[1]), dtype=np.float32)
        for _ in range(pad_size):
            frames.append(pad_frame)

    frames = np.expand_dims(np.array(frames), axis=-1) / 255.0  # チャンネル次元を追加し、0-1の範囲に正規化
    return frames


def preprocess_numerical_data(csv_path, num_frames):
    # CSVファイルを読み込み
    df = pd.read_csv(csv_path)

    # 必要なカラムを指定
    required_columns = ['SP1', 'VSC_GX0', 'VSC_GY0', 'VSC_YAW0']

    # 指定されたカラムだけを取得し、数値型に変換
    df = df[required_columns].apply(pd.to_numeric, errors='coerce')

    

    numerical_data = df.values
    
    # num_framesに合わせてデータをトリミングまたはパディング
    if len(numerical_data) < num_frames:
        pad_size = num_frames - len(numerical_data)
        numerical_data = np.pad(numerical_data, ((0, pad_size), (0, 0)), mode='constant')
    elif len(numerical_data) > num_frames:
        numerical_data = numerical_data[:num_frames]

    return torch.tensor(numerical_data, dtype=torch.float32).unsqueeze(0)  # バッチサイズを追加

class FrameVideoDataset(Dataset):
    def __init__(self, video_path, csv_path, frame_size=frame_size, max_frames=max_frames):
        self.frames = preprocess_video(video_path, frame_size, max_frames)
        self.numerical_data = preprocess_numerical_data(csv_path, max_frames)
        
        # 動画のフレーム数に合わせて数値データを調整
        self.frame_count = len(self.frames)
        self.numerical_data = self.numerical_data.view(self.frame_count, -1)
    
    def __len__(self):
        return self.frame_count
    
    def __getitem__(self, idx):
        frame = torch.tensor(self.frames[idx], dtype=torch.float32)
        numerical_data = self.numerical_data[idx]
        return frame, numerical_data

class VideoLSTM(nn.Module):
    def __init__(self):
        super(VideoLSTM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)  # in_channelsを1に修正
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ダミー入力でサイズ確認
        sample_frame = torch.zeros(1, 1, 200, 200)  # 100x100 のフレームサイズ
        with torch.no_grad():
            sample_output = self.pool(self.conv1(sample_frame))
        self.lstm_input_size = sample_output.numel()  # (out_channels * h/2 * w/2)
        
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=128, num_layers=2, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        # x の形状を確認
        print(f"x.size(): {x.size()}")  # デバッグ用

        batch_size, seq_len, h, w, c = x.size()  # x は [batch_size, seq_len, c, h, w] の5次元テンソル
        
        # (batch_size * seq_len, c, h, w) にリシェイプ
        x = x.view(batch_size * seq_len, c, h, w) 

        # 畳み込み層に適用
        x = self.conv1(x)
        x = self.pool(x)
        
        # (batch_size, seq_len, feature_size) にリシェイプ
        x = x.view(batch_size, seq_len, -1)

        # LSTM に通す
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1])  # LSTM の最後の出力を FC 層に通す

        return torch.sigmoid(x)  # Sigmoid で出力




# データローダーのセットアップ
def create_dataloader(video_folder, csv_folder, batch_size=batch_size):
    video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
    datasets = [FrameVideoDataset(os.path.join(video_folder, f), os.path.join(csv_folder, f.replace('.mp4', '.csv')))
                for f in video_files]

    def collate_fn(batch):
        frames_list = []
        numerical_data_list = []

        for dataset in batch:
            for idx in range(len(dataset)):
                frame, numerical_data = dataset[idx]
                frames_list.append(frame)
                numerical_data_list.append(numerical_data)

        frames = torch.stack(frames_list)
        numerical_data = torch.stack(numerical_data_list)

        # バッチ内のサンプル数を確認
        print(f"frames.size() after stacking: {frames.size()}")
        print(f"numerical_data.size() after stacking: {numerical_data.size()}")

        # バッチサイズに基づいてデータをリシェイプ
        batch_size = frames.size(0) // max_frames
        frames = frames.view(batch_size, max_frames, 200, 200, 1)
        numerical_data = numerical_data.view(batch_size, max_frames, -1)

        return frames, numerical_data

    return DataLoader(datasets, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)



train_loader = create_dataloader(safe_folder, train_csv_folder)

# モデル、ロス関数、オプティマイザのインスタンス作成
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 追加: デバイスの設定

model = VideoLSTM().to(device)  # モデルをデバイスに移動
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 学習ループ
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for frames, numerical_data in train_loader:
        # デバッグ用: バッチサイズとフレーム数を確認
        print(f"frames.size() in training loop: {frames.size()}")
        print(f"numerical_data.size() in training loop: {numerical_data.size()}")

        frames = frames.to(device)
        numerical_data = numerical_data.to(device)
        
        optimizer.zero_grad()
        outputs = model(frames)  # numerical_data を渡さない
        targets = torch.zeros(outputs.size(0), 1).to(device)  # ダミーのターゲット（例: 全て0）
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * frames.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}')

# モデルの保存
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

