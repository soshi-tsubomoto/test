import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import uniform_filter1d
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from torchinfo import summary
import os

# CSVファイルを読み込み、データセットを作成する関数
def load_and_concatenate_csvs(directory_path):
    df_list = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path)
            df_list.append(df)
    combined_df = pd.concat(df_list, axis=0, ignore_index=True)
    return combined_df

# データの前処理
def preprocess_data(df, feature_cols, target_col, scaler=None, target_scaler=None, apply_smoothing=True):
    df = df.dropna()
    for col in feature_cols + [target_col]:
        mean = df[col].mean()
        std = df[col].std()
        df = df[(df[col] >= mean - 3*std) & (df[col] <= mean + 3*std)]
    
    if apply_smoothing:
        for col in feature_cols:
            df[col] = uniform_filter1d(df[col], size=5)
    
    if scaler is None:
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(df[feature_cols])
        target_scaler = MinMaxScaler()
        scaled_target = target_scaler.fit_transform(df[[target_col]])
    else:
        scaled_features = scaler.transform(df[feature_cols])
        scaled_target = target_scaler.transform(df[[target_col]])
    
    return scaled_features, scaled_target, scaler, target_scaler

# データセットの作成
def create_dataset(features, targets, window_size, target_size):
    X, y = [], []
    for i in range(len(features) - window_size - target_size + 1):
        X.append(features[i:i + window_size])
        y.append(targets[i + window_size:i + window_size + target_size])
    return np.array(X), np.array(y)

# ハイパーパラメータ
feature_size = 5
hidden_dim = 64
n_layers = 1
window_size = 20   # 過去--行を入力
target_size = 1   # --行分を予測
batch_size = 64
epochs = 100
learning_rate = 0.001

# データの読み込み
train_directory = '/media/pcs/ボリューム/intern/Soshi_tsubomoto/learn_dataset_only _csv/cut_csv_with_R_after_0'
test_directory = '/media/pcs/ボリューム/intern/Soshi_tsubomoto/test_dataset_only_csv/test_data/'

feature_cols = ['SP1', 'VSC_GX0', 'VSC_GY0', 'VSC_YAW0', 'my_car_R']
target_col = 'SP1'

df_train = load_and_concatenate_csvs(train_directory)
df_test = load_and_concatenate_csvs(test_directory)

train_features, train_targets, feature_scaler, target_scaler = preprocess_data(df_train, feature_cols, target_col)
test_features, test_targets, _, _ = preprocess_data(df_test, feature_cols, target_col, scaler=feature_scaler, target_scaler=target_scaler, apply_smoothing=False)

X_train, y_train = create_dataset(train_features, train_targets, window_size, target_size)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# LSTMモデルの定義
class MyLSTM(nn.Module):
    def __init__(self, feature_size, hidden_dim, n_layers, target_size):
        super(MyLSTM, self).__init__()
        self.feature_size = feature_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.target_size = target_size

        self.lstm = nn.LSTM(feature_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, target_size)  # 出力サイズをtarget_sizeに合わせる

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.n_layers, x.size(0), self.hidden_dim)).to(x.device)
        c_0 = Variable(torch.zeros(self.n_layers, x.size(0), self.hidden_dim)).to(x.device)
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        # 最後のタイムステップの出力を使う
        last_output = output[:, -1, :]
        y = self.fc(last_output)
        return y

net = MyLSTM(feature_size, hidden_dim, n_layers, target_size)
summary(net)

func_loss = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

loss_history = []
for epoch in range(epochs):
    net.train()
    tmp_loss = 0.0
    for x, t in train_loader:
        x = x.to(device)
        t = t.to(device)
        optimizer.zero_grad()
        y = net(x)
        # 出力とターゲットのサイズを確認
        print(f'Output size: {y.size()}, Target size: {t.size()}')
        loss = func_loss(y, t.view(-1, target_size))  # ここでターゲットのサイズを調整
        loss.backward()
        optimizer.step()
        tmp_loss += loss.item()
    tmp_loss /= len(train_loader)
    loss_history.append(tmp_loss)
    print(f'Epoch: {epoch+1}/{epochs}, Loss: {tmp_loss:.4f}')

plt.plot(range(epochs), loss_history, label='Train Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()



# テストデータの予測
initial_test_features = test_features[:100]
initial_test_targets = test_targets[:100]

X_test, _ = create_dataset(initial_test_features, initial_test_targets, window_size, target_size)

test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(initial_test_targets[window_size:window_size + len(X_test)], dtype=torch.float32))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

net.eval()
predicted_test_plot = []
true_test_plot = []
with torch.no_grad():
    for x, y_true in test_loader:
        x = x.to(device)
        y_pred = net(x)
        predicted_test_plot.extend(y_pred.cpu().numpy().flatten())
        true_test_plot.extend(y_true.numpy().flatten())

# データの長さを確認
print(f'Length of true_test_plot: {len(true_test_plot)}, Length of predicted_test_plot: {len(predicted_test_plot)}')

# プロット
plt.figure(figsize=(14, 7))
plt.plot(range(len(true_test_plot)), true_test_plot, label='True')
plt.plot(range(len(predicted_test_plot)), predicted_test_plot, label='Predicted')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.show()


# モデルの出力サイズを確認
print(f'Output size of the model: {target_size}')




