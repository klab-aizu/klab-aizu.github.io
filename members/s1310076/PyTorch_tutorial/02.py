import torch
from torch import nn
from sklearn.datasets import make_moons
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

torch.manual_seed(42)
n_sample = 1000

x, y = make_moons(n_sample, noise=0.07, random_state=42)
print(x[:5], y[:5])

data = pd.DataFrame({"x1":x[:,0],"x2":x[:,1],"label":y})
print(data.head(5))

plt.scatter(x=x[:,0],y=x[:,1],c=y,cmap=plt.cm.RdYlBu);
plt.show()

x = torch.from_numpy(x).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# 80%をトレーニング、20%をテスト
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
print(len(x_train),len(x_test),len(y_train),len(y_test))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# nn.Module のサブクラスとなるモデルクラスを構築
class MoonmodelV0(nn.Module):
    def __init__(self, in_features, out_features, hidden_units):
        super().__init__()
        # X と y の入力形状と出力形状を処理できる 2つの nn.Linear レイヤーを作成
        self.layer_1 = nn.Linear(in_features=in_features, out_features=hidden_units)
        self.layer_2 = nn.Linear(in_features=hidden_units, out_features=hidden_units)
        self.layer_3 = nn.Linear(in_features=hidden_units, out_features=out_features)
        self.relu = nn.ReLU()
        
    # 順方向パスの計算を含む forward メソッドを定義
    def forward(self, x):
        # y と同じ形状の単一特徴である、layer_2 の出力を返す
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))
    # 計算はまずlayer_1を通過し、次にlayer_1の出力がlayer_2を通過・・・
    
    
# モデルのインスタンスを作成し、ターゲットデバイスに送信
model_0 = MoonmodelV0(in_features=2, out_features=1, hidden_units=10).to(device)
print(model_0)
print(model_0.state_dict())

# 損失関数を作成する
loss_fn = nn.BCEWithLogitsLoss()

# オプティマイザーを作成する
optimizer = torch.optim.SGD(params=model_0.parameters(),lr=0.1) 
# lr = learning rate

# 精度（分類指標）を計算する
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true,y_pred).sum().item()
    # torch.eq() は2つのテンソルが等しい位置を計算
    acc = (correct / len(y_pred)) * 100
    return acc

# モデルロジットにシグモイドを使用する
y_logits = model_0(x_test.to(device))[:10]
y_pred_probs = torch.sigmoid(y_logits)

# 予測ラベルを見つける
y_preds = torch.round(y_pred_probs)
y_pred_labels = torch.round(torch.sigmoid(model_0(x_test.to(device))[:10]))
y_preds.squeeze()


# トレーニング及びテストループ
#torch.manual_seed(42)

epochs = 1000

# ターゲットデバイスにデータを送信する
x_train, y_train = x_train.to(device), y_train.to(device)
x_test, y_test = x_test.to(device), y_test.to(device)

# トレーニングと評価のループを構築する
for epoch in range(epochs):
    model_0.train()
    
    y_logits = model_0(x_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
    
    # 最適化ゼロ勾配
    optimizer.zero_grad()
    
    # 後方損失
    loss.backward()
    
    # 最適化ステップ
    optimizer.step()
    
    # トレーニング
    model_0.eval()
    with torch.inference_mode():
        # Forward pass
        test_logits = model_0(x_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        
        # 損失/精度を計算
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)
    
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")


# helper_functions.py からコピペ
def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

# プロット
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_0, x_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_0, x_test, y_test)
plt.show()

A = torch.arange(-10,10,1)
plt.plot(torch.tanh(A))
plt.show()
