import torch 
from torch import nn
import matplotlib.pyplot as plt 
# nn には PyTorch のニューラルネットワークの構成要素がすべて含まれています
from pathlib import Path

# デバイス設定
device = "cuda" if torch.cuda.is_available() else "cpu"

#ex1
# Create *known* parameters
weight = 0.3
bias = 0.9

# Create data
start = 0
end = 1
step = 0.01
x = torch.arange(start,end,step).unsqueeze(dim=1)
y = weight * x + bias
print(f"First 10 X & y samples:\nX: {x[:10]}\ny: {y[:10]}")
print(f"First 10 X & y samples:\nX: {x[:10]}\ny: {y[:10]}")
# Create train/test split
train_split = int(0.8 * len(x)) # 80% training, 20% testing
x_train, y_train = x[:train_split], y[:train_split]
x_test, y_test = x[train_split:], y[train_split:]

# デバイスに転送
x_train, y_train = x_train.to(device), y_train.to(device)
x_test, y_test = x_test.to(device), y_test.to(device)

len(x_train), len(y_train), len(x_test), len(y_test)

def plot_predictions(train_data=x_train,train_labels=y_train,test_data=x_test,test_labels=y_test,predictions=None):
    # トレーニング データとテスト データをプロットし、予測を比較します
    plt.figure(figsize=(10,7))
    # trining data >> blue
    plt.scatter(train_data.cpu(),train_labels.cpu(),c="b",s=4,label="Training data")
    # test data >> green
    plt.scatter(test_data.cpu(),test_labels.cpu(),c="g",s=4,label="Testing data")

    if predictions is not None:
        # predictions >> red
        plt.scatter(test_data.cpu(),predictions.cpu(),c="r",s=4,label="Predictions")
        
    plt.legend(prop={"size":14})
    
plot_predictions()
plt.show()

# ex2
# 線形回帰モデルクラスを作成
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,dtype=torch.float),requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1,dtype=torch.float),requires_grad=True)

        # 1 <- ランダムな重みから開始します (これは、モデルの学習に応じて調整される)
        # dtype=torch.float<- PyTorch はデフォルトで float32 を使用
        
    # Forward はモデル内の計算を定義する
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # <- "x" は入力データ (例: トレーニング/テスト機能)
        return self.weights * x + self.bias
        # 線形回帰式を返す(y = m*x + b)
        
# nn.Parameterはランダムに初期化されるため、手動でシードを設定する
torch.manual_seed(42)

# モデルのインスタンスを作成（これはnn.Parameterを含むnn.Moduleのサブクラス）
model_1 = LinearRegressionModel().to(device)
model_1,model_1.state_dict()

# 作成した nn.Module サブクラス内の nn.Parameter(s) を確認
list(model_1.parameters())

# 名前付きパラメータのリスト
model_1.state_dict()

# torchを使用した モデルによる予測を行う
with torch.inference_mode():
    y_preds = model_1(x_test)
    
# 予測を確認する
print(f"Number of testing samples: {len(x_test)}") 
print(f"Number of predictions made: {len(y_preds)}")
print(f"Predicted values:\n{y_preds}")

plot_predictions(predictions=y_preds)
plt.show()

y_test - y_preds

# ex3
# 損失関数の作成
# モデルの予測値（例：y_preds）が真のラベル（例：y_test）と比較してどの程度間違っているかを測定。値が低いほど良い。
loss_fn = nn.L1Loss()

# オプティマイザーの作成
# 損失を最大限に低減するために内部パラメータを更新する方法をモデルに指示します
optimizer = torch.optim.SGD(params= model_1.parameters(),lr = 0.01)
# 最適化するターゲットモデルのパラメーター
# 学習率 (オプティマイザーが各ステップでパラメーターをどれだけ変更するか、高いほど多く (安定性が低い)、低いほど少なく (時間がかかる可能性がある))

# トレーニング
#torch.manual_seed(42)
epochs = 300

train_loss_values  =  [] 
test_loss_values  = [] 
epoch_count  =  [] 

for epoch in range(epochs):
    
    model_1.train()
    y_pred = model_1(x_train)
    loss = loss_fn(y_pred,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # テスト
    
    # モデルを評価モードにする
    model_1.eval()
    

    with torch.inference_mode():
        test_pred = model_1(x_test)
        test_loss = loss_fn(test_pred, y_test)
        
        if epoch % 20 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.item()) 
            test_loss_values.append(test_loss.item()) 
            print(f"Epoch: {epoch} | Train loss: {loss:.3f} | Test loss: {test_loss:.3f}")


# 損失曲線のプロット
plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()

# モデルが学習したパラメータの表示
print("The model learned the following values for weights and bias:")
print(model_1.state_dict())
print("\nAnd the original values for weights and bias are:")
print(f"weights: {weight}, bias: {bias}")


# ex4
# モデルを評価モードに設定
with torch.inference_mode():
    predictions = model_1(x_test)

plot_predictions(predictions=predictions.cpu())
plt.show()

# ex5
# モデルの保存
# modelディレクトリの作成
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 保存するmodelのパス
MODEL_NAME = "01_pytorch_workflow_model_1.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_1.state_dict(),f=MODEL_SAVE_PATH)

loaded_model_1 = LinearRegressionModel()
loaded_model_1.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
loaded_model_1.to(device)

loaded_model_1.eval()

with torch.inference_mode():
    loaded_model_preds = loaded_model_1(x_test)

print(predictions == loaded_model_preds)
print(loaded_model_1.state_dict())
