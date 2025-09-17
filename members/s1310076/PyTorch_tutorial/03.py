
# model_0 - 2つのnn.Linear()層を持つベースラインモデル。
# model_1 - ベースラインモデルと同じ構成ですが、nn.Linear()層の間にnn.ReLU()層があります。
# model_2 - CNN ExplainerウェブサイトにあるTinyVGGアーキテクチャを模倣した、最初のCNNモデル。

import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import requests
from pathlib import Path
from helper_functions import accuracy_fn
# Import tqdm for progress bar
from tqdm.auto import tqdm
from timeit import default_timer as timer
import pandas as pd
import random
import subprocess
import sys
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix


random.seed(42)
torch.manual_seed(42)
train_time_start_on_cpu = timer()

device = "cuda" if torch.cuda.is_available() else "cpu"

# トレーニングデータの設定
train_data = datasets.FashionMNIST(
    root = "data", # データのダウンロード先
    train = True, # トレーニングデータを取得する
    download = True, # ディスク上にデータが存在しない場合はダウンロードする
    transform = ToTensor(), # 画像はPIL形式で提供されるためTorchテンソルに変換する
    target_transform = None # ラベルも変換可能
)

# テストデータの設定
test_data = datasets.FashionMNIST(
    root = "data",
    train = False, # テストデータを取得する
    download = True,
    transform = ToTensor()
)

# サンプル数の確認
print(len(train_data.data), len(train_data.targets), len(test_data.data), len(test_data.targets))

# クラスの確認
class_names = train_data.classes
print(class_names)

# 画像の表示
image, label = train_data[0]
print(f"Image shape: {image.shape}") # 画像の形状は [1, 28, 28] (カラーチャンネル、高さ、幅)
# printについているfは文字列の中に変数や式を直接埋め込むことができる書き方
plt.imshow(image.squeeze()) # サイズが1の次元を削除する
plt.title(label);
plt.show()

# グレースケール
plt.imshow(image.squeeze(), cmap="gray")
plt.title(class_names[label])
plt.show()

# バッチサイズのハイパーパラメータを設定
BATCH_SIZE = 32

# データセットを反復可能オブジェクト（バッチ）に変換
train_dataloader = DataLoader(train_data, # 反復可能データに変換するデータセット
                                batch_size=BATCH_SIZE, # バッチあたりのサンプル数は？
                                shuffle=True # エポックごとにデータをシャッフルする？
                                )

test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


# model 0：ベースラインモデルの構築
# 平坦化レイヤーを作成
flatten_model = nn.Flatten()

# nn.Flatten()最初のレイヤーとして使用して最初のモデルを作成
class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape:int, hidden_units:int, output_shape:int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), # ニューラルネットワークはベクトル形式の入力
            nn.Linear(in_features=input_shape, out_features=hidden_units),# in_features = データサンプル内の特徴の数（784ピクセル）
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )
    
    def forward(self,x):
        return self.layer_stack(x)

model_0 = FashionMNISTModelV0(input_shape=784, # (28x28)
                                hidden_units=10, # 隠れ層のユニット数
                                output_shape=len(class_names)
)
model_0.to("cpu")

from timeit import default_timer as timer 
def print_train_time(start: float, end: float, device: torch.device = None):
    """開始時刻と終了時刻の差を表示します。

    引数:
    start (float): 計算開始時刻（timeit 形式を推奨）。
    end (float): 計算終了時刻。
    device ([type], オプション): 計算を実行するデバイス。デフォルトは None。

    戻り値:
    float: 開始時刻と終了時刻の間の時間（秒数、大きいほど長くなります）。
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time



# 損失関数とオプティマイザーをセットアップ
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

# トレーニングループの作成とバッチデータでのモデルのトレーニング
epochs = 3

# トレーニングとテストループ
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n")
    
    # トレーニング
    
    train_loss = 0
    # トレーニングバッチをループ処理するループを追加
    for batch,(x,y) in enumerate(train_dataloader):
        model_0.train()
        
        # 1. forward pass
        y_pred = model_0(x)
        
        # 2. calculate loss par batch
        loss = loss_fn(y_pred,y)
        train_loss += loss # エポックごとの損失を累積加算
        
        # 3. optimizer zero grad
        optimizer.zero_grad()
        
        # 4. loss backward
        loss.backward()
        
        # 5. optimizer step
        optimizer.step()
        
        # Print out how many samples have been seen
        if batch % 400 == 0:
            print(f"Looked at {batch * len(x)}/{len(train_dataloader.dataset)} samples")

        
    # 学習データローダーの長さ（バッチごと、エポックごと）で学習データローダーの総損失を割る
    train_loss /= len(train_dataloader)
    
    
    # テスト
    # 損失(loss)と精度(accuracy)を累積加算するための変数を設定
    
    test_loss, test_acc = 0, 0
    model_0.eval()
    with torch.inference_mode():
        for x, y in test_dataloader:
            
            # 1. forward pass
            test_pred = model_0(x)
            
            # 2. calculate loss (累積)
            test_loss += loss_fn(test_pred,y)# エポックごとの損失を累積加算
            
            # 3. calculate accuracy (予測値は y_true と同じである必要がある)
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
        
        # テストメトリクスの計算は torch.inference_mode() 内で行う必要がある
        # テスト損失の合計をテストデータローダーの長さ（バッチあたり）で割る
        test_loss /= len(test_dataloader)
        
        # 精度の合計をテストデータローダーの長さ（バッチあたり）で割る
        test_acc /= len(test_dataloader)
        
    ## 何が起こっているかを出力します
    print(f"\n訓練損失: {train_loss:.5f} | テスト損失: {test_loss:.5f}, テスト精度: {test_acc:.2f}%\n")

# 訓練時間を計算します
train_time_end_on_cpu = timer()
total_train_time_model_0 = print_train_time(start=train_time_start_on_cpu, 
                                            end=train_time_end_on_cpu, 
                                            device=str(next(model_0.parameters()).device))


# 予測

def eval_model(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, accuracy_fn, 
               device: torch.device = device):
    """data_loader でのモデル予測結果を含む辞書を返します。
    引数:
    model (torch.nn.Module): data_loader で予測を行うことができる PyTorch モデル。
    data_loader (torch.utils.data.DataLoader): 予測対象となるデータセット。
    loss_fn (torch.nn.Module): モデルの損失関数。
    acceleration_fn: モデルの予測値を真のラベルと比較するための精度関数。

    戻り値:
    (dict): data_loader でのモデル予測結果。
    """
    model.to(device)
    
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for x, y in data_loader:
            # モデルを使って予測を行う
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            
            # バッチごとに損失と精度の値を累積する
            loss += loss_fn(y_pred,y)
            acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
            # 精度には予測ラベルが必要 (logits -> pred_prob -> pred_labels)
            
        # lossとaccをスケールして、バッチあたりの平均loss/accを計算
        loss /= len(data_loader)
        acc /= len(data_loader)
    
    return {"model_name": model.__class__.__name__, # モデルがクラス付きで作成された場合にのみ機能
            "model_loss": loss.item(), "model_acc": acc}


# テストデータセットでモデル0の結果を計算する
model_0_results = eval_model(model=model_0, data_loader=test_dataloader, 
                            loss_fn=loss_fn, accuracy_fn=accuracy_fn)
print(model_0_results)


# model 1：非線形性を用いたより良いモデルの構築
# 非線形層と線形層を持つモデルを作成

class FashionMNISTModelV1(nn.Module):
    def __init__(self, input_shape:int, hidden_units:int, output_shape:int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),# 入力を単一のベクトルに平坦化
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU()
        )
    
    def forward(self, x:torch.Tensor):
        return self.layer_stack(x)

model_1 = FashionMNISTModelV1(input_shape=784, # 入力特徴量数
                              hidden_units=10, # 必要な出力クラス数
                              output_shape=len(class_names) # GPUが利用可能な場合、モデルをGPUに送信する
).to(device)
next(model_1.parameters()).device # モデルデバイスをチェックする

# 損失関数とオプティマイザーをセットアップ
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.1)

def train_step(model: torch.nn.Module,
            data_loader: torch.utils.data.DataLoader,
            loss_fn: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            accuracy_fn,
            device: torch.device = device):
    
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (x, y) in enumerate(data_loader):
        # Send data to GPU
        x, y = x.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(x)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                y_pred=y_pred.argmax(dim=1)) # Go from logits -> pred labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # エポックごとに損失と精度を計算し、出力
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

def test_step(data_loader: torch.utils.data.DataLoader,
            model: torch.nn.Module,
            loss_fn: torch.nn.Module,
            accuracy_fn,
            device: torch.device = device):
    
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval() # put model in eval mode
    # 推論コンテキストマネージャーをオンにする
    with torch.inference_mode(): 
        for x, y in data_loader:
            # Send data to GPU
            x, y = x.to(device), y.to(device)
            
            # 1. Forward pass
            test_pred = model(x)
            
            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                y_pred=test_pred.argmax(dim=1) # Go from logits -> pred labels
            )
        
        # メトリックを調整して出力
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

train_time_start_on_gpu = timer()

epochs = 3
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")
    train_step(data_loader=train_dataloader, 
        model=model_1, 
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn
    )
    test_step(data_loader=test_dataloader,
        model=model_1,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn
    )

train_time_end_on_gpu = timer()
total_train_time_model_1 = print_train_time(start=train_time_start_on_gpu, 
                                            end=train_time_end_on_gpu, 
                                            device=device)

model_1_results = eval_model(model=model_1, 
    data_loader=test_dataloader,
    loss_fn=loss_fn, 
    accuracy_fn=accuracy_fn,
    device=device) 

print(model_1_results)


# model 2：畳み込みニューラルネットワーク（CNN）の構築
# 入力層 -> [畳み込み層 -> 活性化層 -> プーリング層] -> 出力層
# [畳み込み層 -> 活性化層 -> プーリング層]の内容は、要件に応じて拡大したり、複数回繰り返したりすることができる

# 畳み込みニューラルネットワークを作成
class FashionMNISTModelV2(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                    out_channels=hidden_units,
                    kernel_size=3, # 画像を覆う正方形の大きさ
                    stride=1, # デフォルト
                    padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units,
                    kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)# デフォルトのストライド値はkernel_sizeと同じ
        )
        
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # in_features の形状はネットワークの各層が入力データの形状を圧縮して変更するため
            nn.Linear(in_features=hidden_units*7*7, 
                    out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        # print(x.shape)
        x = self.block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x

torch.manual_seed(42)
model_2 = FashionMNISTModelV2(input_shape=1, 
    hidden_units=10, 
    output_shape=len(class_names)).to(device)
model_2

# nn.Conv2d() は畳み込み層とも呼ばれます。
# nn.MaxPool2d() は最大プーリング層とも呼ばれます。
# 質問: nn.Conv2d() の「2d」は何を表していますか？

# 2d は2次元データを表します。つまり、画像には高さと幅の2次元があります。
# カラーチャンネルの次元もありますが、それぞれのカラーチャンネルの次元も高さと幅の2次元です。

# 他の次元データ（テキストの場合は1次元、3Dオブジェクトの場合は3次元など）には、
# nn.Conv1d() と nn.Conv3d() もあります。


# 画像バッチと同じサイズの乱数のサンプルバッチを作成
images = torch.randn(size=(32, 3, 64, 64)) # [batch_size, color_channels, height, width]
test_image = images[0] # テスト用に1枚の画像を取得します
print(f"Image batch shape: {images.shape} -> [batch_size, color_channels, height, width]")
print(f"Single image shape: {test_image.shape} -> [color_channels, height, width]") 
print(f"Single image pixel values:\n{test_image}")


# TinyVGGと同じ寸法の畳み込み層を作成する
# （パラメータを変更して挙動を確認してみましょう）
conv_layer = nn.Conv2d(in_channels=3,         # 入力チャネル数（RGBの3）
                       out_channels=10,       # 出力チャネル数（特徴マップの数）
                       kernel_size=3,         # カーネルサイズ（3x3の畳み込み）
                       stride=1,              # ストライド（移動ステップ数）
                       padding=0)             # パディング（周辺にゼロを追加）。"valid" や "same" を試すのもOK

# データを畳み込み層に通す（ただしこの例では戻り値を表示していません）
conv_layer(test_image)

# テスト画像にバッチ次元（先頭に1次元）を追加する
test_image.unsqueeze(dim=0).shape

# バッチ次元を追加したテスト画像を畳み込み層に通す
conv_layer(test_image.unsqueeze(dim=0)).shape

# 新しい畳み込み層 conv_layer_2 を定義
conv_layer_2 = nn.Conv2d(in_channels=3,       # 入力画像の色チャネル数（RGB）
                         out_channels=10,     # 出力特徴マップ数
                         kernel_size=(5, 5),  # カーネルサイズ（通常は正方形なのでタプルもOK）
                         stride=2,            # ストライド（2ピクセルずつ移動）
                         padding=0)           # パディングなし

# 単一画像を新しい conv_layer_2 に通す（nn.Conv2d の forward() メソッドが呼ばれる）
conv_layer_2(test_image.unsqueeze(dim=0)).shape


# 圧縮されていない寸法がある場合とない場合の元の画像の形状を出力
print(f"Test image original shape: {test_image.shape}")
print(f"Test image with unsqueezed dimension: {test_image.unsqueeze(dim=0).shape}")

# サンプルの nn.MaxPoo2d() レイヤーを作成します
max_pool_layer  =  nn.MaxPool2d ( kernel_size = 2 ) 

# データを conv_layer のみに渡します
test_image_through_conv  =  conv_layer ( test_image . unsqueeze ( dim = 0 )) 
print(f"Shape after going through conv_layer(): {test_image_through_conv.shape}")
# データを max pool レイヤーに渡します
test_image_through_conv_and_max_pool  =  max_pool_layer ( test_image_through_conv ) 
print(f"Shape after going through conv_layer() and max_pool_layer(): {test_image_through_conv_and_max_pool.shape}")

# 画像と同程度の次元数のランダムテンソルを作成
random_tensor = torch.randn(size=(1, 1, 2, 2))
print(f"Random tensor:\n{random_tensor}")
print(f"Random tensor shape: {random_tensor.shape}")

# Maxプール層を作成します
max_pool_layer = nn.MaxPool2d(kernel_size=2) # kernel_size の値を変更した場合の動作を確認
# ランダムテンソルをMaxプールに渡します
max_pool_tensor = max_pool_layer(random_tensor)

print(f"\nMax pool tensor:\n{max_pool_tensor} <- this is the maximum value from random_tensor")
print(f"Max pool tensor shape: {max_pool_tensor.shape}")

# Setup loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_2.parameters(),lr=0.1)


# Training and testing model_2
# Measure time
from timeit import default_timer as timer
train_time_start_model_2 = timer()

# Train and test model 
epochs = 3
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")
    train_step(data_loader=train_dataloader, 
        model=model_2, 
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn,
        device=device
    )
    test_step(data_loader=test_dataloader,
        model=model_2,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        device=device
    )

train_time_end_model_2 = timer()
total_train_time_model_2 = print_train_time(start=train_time_start_model_2,
                                            end=train_time_end_model_2,
                                            device=device)
# 畳み込み層と最大プーリング層のおかげで、パフォーマンスが少し向上した

# model_2 の結果を取得
model_2_results = eval_model(
    model=model_2,
    data_loader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn
)
model_2_results


# モデル結果と学習時間を比較
compare_results = pd.DataFrame([model_0_results, model_1_results, model_2_results])
compare_results["training_time"] = [total_train_time_model_0,
                                    total_train_time_model_1,
                                    total_train_time_model_2]
print(compare_results)

# Visualize our model results
compare_results.set_index("model_name")["model_acc"].plot(kind="barh")
plt.xlabel("accuracy (%)")
plt.ylabel("model");
plt.show()

# 最適なモデルでランダム予測を行い、評価する
def make_predictions(model: torch.nn.Module, data: list, device: torch.device = device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # サンプルを準備
            sample = torch.unsqueeze(sample, dim=0).to(device) # 次元を追加し、サンプルをデバイスに送信する
            
            # Forward pass
            pred_logit = model(sample)
            
            # 予測確率を取得 (logit -> 予測確率)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)
            # 注: ソフトマックス法は「バッチ」次元ではなく「ロジット」次元で実行
            # （この場合、バッチサイズは1なので、dim=0で実行できる）
            
            # 以降の計算のために、pred_probをGPUから取得
            pred_probs.append(pred_prob.cpu())
            
        # pred_probs をスタックしてリストをテンソルに変換
        return torch.stack(pred_probs)

test_samples = []
test_labels = []
for sample , label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)

# 最初のテストサンプルの形状とラベルを表示
print(f"Test sample image shape: {test_samples[0].shape}\nTest sample label: {test_labels[0]} ({class_names[test_labels[0]]})")

# モデル2を使用してテストサンプルの予測を行う
pred_probs= make_predictions(model=model_2, data=test_samples)

# 最初の2つの予測確率リストを表示する
pred_probs[:2]

# argmax() を使って予測確率を予測ラベルに変換
pred_classes = pred_probs.argmax(dim=1)
print(test_labels, pred_classes)


# 予測値をプロットする
plt.figure(figsize=(9, 9))
nrows = 3
ncols = 3
for i, sample in enumerate(test_samples):
    # サブプロットを作成する
    plt.subplot(nrows, ncols, i+1)

    # ターゲット画像をプロットする
    plt.imshow(sample.squeeze(), cmap="gray")

    # 予測ラベルを取得する（テキスト形式、例："Sandal")
    pred_label = class_names[pred_classes[i]]

    # 予測ラベルを取得する（テキスト形式、例："T-shirt")
    truth_label = class_names[test_labels[i]]

    # プロットのタイトルテキストを作成する
    title_text = f"Pred: {pred_label} | Truth: {truth_label}"

    # 等価性をチェックし、それに応じてタイトルの色を変更する
    if pred_label == truth_label:
        plt.title(title_text, fontsize=10, c="g") # 正しい場合は緑のテキスト
    else:
        plt.title(title_text, fontsize=10, c="r") # 間違っている場合は赤のテキスト
    plt.axis(False);

plt.show()


# さらなる予測評価のための混同行列の作成
# 学習済みモデルで予測を行う
y_preds = []
model_2.eval()
with torch.inference_mode():
    for x, y in tqdm(test_dataloader, desc="Making predictions"):
        # データとターゲットをターゲットデバイスに送信
        x, y = x.to(device), y.to(device)
        # forward pass を実行
        y_logit = model_2(x)
        # 予測をロジット -> 予測確率 -> 予測ラベル へと変換
        y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)
        # 注: ソフトマックス法は「バッチ」次元ではなく「ロジット」次元で実行します 
        # (この場合、バッチサイズは 32 なので、dim=1 で実行できます)
        # 評価のために予測を CPU に渡します
        y_preds.append(y_pred.cpu())
# 予測のリストをテンソルに連結
y_pred_tensor = torch.cat(y_preds)


# torchmetrics.ConfusionMatrix を使用して混同行列を作成
# mlxtend.plotting.plot_confusion_matrix() を使用して混同行列をプロット

# torchmetrics が存在するかどうかを確認し、存在しない場合はインストール
try:
    import torchmetrics
    import mlxtend
    print(f"mlxtend version: {mlxtend.__version__}")
    assert int(mlxtend.__version__.split(".")[1]) >= 19, "mlxtend version should be 0.19.0 or higher"
except (ImportError, AssertionError) as e:
    print("必要なパッケージが見つからない、またはバージョンが古いためインストールします...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torchmetrics", "-U", "mlxtend"])
    import torchmetrics
    import mlxtend
    print(f"mlxtend version (after install): {mlxtend.__version__}")
    assert int(mlxtend.__version__.split(".")[1]) >= 19, "mlxtend version should be 0.19.0 or higher"

# 混同行列を作成
# 混同行列インスタンスをセットアップし、予測値とターゲット値を比較
confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
confmat_tensor = confmat(preds=y_pred_tensor,
                        target=test_data.targets)

# 混同行列をプロット
fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(), 
    class_names=class_names, # 行と列のラベルをクラス名に変換します
    figsize=(10, 7)
);


# 最もパフォーマンスの高いモデルの保存と読み込み
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, # 必要に応じて親ディレクトリを作成
                exist_ok=True
)

# モデル保存パスを作成
MODEL_NAME = "03_pytorch_computer_vision_model_2.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# モデル状態辞書を保存
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_2.state_dict(), # state_dict() のみを保存します。学習したパラメータのみが保存される
            f=MODEL_SAVE_PATH)
# これで保存したモデル state_dict() ができたので、
# load_state_dict() と torch.load() を組み合わせてロードすることができる

# FashionMNISTModelV2 の新しいインスタンスを作成 (保存した state_dict() と同じクラス
# # 注: ここで指定した形状が保存済みのバージョンと異なる場合、モデルの読み込み時にエラーが発生
loaded_model_2 = FashionMNISTModelV2(input_shape=1, hidden_units=10, output_shape=10)

# 保存した state_dict() をロード
loaded_model_2.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

# モデルを GPU に送信
loaded_model_2 = loaded_model_2.to(device)

# 結果が互いに近いかどうかを確認します（非常に離れている場合はエラーが発生する可能性があります）

loaded_model_2_results = eval_model(
    model=loaded_model_2,
    data_loader=test_dataloader,
    loss_fn=loss_fn, 
    accuracy_fn=accuracy_fn
)

result = torch.isclose(torch.tensor(model_2_results["model_loss"]),
                torch.tensor(loaded_model_2_results["model_loss"]),
                atol=1e-08, # 絶対許容値
                rtol=0.0001) # 相対許容値

print(result)