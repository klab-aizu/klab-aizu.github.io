import torch
import time

device_cpu = torch.device("cpu")
device_gpu = torch.device("cuda")

#ex4 ランダムシードを に設定し0、演習 2 と 3 をもう一度実行します。
# ランダムシードを固定すれば、毎回同じ値が生成されるようになります。
# rand()　0から1の範囲で一様分布　randn()　平均 0 標準偏差 1 の正規分布(ガウス分布)
# CPU上のPyTorchの乱数生成を制御します。ただし、CUDA（GPU）には影響しません。

#torch.manual_seed(0)

#ex1
x1_cpu = torch.rand(size = (7,7)).to(device_cpu)
print(x1_cpu.shape)
print(x1_cpu)

#ex2
x2_cpu = torch.rand(size = (1,7)).to(device_cpu)
print(x2_cpu.shape)
print(x2_cpu)

#ex3
x2_cpu = x2_cpu.T
#行列の掛け算 @ は行と列が適切に整合している必要があり、
# .T（転置）をして行と列を入れ替えることで計算できるようになる
y_cpu = x1_cpu @ x2_cpu
print(y_cpu.shape)
print(y_cpu)

#ex5 GPU上でのランダムシードの固定
torch.cuda.manual_seed_all(1234)
#複数のGPUを使う場合
# torch.cuda.manual_seed_all(0)

#ex6
x1_gpu = torch.rand(size=(2,3), device=device_gpu)
print(x1_gpu.shape)
print(x1_gpu)

x2_gpu = torch.rand(size=(2,3), device=device_gpu)
print(x2_gpu.shape)
print(x2_gpu)

#ex7
x2_gpu = x2_gpu.T
y_gpu = x1_gpu @ x2_gpu
print(y_gpu.shape)
print(y_gpu)

# x1_gpu = torch.rand(size = (2,3)).to(device_gpu) ではなく
# x1_gpu = torch.rand(size=(2,3), device=device_gpu) になっている理由
# 前者は CPUで乱数を生成し、その後GPUへ転送する。
# そのため、GPU側のシードをを設定しても、CPUとGPUの乱数のシードは別扱いになるため、一貫性が保証されない。
# 一方で後者は、torch.cuda.manual_seed_all(1234)の影響を 直接 受けるため、毎回同じ値が出ます。
# つまり、乱数をどちら側で生成しているかの違い

#ex8 出力の最大最小
max = torch.max(y_gpu)
min = torch.min(y_gpu)
print(max,min)

#ex9 インデックスの最大最小
arg_max = torch.argmax(y_gpu)
arg_min = torch.argmin(y_gpu)
print(arg_max,arg_min)

#ex10
torch.cuda.manual_seed_all(7)
x1_gpu = torch.rand(size=(1,1,1,10), device=device_gpu)
print(x1_gpu.shape)
print(x1_gpu)


#サイズ 1 の次元をすべて削除
# なぜうれしいか　サイズが１の次元はデータ量が変わらないないので次元を減らすことができるため
x1_gpu = x1_gpu.squeeze()
print(x1_gpu.shape)
print(x1_gpu)
