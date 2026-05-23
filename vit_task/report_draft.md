# DNN実践課題2 第6題: Vision Transformer

## 1. 目的

Vision Transformer（ViT）は、画像を小さなpatchに分割し、それぞれのpatchをtokenとしてTransformer Encoderに入力する画像認識モデルである。本実験では、CIFAR-10を用いて小さなVision Transformerを学習し、畳み込みを中心とするCNNとは異なる画像分類の方法を確認する。

## 2. 方法

データセットにはCIFAR-10を用いた。入力画像は32x32 pixelであり、4x4 pixelのpatchに分割した。そのため、1枚の画像は64個のpatch tokenとして表現される。各patchを線形埋め込みし、class tokenとpositional embeddingを加えた後、Transformer Encoderに入力した。

モデルは、embedding dimensionを128、Transformer Encoderの層数を4、attention head数を4とした小型ViTである。分類にはclass tokenの出力を用いた。optimizerにはAdamWを用い、学習率はcosine scheduleで変化させた。学習にはCIFAR-10のtrain splitから20000枚、評価にはtest splitから2000枚を用いた。

## 3. 結果

学習曲線を以下に示す。

![training curve](figures/training_curve.png)

予測例を以下に示す。

![predictions](figures/predictions.png)

学習結果の詳細は `data/training_metrics.csv` に保存した。

## 4. 考察

ViTは画像をpatch列として扱うため、自然言語処理におけるTransformerと同様に、self-attentionによってtoken間の関係を学習できる。CNNでは局所的な畳み込みを積み重ねて受容野を広げるのに対し、ViTではself-attentionによりpatch間の大域的な関係を直接扱える点が特徴である。

一方で、ViTはCNNに比べて画像に対する帰納バイアスが弱いため、小規模データのみで一から学習すると高精度を得るのは難しい場合がある。より高い精度を得るには、より大きなデータセット、強いdata augmentation、またはImageNetなどで事前学習したViTをfine-tuningすることが有効だと考えられる。

## 5. まとめ

CIFAR-10を用いて小型Vision Transformerを学習した。画像をpatch token列として扱い、class tokenを用いて分類するViTの基本的な仕組みを確認した。
