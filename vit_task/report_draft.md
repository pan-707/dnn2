# DNN実践課題2 第6題: Vision Transformer

## 1. 目的

Vision Transformer（ViT）は、画像を小さなpatchに分割し、それぞれのpatchをtokenとしてTransformer Encoderに入力する画像認識モデルである。本実験では、CIFAR-10を用いて小さなVision Transformerを学習し、CNNとは異なる画像分類の方法を試した。

## 2. 方法

データセットにはCIFAR-10を用いた。入力画像は32x32 pixelであり、4x4 pixelのpatchに分割した。したがって、1枚の画像は64個のpatch tokenとして表現される。各patchを線形埋め込みし、class tokenとpositional embeddingを加えた後、Transformer Encoderに入力した。

モデルは、embedding dimensionを128、Transformer Encoderの層数を4、attention head数を4とした小型ViTである。分類にはclass tokenの出力を用いた。optimizerにはAdamWを用い、学習率はcosine scheduleで変化させた。学習にはCIFAR-10のtrain splitから20000枚、評価にはtest splitから2000枚を用いた。

## 3. 結果

学習曲線を以下に示す。

![training curve](figures/training_curve.png)

予測例を以下に示す。

![predictions](figures/predictions.png)

学習結果の詳細は `data/training_metrics.csv` に保存した。train lossは1 epoch目の1.9486から10 epoch目の1.3070まで低下した。train accuracyは0.2751から0.5274まで上昇した。validation側でも、val lossは1.8433から1.3488まで下がり、val accuracyは0.3020から0.5345まで上昇した。

最終的なval accuracyは約53.4%であり、CIFAR-10を十分に分類できるほど高い精度ではない。それでも、学習に伴って精度は着実に上がっていた。

## 4. 考察

ViTは画像をpatch列として扱うため、自然言語処理のTransformerと同じように、self-attentionでtoken間の関係を学習する。CNNでは局所的な畳み込みを積み重ねて受容野を広げるが、ViTではpatch間の大域的な関係を直接扱える。この点がCNNと大きく異なる。

今回の小型ViTでも、lossの低下とaccuracyの上昇が見られたので、patch tokenとclass tokenを用いた分類の仕組みは動作している。一方で、val accuracyは約53%にとどまった。ViTはCNNに比べて画像に対する帰納バイアスが弱く、小規模データを一から学習する場合は高精度を出しにくい。より高い精度を目指すなら、学習データを増やす、data augmentationを強くする、またはImageNetなどで事前学習したViTをfine-tuningする方法が必要になる。

## 5. まとめ

CIFAR-10を用いて小型Vision Transformerを学習した。画像をpatch token列として扱い、class tokenを用いて分類するViTの基本的な流れを実装した。10 epochの学習でval accuracyは0.3020から0.5345まで上昇し、Transformerによる画像分類が実際に学習できることが分かった。
