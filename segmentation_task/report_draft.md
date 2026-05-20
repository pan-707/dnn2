# DNN実践課題2 第3題: UNetによるSegmentation

## 1. 目的

Segmentationは、画像全体に1つのラベルを付ける分類とは異なり、各pixelに対してカテゴリを予測する課題である。本実験では、FoodSeg103データセットを用いてUNetを学習し、Encoder-Decoder型ネットワークによるpixel-wise predictionを確認する。

## 2. 方法

入力画像にはFoodSeg103のRGB画像を用い、教師信号には `Images/ann_dir` のpixel label画像を用いた。入力画像とmaskは128x128 pixelにリサイズした。maskでは、各pixelが背景または食材カテゴリのいずれかを表すため、モデルは各pixelに対して多クラス分類を行う。

モデルには小さなUNetを用いた。Encoderでは畳み込みとmax poolingにより特徴を抽出しながら空間解像度を下げ、Decoderでは転置畳み込みにより解像度を戻した。また、Encoder側の特徴をDecoder側へ連結するskip connectionを用いることで、物体の位置や輪郭情報を復元しやすくした。

損失関数にはCrossEntropyLossを用い、評価指標としてpixel accuracyとmean IoUを計算した。学習にはtrain splitから800枚、評価にはtest splitから200枚を用い、8 epoch学習した。

## 3. 結果

学習曲線を以下に示す。

![training curve](figures/training_curve.png)

予測結果を以下に示す。上から順に、入力画像、正解mask、予測mask、予測maskを入力画像に重ねたoverlayである。

![predictions](figures/predictions.png)

学習結果を見ると、train lossは1 epoch目の3.9939から8 epoch目の1.9687まで低下した。また、validation lossも3.4896から2.1297まで低下した。pixel accuracyは0.4193から最大0.5114まで上昇し、最終epochでは0.4993となった。mean IoUは1 epoch目の0.0300から最終epochの0.0539まで上昇した。

予測maskを見ると、入力画像に対してmaskを出力するpixel-wise prediction自体は実行できていることが分かる。一方で、mean IoUは0.05程度にとどまり、細かい食材領域や境界では誤りも多く残った。

## 4. 考察

UNetはEncoder-Decoder構造を持つため、画像全体の文脈を利用しながら、入力と同じ解像度のmaskを出力できる。また、skip connectionによってEncoderの浅い層に含まれる位置情報をDecoderへ渡せるため、単純なDecoderのみの場合よりも輪郭を復元しやすい。

今回の結果では、lossの低下とpixel accuracyの上昇から、モデルがFoodSeg103のsegmentationを一定程度学習したことが確認できた。一方で、mean IoUは低い値にとどまった。これはFoodSeg103が103種類の食材カテゴリを含む多クラスsegmentationであり、食材同士の境界が曖昧な場合や、同じ皿の中に複数カテゴリが細かく混在する場合があるためだと考えられる。

また、今回のモデルは小さなUNetであり、学習枚数を800枚、epoch数を8に制限している。そのため、実画像の多クラスsegmentationとしては十分な表現力と学習量が不足していた可能性がある。精度を上げるには、より大きなUNetやSegNet、pre-trained backbone、data augmentation、class imbalanceを考慮したlossを用いることが有効だと考えられる。

## 5. まとめ

FoodSeg103を用いてUNetによるsemantic segmentationを行った。train lossは3.9939から1.9687へ低下し、pixel accuracyも約0.42から約0.50へ上昇した。mean IoUは低かったものの、各pixelに対して多クラス分類を行い、入力画像と同じ解像度のmaskを出力するsegmentationの基本的な流れを確認できた。
