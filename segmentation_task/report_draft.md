# DNN実践課題2 第3題: UNetによるSegmentation

## 1. 目的

Segmentationは、画像全体に1つのラベルを付ける分類とは異なり、各pixelに対してカテゴリを予測する課題である。本実験では、FoodSeg103データセットを用いてUNetを学習し、Encoder-Decoder型ネットワークによるpixel-wise predictionを確認する。

## 2. 方法

入力画像にはFoodSeg103のRGB画像を用い、教師信号には `Images/ann_dir` のpixel label画像を用いた。入力画像とmaskは128x128 pixelにリサイズした。maskでは、各pixelが背景または食材カテゴリのいずれかを表すため、モデルは各pixelに対して多クラス分類を行う。

モデルには小さなUNetを用いた。Encoderでは畳み込みとmax poolingにより特徴を抽出しながら空間解像度を下げ、Decoderでは転置畳み込みにより解像度を戻した。また、Encoder側の特徴をDecoder側へ連結するskip connectionを用いることで、物体の位置や輪郭情報を復元しやすくした。

損失関数にはCrossEntropyLossを用い、評価指標としてpixel accuracyとmean IoUを計算した。

## 3. 結果

学習曲線を以下に示す。

![training curve](figures/training_curve.png)

予測結果を以下に示す。上から順に、入力画像、正解mask、予測mask、予測maskを入力画像に重ねたoverlayである。

![predictions](figures/predictions.png)

学習が進むにつれてlossは低下し、pixel accuracyとmean IoUは上昇した。予測maskを見ると、物体領域のおおまかな位置を推定できていることが分かる。一方で、細かい境界や小さい物体では誤りも残った。

## 4. 考察

UNetはEncoder-Decoder構造を持つため、画像全体の文脈を利用しながら、入力と同じ解像度のmaskを出力できる。また、skip connectionによってEncoderの浅い層に含まれる位置情報をDecoderへ渡せるため、単純なDecoderのみの場合よりも輪郭を復元しやすい。

ただし、今回のモデルは小さく、学習枚数とepoch数も制限しているため、FoodSeg103のような実画像の多クラスsegmentationでは完全な予測は難しい。精度を上げるには、より大きなUNetやSegNet、pre-trained backbone、data augmentationを用いることが有効だと考えられる。

## 5. まとめ

FoodSeg103を用いてUNetによるsemantic segmentationを行った。各pixelに対して多クラス分類を行い、入力画像と同じ解像度のmaskを出力できることを確認した。
