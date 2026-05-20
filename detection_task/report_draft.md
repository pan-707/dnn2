# DNN実践課題2 第4題: Pre-trained modelによる物体検出

## 1. 目的

物体検出は、画像中に存在する物体カテゴリを分類するだけでなく、その位置をbounding boxとして推定する課題である。本実験では、pre-trained Faster R-CNNとSSDLiteを用いて、研究室サーバ上のCOCO画像に対する物体検出を行い、2種類の検出モデルの特徴を比較する。

## 2. 方法

入力画像には研究室サーバ上のCOCOデータセットの画像を用いた。モデルにはtorchvisionで提供されているCOCO事前学習済みのFaster R-CNN ResNet-50 FPNとSSDLite MobileNetV3を用いた。スコア閾値は0.45とし、各画像につき上位の検出結果を可視化した。

Faster R-CNNはtwo-stage detectorであり、候補領域を生成した後に各候補を分類・回帰する。一方、SSD系のモデルはone-stage detectorであり、画像全体の特徴マップ上で直接bounding boxとカテゴリを予測する。そのため、一般にFaster R-CNNは精度を重視し、SSDは速度を重視する設計である。

## 3. 結果

Faster R-CNNによる検出結果を以下に示す。

![fasterrcnn predictions](figures/fasterrcnn_predictions.png)

SSDLiteによる検出結果を以下に示す。

![ssdlite predictions](figures/ssdlite_predictions.png)

検出数と上位クラスは `data/detection_summary.csv` に保存した。

## 4. 考察

Faster R-CNNは候補領域を明示的に生成してから分類を行うため、物体領域を比較的丁寧に検出できる。一方で、SSDLiteはone-stage detectorであり、軽量なMobileNetV3 backboneを用いるため、Faster R-CNNより高速に動作しやすい。ただし、小さい物体や重なりのある物体では検出漏れや誤検出が起こりやすい場合がある。

今回の実験では学習は行わず、COCOで事前学習済みのモデルをそのまま用いた。これにより、pre-trained modelを利用すれば、追加学習なしでも一般的な物体カテゴリを検出できることを確認できた。

## 5. まとめ

Pre-trained Faster R-CNNとSSDLiteを用いてCOCO画像の物体検出を行った。物体検出では、分類と位置推定を同時に行い、bounding boxとして結果を可視化できることを確認した。また、two-stage detectorとone-stage detectorでは、設計思想や精度・速度のバランスが異なることを確認した。
