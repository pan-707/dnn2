# DNN実践課題2 第4題: Pre-trained modelによる物体検出

## 1. 目的

物体検出は、画像中の物体カテゴリを分類するだけでなく、その位置をbounding boxとして推定する課題である。本実験では、pre-trained Faster R-CNNとSSDLiteを用いて、研究室サーバ上のCOCO画像に対する物体検出を行い、2種類の検出モデルを比較した。

## 2. 方法

入力画像には研究室サーバ上のCOCOデータセットの画像を用いた。モデルには、torchvisionで提供されているCOCO事前学習済みのFaster R-CNN ResNet-50 FPNとSSDLite MobileNetV3を用いた。スコア閾値は0.45とし、各画像につき上位の検出結果を可視化した。

Faster R-CNNはtwo-stage detectorであり、まず候補領域を生成し、その後で各候補を分類・回帰する。これに対してSSD系のモデルはone-stage detectorであり、特徴マップ上でbounding boxとカテゴリを直接予測する。一般には、Faster R-CNNは精度寄り、SSDは速度寄りの設計である。

## 3. 結果

Faster R-CNNによる検出結果を以下に示す。

![fasterrcnn predictions](figures/fasterrcnn_predictions.png)

SSDLiteによる検出結果を以下に示す。

![ssdlite predictions](figures/ssdlite_predictions.png)

検出数と上位クラスは `data/detection_summary.csv` に保存した。Faster R-CNNでは、6枚の画像に対してそれぞれ5, 5, 2, 2, 1, 3個の物体が閾値0.45以上で検出された。検出されたカテゴリには、car, suitcase, broccoli, bowl, orange, giraffe, vase, potted plant, zebra, umbrella, person が含まれていた。

SSDLiteでは、同じ6枚の画像に対してそれぞれ5, 4, 1, 2, 1, 2個の物体が検出された。カテゴリはFaster R-CNNとほぼ共通しており、car, suitcase, bowl, broccoli, giraffe, vase, zebra, umbrella, person が検出された。ただし、Faster R-CNNがgiraffeを2個検出した画像ではSSDLiteは1個のみであり、umbrellaの画像でもFaster R-CNNは3個、SSDLiteは2個だった。

## 4. 考察

Faster R-CNNは候補領域を明示的に作ってから分類するため、物体領域を比較的丁寧に拾いやすい。今回も、giraffeやumbrellaのように複数の物体が写っている画像では、SSDLiteより検出数が多い場合があった。

一方、SSDLiteも主要な物体カテゴリは検出できていた。軽量なMobileNetV3 backboneを使うone-stage detectorでも、COCOで事前学習済みのモデルであれば、追加学習なしで実用的な検出結果が得られる。ただし、小さい物体や重なった物体では、検出漏れが起こりやすいと考えられる。

今回の実験では学習は行わず、COCOで事前学習済みのモデルをそのまま使った。そのため、検出できるカテゴリはCOCOで学習されたカテゴリに依存する。COCOにない物体を検出したい場合は、追加学習やfine-tuningが必要になる。

## 5. まとめ

Pre-trained Faster R-CNNとSSDLiteを用いてCOCO画像の物体検出を行った。物体検出では、分類と位置推定を同時に行い、bounding boxとして結果を可視化できることを確認した。Faster R-CNNはSSDLiteより検出数が多い場合があり、two-stage detectorとone-stage detectorの違いが結果にも表れた。
