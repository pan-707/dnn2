# DNN Practice 2

DNN実践課題2の作業用リポジトリです。課題を順番に進め、各課題ごとにコード、実行方法、結果、レポート下書きを整理します。

## Structure

```text
.
├── autoencoder_task/        # 1. AutoEncoder
│   ├── src/
│   ├── figures/
│   ├── data/
│   ├── checkpoints/
│   ├── run_server.sh
│   ├── README.md
│   └── report_draft.md
├── pca_feature_task/        # 2. PCA on DCNN features
│   ├── src/
│   ├── figures/
│   ├── data/
│   ├── run_server.sh
│   ├── README.md
│   └── report_draft.md
├── segmentation_task/       # 3. Segmentation with UNet
│   ├── src/
│   ├── figures/
│   ├── data/
│   ├── checkpoints/
│   ├── run_server.sh
│   ├── README.md
│   └── report_draft.md
├── detection_task/          # 4. Detection with pretrained models
│   ├── src/
│   ├── figures/
│   ├── data/
│   ├── run_server.sh
│   └── report_draft.md
├── word2vec_task/           # 5. Word2Vec / word embedding
│   ├── src/
│   ├── data/
│   ├── models/
│   ├── run_server.sh
│   └── report_draft.md
├── vit_task/                # 6. Vision Transformer
│   ├── src/
│   ├── figures/
│   ├── data/
│   ├── checkpoints/
│   ├── run_server.sh
│   └── report_draft.md
├── caption_task/            # 7. Sentence Generation with CNN+LSTM
│   ├── src/
│   ├── figures/
│   ├── data/
│   ├── checkpoints/
│   ├── run_server.sh
│   └── report_draft.md
├── requirements.txt
└── README.md
```

## Task 1: AutoEncoder

研究室サーバで実行します。

```bash
chmod +x autoencoder_task/run_server.sh
./autoencoder_task/run_server.sh /export/data/dataset
```

生成される主なファイル:

- `autoencoder_task/figures/reconstructions.png`
- `autoencoder_task/figures/kmeans_k5.png`
- `autoencoder_task/figures/kmeans_k10.png`
- `autoencoder_task/data/training_loss.csv`
- `autoencoder_task/data/cluster_summary.csv`

## Task 2: PCA

VGG16の4096次元特徴をPCAで圧縮し、k-meansクラスタリングを比較します。

```bash
chmod +x pca_feature_task/run_server.sh
./pca_feature_task/run_server.sh /export/space0/pan-p/data
```

生成される主なファイル:

- `pca_feature_task/data/pca_dimensions.csv`
- `pca_feature_task/data/pca_cluster_summary.csv`
- `pca_feature_task/figures/pca_dimensions.png`
- `pca_feature_task/figures/clusters_*.png`

## Task 3: Segmentation

研究室サーバ上のFoodSeg103を使い、小さなUNetでsemantic segmentationを行います。

```bash
chmod +x segmentation_task/run_server.sh
./segmentation_task/run_server.sh /export/data/dataset/FoodSeg103
```

生成される主なファイル:

- `segmentation_task/data/training_metrics.csv`
- `segmentation_task/figures/training_curve.png`
- `segmentation_task/figures/predictions.png`

## Task 4: Detection

Pre-trained Faster R-CNNとSSDLiteでCOCO画像の物体検出を行います。

```bash
chmod +x detection_task/run_server.sh
./detection_task/run_server.sh /export/data/dataset/COCO
```

生成される主なファイル:

- `detection_task/data/detection_summary.csv`
- `detection_task/figures/fasterrcnn_predictions.png`
- `detection_task/figures/ssdlite_predictions.png`

## Task 5: Word2Vec

事前学習済み単語ベクトルを用いて、近傍語検索と単語ベクトル演算を行います。

```bash
chmod +x word2vec_task/run_server.sh
./word2vec_task/run_server.sh
```

生成される主なファイル:

- `word2vec_task/data/analogies.csv`
- `word2vec_task/data/nearest_words.csv`

## Task 6: Vision Transformer

CIFAR-10を使って、小型Vision Transformerを学習します。

```bash
chmod +x vit_task/run_server.sh
./vit_task/run_server.sh /export/space0/pan-p/data
```

生成される主なファイル:

- `vit_task/data/training_metrics.csv`
- `vit_task/figures/training_curve.png`
- `vit_task/figures/predictions.png`

## Task 7: Sentence Generation

COCO captionを使って、CNN encoder + LSTM decoderによる画像キャプション生成を行います。

```bash
chmod +x caption_task/run_server.sh
./caption_task/run_server.sh /export/data/dataset/COCO
```

生成される主なファイル:

- `caption_task/data/training_metrics.csv`
- `caption_task/data/generated_captions.csv`
- `caption_task/figures/training_curve.png`
- `caption_task/figures/generated_captions.png`

## Git Workflow

作業が一区切りついたら以下で確認します。

```bash
git status
git add .
git commit -m "Add autoencoder experiment"
git push
```

重いモデルファイル（`.pt`, `.pth`）はGitHubに上げない設定にしています。

第8題以降は、第7題が終わってから順番に追加します。
