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
├── docs/
│   ├── step_01_autoencoder.md
│   ├── step_02_pca.md
│   └── step_03_segmentation.md
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

研究室サーバ上のPASCAL VOC形式データセットを使い、小さなUNetでsemantic segmentationを行います。

```bash
chmod +x segmentation_task/run_server.sh
./segmentation_task/run_server.sh /export/data/dataset/VOCdevkit/VOC2012
```

生成される主なファイル:

- `segmentation_task/data/training_metrics.csv`
- `segmentation_task/figures/training_curve.png`
- `segmentation_task/figures/predictions.png`

## Git Workflow

作業が一区切りついたら以下で確認します。

```bash
git status
git add .
git commit -m "Add autoencoder experiment"
git push
```

重いモデルファイル（`.pt`, `.pth`）はGitHubに上げない設定にしています。

第4題以降は、第3題が終わってから順番に追加します。
