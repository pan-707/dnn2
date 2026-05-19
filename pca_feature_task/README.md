# PCA Feature Task

第2題用の実験コードです。CIFAR-10画像からVGG16の4096次元特徴を抽出し、PCAで次元削減してk-meansクラスタリングを比較します。

## 研究室サーバで実行

第1題でCIFAR-10を `/export/space0/pan-p/data` にダウンロード済みの場合:

```bash
chmod +x pca_feature_task/run_server.sh
./pca_feature_task/run_server.sh /export/space0/pan-p/data
```

CIFAR-10がまだない場合:

```bash
python3 pca_feature_task/src/pca_dcnn_features.py \
  --data-root /export/space0/pan-p/data \
  --download
```

## 生成されるファイル

- `pca_feature_task/data/pca_dimensions.csv`
- `pca_feature_task/data/pca_cluster_summary.csv`
- `pca_feature_task/data/vgg16_fc7_features.npz`
- `pca_feature_task/figures/pca_dimensions.png`
- `pca_feature_task/figures/clusters_4096dim_k5.png`
- `pca_feature_task/figures/clusters_4096dim_k10.png`
- `pca_feature_task/figures/clusters_pca95_k5.png`
- `pca_feature_task/figures/clusters_pca95_k10.png`
- `pca_feature_task/figures/clusters_pca90_k5.png`
- `pca_feature_task/figures/clusters_pca90_k10.png`
- `pca_feature_task/figures/clusters_pca128_k5.png`
- `pca_feature_task/figures/clusters_pca128_k10.png`

## 実験内容

VGG16のfc7特徴は4096次元です。これをPCAで寄与率95%、寄与率90%、128次元に圧縮し、元の4096次元特徴と比較します。各特徴表現について、k=5とk=10でk-meansクラスタリングを行います。
