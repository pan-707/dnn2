# Step 02: PCA

第2題では、VGG16の4096次元特徴をPCAで圧縮し、k-meansクラスタリングを比較します。

## 1. 最新版を取得

サーバのプロジェクト内で実行します。

```bash
cd /export/space0/pan-p/projects/dnn2
git pull
```

## 2. 実行する

第1題でCIFAR-10を `/export/space0/pan-p/data` にダウンロード済みなので、以下で実行します。

```bash
chmod +x pca_feature_task/run_server.sh
./pca_feature_task/run_server.sh /export/space0/pan-p/data
```

初回はVGG16のpre-trained weightをダウンロードするため、少し時間がかかる可能性があります。

## 3. 結果を確認する

```bash
cat pca_feature_task/data/pca_dimensions.csv
cat pca_feature_task/data/pca_cluster_summary.csv
ls -lh pca_feature_task/figures
```

## 4. GitHubへpush

```bash
git add pca_feature_task docs/step_02_pca.md README.md
git commit -m "Add PCA feature experiment"
git push
```
