# Step 01: AutoEncoder

第1題では、CIFAR-10を使ってAutoEncoderを学習します。

## 1. サーバに入る

```bash
ssh ユーザー名@研究室サーバ
```

## 2. プロジェクトを置く

GitHubにまだpushしていない場合は、ローカルのファイルをサーバへコピーします。GitHubにpush済みなら、サーバでcloneします。

```bash
git clone GitHubのURL
cd リポジトリ名
```

## 3. 実行する

```bash
chmod +x autoencoder_task/run_server.sh
./autoencoder_task/run_server.sh /export/data/dataset
```

もしCIFAR-10の場所が違う場合は、最後のパスを変更します。

## 4. 結果を確認する

```bash
ls autoencoder_task/figures
cat autoencoder_task/data/training_loss.csv
cat autoencoder_task/data/cluster_summary.csv
```

確認したい図:

- `autoencoder_task/figures/reconstructions.png`
- `autoencoder_task/figures/kmeans_k5.png`
- `autoencoder_task/figures/kmeans_k10.png`

## 5. レポートに書く観点

- epochごとにMSEが下がったか
- 復元画像が元画像の色や形をどれくらい保っているか
- k-meansで似た画像が同じクラスタに集まったか
- CIFAR-10のラベルと完全一致しない理由は何か
