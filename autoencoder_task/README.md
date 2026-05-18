# AutoEncoder Task

第1題用の実験コードです。CIFAR-10画像をAutoEncoderで復元し、encoderのbottleneck特徴をk-meansでクラスタリングします。

## 研究室サーバで実行

```bash
cd プロジェクトの場所
chmod +x autoencoder_task/run_server.sh
./autoencoder_task/run_server.sh /export/data/dataset
```

CIFAR-10がその場所にない場合は、研究室サーバ上でデータセットの場所を探して、上の引数を変更してください。

## 生成されるファイル

- `autoencoder_task/figures/reconstructions.png`
- `autoencoder_task/figures/kmeans_k5.png`
- `autoencoder_task/figures/kmeans_k10.png`
- `autoencoder_task/data/training_loss.csv`
- `autoencoder_task/data/cluster_summary.csv`
- `autoencoder_task/checkpoints/autoencoder.pt`

## メモ

`train_mse` がepochごとに下がっていれば、入力画像を復元する学習が進んでいます。k-meansの図では、同じクラスタに見た目の似た画像が集まっているかを観察します。
