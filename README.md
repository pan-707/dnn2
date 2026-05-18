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
├── docs/
│   └── step_01_autoencoder.md
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

## Git Workflow

作業が一区切りついたら以下で確認します。

```bash
git status
git add .
git commit -m "Add autoencoder experiment"
git push
```

重いモデルファイル（`.pt`, `.pth`）はGitHubに上げない設定にしています。

第2題以降は、第1題が終わってから順番に追加します。
