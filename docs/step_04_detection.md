# Step 04: Detection

第4題では、pre-trained Faster R-CNNとSSDLiteを使ってCOCO画像の物体検出を行います。

## 1. 最新版を取得

```bash
cd /export/space0/pan-p/projects/dnn2
git pull
```

## 2. 実行する

```bash
chmod +x detection_task/run_server.sh
./detection_task/run_server.sh /export/data/dataset/COCO
```

初回はpre-trained weightをダウンロードするため、少し時間がかかる場合があります。

## 3. 結果を確認する

```bash
cat detection_task/data/detection_summary.csv
ls -lh detection_task/figures
```

生成される主な図:

- `detection_task/figures/fasterrcnn_predictions.png`
- `detection_task/figures/ssdlite_predictions.png`

## 4. GitHubへpush

```bash
git add detection_task docs/step_04_detection.md README.md
git commit -m "Add detection experiment"
git push
```
