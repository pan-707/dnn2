# Step 03: Segmentation

第3題では、研究室サーバ上のPASCAL VOC形式データセットを使ってUNetを学習します。

## 1. 最新版を取得

```bash
cd /export/space0/pan-p/projects/dnn2
git pull
```

## 2. VOCデータセットの場所を探す

```bash
find /export -maxdepth 6 -type d -name SegmentationClass 2>/dev/null
```

出力例:

```text
/export/data/dataset/VOCdevkit/VOC2012/SegmentationClass
```

この場合、VOC rootは `/export/data/dataset/VOCdevkit/VOC2012` です。

## 3. 実行する

```bash
chmod +x segmentation_task/run_server.sh
./segmentation_task/run_server.sh /export/data/dataset/VOCdevkit/VOC2012
```

## 4. 結果を確認する

```bash
cat segmentation_task/data/training_metrics.csv
ls -lh segmentation_task/figures
```

## 5. GitHubへpush

```bash
git add segmentation_task docs/step_03_segmentation.md README.md .gitignore
git commit -m "Add segmentation experiment"
git push
```
