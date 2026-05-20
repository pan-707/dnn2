# Step 03: Segmentation

第3題では、研究室サーバ上のFoodSeg103を使ってUNetを学習します。

## 1. 最新版を取得

```bash
cd /export/space0/pan-p/projects/dnn2
git pull
```

## 2. FoodSeg103の場所を確認する

```bash
find /export/data/dataset/FoodSeg103 -maxdepth 3 -type d
find /export/data/dataset/FoodSeg103 -maxdepth 3 -type f | head -20
```

## 3. 実行する

```bash
chmod +x segmentation_task/run_server.sh
./segmentation_task/run_server.sh /export/data/dataset/FoodSeg103
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
