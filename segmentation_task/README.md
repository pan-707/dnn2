# Segmentation Task

第3題用の実験コードです。研究室サーバ上のFoodSeg103を使い、小さなUNetでsemantic segmentationを行います。

## データセット確認

FoodSeg103 rootは、内部に以下のディレクトリを持つ場所です。

```text
Images/img_dir/train/
Images/img_dir/test/
Images/ann_dir/train/
Images/ann_dir/test/
ImageSets/
```

場所が分からない場合はサーバで探します。

```bash
find /export/data/dataset/FoodSeg103 -maxdepth 3 -type d
```

## 研究室サーバで実行

```bash
chmod +x segmentation_task/run_server.sh
./segmentation_task/run_server.sh /export/data/dataset/FoodSeg103
```

## 生成されるファイル

- `segmentation_task/data/training_metrics.csv`
- `segmentation_task/figures/training_curve.png`
- `segmentation_task/figures/predictions.png`
- `segmentation_task/checkpoints/unet_foodseg103.pt`
