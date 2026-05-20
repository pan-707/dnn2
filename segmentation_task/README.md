# Segmentation Task

第3題用の実験コードです。研究室サーバ上のPASCAL VOC形式データセットを使い、小さなUNetでsemantic segmentationを行います。

## データセット確認

VOC rootは、内部に以下のディレクトリを持つ場所です。

```text
JPEGImages/
SegmentationClass/
ImageSets/Segmentation/
```

場所が分からない場合はサーバで探します。

```bash
find /export -maxdepth 6 -type d -name SegmentationClass 2>/dev/null
```

たとえば `/export/data/dataset/VOCdevkit/VOC2012/SegmentationClass` が見つかった場合、VOC rootは `/export/data/dataset/VOCdevkit/VOC2012` です。

## 研究室サーバで実行

```bash
chmod +x segmentation_task/run_server.sh
./segmentation_task/run_server.sh /export/data/dataset/VOCdevkit/VOC2012
```

## 生成されるファイル

- `segmentation_task/data/training_metrics.csv`
- `segmentation_task/figures/training_curve.png`
- `segmentation_task/figures/predictions.png`
- `segmentation_task/checkpoints/unet_voc.pt`
