# DNN実践課題2 第7題: CNN+LSTMによるSentence Generation

## 1. 目的

画像キャプション生成は、入力画像から自然言語の説明文を生成する課題である。本実験では、COCOの画像とcaptionを用いて、CNN encoderとLSTM decoderからなる簡単なimage captioningモデルを学習した。画像分類のように固定されたカテゴリを出すのではなく、画像特徴から単語列を出力する流れを試すことが目的である。

## 2. 方法

画像特徴の抽出には、ImageNetで事前学習済みのResNet-18を用いた。ResNet-18の畳み込み部分で画像特徴を取り出し、全結合層でLSTMに入力する埋め込み次元へ変換した。decoderにはLSTMを用い、画像特徴を最初の入力として与え、その後は単語を1つずつ入力して次の単語を予測した。

データセットには、研究室サーバ上のCOCO caption annotationを用いた。COCOはコピーせず、課題の指示通り `ln -s /export/data/dataset/COCO` でプロジェクト内にシンボリックリンクを作成して参照した。captionは小文字化し、英数字のtokenに分割して語彙を作成した。学習には5000個のimage-caption pairを用い、5 epoch学習した。損失関数にはCrossEntropyLossを用いた。

## 3. 結果

学習曲線を以下に示す。

![training curve](figures/training_curve.png)

生成例を以下に示す。

![generated captions](figures/generated_captions.png)

学習結果の詳細は `data/training_metrics.csv` に保存した。train lossは1 epoch目の5.3636から5 epoch目の3.6644まで低下した。lossが下がっているので、モデルはcaptionの単語列をある程度学習している。

一方で、生成文を見ると、8枚すべてに対してほぼ同じ文が出力された。例えば、野球少年、皿、キリン、シマウマ、時計塔、バイク、電車の画像に対しても、生成文は `a man is standing on a table with a table` となった。COCO captionでよく出る単語や文型を学習しているが、画像内容に合わせて文を変えるところまでは十分に学習できていない。

生成文の詳細は `data/generated_captions.csv` に保存した。

## 4. 考察

CNN+LSTMの構成では、CNNが画像を特徴ベクトルに変換し、LSTMがその特徴をもとに単語列を生成する。画像分類とは違い、出力は1つのカテゴリではなく、長さの変わる文である。この点で、caption generationは分類よりも難しい。

今回の実験ではtrain lossは下がったが、生成文は高頻度の定型文に偏った。これは、学習に用いたcaption数を5000個に制限したこと、epoch数が5と少ないこと、decoderがattentionを持たない単純なLSTMであることが原因だと考えられる。画像特徴がLSTMの最初に一度入るだけなので、文を生成している途中で画像中のどの領域を見るかを切り替えることもできない。

生成文が同じ文に寄ってしまった点は失敗にも見えるが、caption生成で起こりやすい問題でもある。モデルはまず「COCO captionらしい文」を覚え、その後で画像ごとの差を反映できるようになる。より良い結果にするには、学習データを増やす、epoch数を増やす、attention機構を入れる、あるいはTransformer decoderを使う方法が必要になる。

## 5. まとめ

COCO画像を用いてCNN+LSTMによるcaption生成を行った。train lossは5.3636から3.6644まで低下し、モデルがcaptionの単語列を学習していることは分かった。一方で、生成文はほとんど同じ表現に偏り、画像内容を正確に説明するところまでは届かなかった。CNNで画像特徴を抽出し、LSTMで文を生成する基本的な流れは実装できたが、実用的なcaption生成にはattentionやより多くの学習データが必要である。
