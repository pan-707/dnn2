# DNN実践課題2 第5題: Word2Vecによる単語ベクトル演算

## 1. 目的

Word2VecやGloVeなどの単語分散表現では、単語を高次元ベクトルとして表す。単語ベクトル空間では、意味が近い単語が近くに配置されるだけでなく、`king - man + woman = queen` のような演算が成り立つ場合がある。本実験では、公開されている事前学習済み単語ベクトルを使い、近傍語検索と単語ベクトル演算を試した。

## 2. 方法

事前学習済み単語ベクトルとして、gensim-dataで提供されている `glove-wiki-gigaword-50` を用いた。このモデルはWikipediaとGigawordから学習された50次元の英語単語ベクトルである。

単語演算では、`positive_word - negative_word + add_word` の形でベクトルを計算し、cosine similarityが高い上位5語を取得した。また、いくつかの単語について近傍語も調べた。

## 3. 結果

単語演算の結果は `data/analogies.csv` に保存した。近傍語の結果は `data/nearest_words.csv` に保存した。

単語演算では、7例中6例で期待した単語が上位5語以内に含まれた。代表的な結果を以下に示す。

| formula | expected | top1 | hit top5 |
| --- | --- | --- | --- |
| king - man + woman | queen | queen | True |
| brother - man + woman | sister | daughter | False |
| paris - france + japan | tokyo | tokyo | True |
| rome - italy + france | paris | paris | True |
| berlin - germany + italy | rome | rome | True |
| better - good + bad | worse | worse | True |
| bigger - big + small | smaller | larger | True |

`king - man + woman` では `queen` がtop1となり、cosine similarityは0.8524であった。国と首都の関係では、`paris - france + japan` から `tokyo`、`rome - italy + france` から `paris`、`berlin - germany + italy` から `rome` がtop1として得られた。また、比較級の関係では、`better - good + bad` から `worse` がtop1となった。

一方で、`brother - man + woman` では期待した `sister` は上位5語に入らず、top1は `daughter` であった。`bigger - big + small` では `smaller` が上位5語に入ったが、top1は `larger` だった。

近傍語検索では、`king` の近傍に `prince`, `queen`, `emperor` が現れ、`japan` の近傍に `japanese`, `china`, `korea`, `tokyo` が現れた。`computer` の近傍には `computers`, `software`, `technology`, `internet` が含まれており、意味的に関連する語が近くに配置されていた。

## 4. 考察

単語ベクトルは、単語の共起関係に基づいて学習される。そのため、意味が近い単語や似た文脈で使われる単語は、ベクトル空間でも近くなりやすい。今回の結果では、王と女王、国と首都、比較級のような関係が比較的うまく表れた。特に `paris - france + japan = tokyo` の結果から、「国から首都へ向かう方向」のような関係がベクトル空間にある程度保存されていることが分かる。

ただし、すべての単語演算が期待通りになるわけではない。`brother - man + woman` では `sister` ではなく `daughter` がtop1となった。家族関係語は文脈が近く、学習データ中での使われ方も重なりやすいため、単純な性別変換としてはうまく出なかったと考えられる。また、`bigger - big + small` でも `smaller` は上位に入ったが、top1は `larger` だった。単語ベクトル演算は便利だが、人間の論理そのものではなく、あくまで共起から得られた統計的な関係である。

## 5. まとめ

事前学習済み単語ベクトルを用いて、近傍語検索と単語ベクトル演算を行った。7例中6例で期待した単語が上位5語以内に含まれ、単語間の意味関係がベクトルの差として表れる場合があることが分かった。一方で、期待と異なる結果もあり、単語ベクトル演算の解釈には注意が必要である。
