# DNN実践課題2 第5題: Word2Vecによる単語ベクトル演算

## 1. 目的

Word2VecやGloVeなどの単語分散表現では、単語を高次元ベクトルとして表現する。単語ベクトル空間では、意味的に近い単語が近い位置に配置されるだけでなく、`king - man + woman = queen` のような意味的な演算が可能になる場合がある。本実験では、公開されている事前学習済み単語ベクトルを用いて、単語の類似検索と単語ベクトル演算を確認する。

## 2. 方法

事前学習済み単語ベクトルとして、gensim-dataで提供されている `glove-wiki-gigaword-50` を用いた。このモデルはWikipediaとGigawordから学習された50次元の英語単語ベクトルである。

単語演算では、`positive_word - negative_word + add_word` の形でベクトルを計算し、cosine similarityが高い上位5語を取得した。また、いくつかの単語について近傍語も確認した。

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

一方で、`brother - man + woman` では期待した `sister` は上位5語に入らず、top1は `daughter` であった。また、`bigger - big + small` では `smaller` が上位5語に入ったが、top1は `larger` であった。

近傍語検索では、`king` の近傍に `prince`, `queen`, `emperor` が現れ、`japan` の近傍に `japanese`, `china`, `korea`, `tokyo` が現れた。`computer` の近傍には `computers`, `software`, `technology`, `internet` が含まれ、意味的に関連する語が近く配置されていることが確認できた。

## 4. 考察

単語ベクトルは、単語の共起関係に基づいて学習されるため、意味的に近い単語や似た文脈で使われる単語が近いベクトルになる。そのため、性別、国と首都、比較級のような関係が、ベクトル空間上の方向として表現される場合がある。

今回の結果では、王と女王、国と首都、比較級のような関係は比較的うまく表現された。特に `paris - france + japan = tokyo` のような結果から、単語ベクトル空間では「国から首都へ向かう方向」のような意味的関係がある程度保存されていると考えられる。

一方で、すべての単語演算が正しく成立するわけではない。`brother - man + woman` では `sister` ではなく `daughter` がtop1となった。これは、単語の多義性、家族関係語の文脈の近さ、学習コーパスの偏り、モデルの次元数などの影響を受けるためだと考えられる。また、`bigger - big + small` では `smaller` も上位に入ったが、top1は `larger` であり、単純な反対語関係が常に最上位に出るわけではなかった。

したがって、単語ベクトル演算は意味関係を可視化する有用な方法であるが、結果は統計的な共起に基づくものであり、人間の知識や論理演算と完全に一致するわけではない。

## 5. まとめ

事前学習済み単語ベクトルを用いて、近傍語検索と単語ベクトル演算を行った。7例中6例で期待した単語が上位5語以内に含まれ、単語間の意味関係がベクトルの差として表現される場合があることを確認した。一方で、期待と異なる結果もあり、単語ベクトル演算の解釈には注意が必要である。
