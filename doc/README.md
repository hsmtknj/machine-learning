# 機械学習モデルを作るにあたって

[【随時更新】Kaggleテーブルデータコンペできっと役立つTipsまとめ - ML_BearのKaggleな日常](https://naotaka1128.hatenadiary.jp/entry/kaggle-compe-tips)

# 探索的データ分析 (EDA)
データから**探索的に特徴**を探し**仮説**を立てる

##### テクニック
- `pandas-profiling` を使うと便利
  - 相関係数が1の特徴はないか探す
- 相関係数を可視化する
  - `pd.plotting.scatter_matrix(df)` 
- XGBoost や LightBGM の feature_impotance をみる


##### 確認するもの
- 統計量
  - 分布の中心的傾向
    - 平均値，中央値，最頻値，
  - 分布の散らばり
    - 分散，標準偏差
- Topological data analysis
  - 収束具合
  - 穴の個数
  - 穴が消えるまでの時間
- 訓練データと評価データの分布を比較確認
  - 分布の類似度を確認
- 不均衡データの確認
  - 不均衡の場合は undersampling や oversampling が必要か
- 関係なさそうな特徴はないかを確認


##### TODO
- [ ] 仮説の立て方


# データの前処理

### 主にデータ全体で実施する変換
  - 基本的に統計量が絡まない変換
    - 外れ値のクリッピング
      - 閾値で上下限を決定
        - e.g. 99%タイルでクリッピング
    - 表現変換
      - Label Encoding
        - One Hot Encoding
          - 線形モデルでは良い？
          - 木モデルでは不要なことが多い (遅くなる)
      - Frequency Encoding
        - 出現比率でエンコーディング
      - Target Encoding
      - rank transformation (kNN, NNに効く)
      - log transformation (Nに効く
      - Numerical -> Categorical (10代前半)
    - 特徴選択
      - 無関係または冗長な特徴を無視
        - 1種類の値しか入っていないカラムの削除
        - 相関係数が1のカラムの削除
      - Ridge
    - 特徴構築
      - 多項式展開など
      - PolynomialFeatures
        - 相互作用
      - 小数点だけ切り出した値
      - 木モデルスペシャル対応
        - 積み上げの値を普通に戻す (線形モデルでは不要)
        - 和・差・積など
        - 差や比率を直接表現できないため
    - 特徴増築
      - random noise
        - 計測誤差をカバーできる?

### 主にtrainデータで実施する変換
  - 統計量が絡む変換
     - trainデータで算出した統計量でvalidやtestも変換する
        - validやtestの情報をtrainに含めないため
  - Feature Scaling (※ 決定木には関係ない)
    - Mean Normalization
      - 平均値で引いて，分散(標準偏差)やレンジで割る
        - StandardScaler: 分散(標準偏差)
        - MinMaxScalers: レンジを使う
          - ヒストグラムの上位下位0.03%は除いても良い? (3σ)
  - 数値特徴のバケット化
  - 欠損値の補間
    - ヒストグラムなどを書いて見つける
    - feature generation の前に補間しない方が良い
    - XGBoost や LightGBM では NaN を直接扱える

    - Numerical Data
      - 適当な outlier
        - 木モデルに有効(-99999, MAX+1, MIN-1 など)
      - 平均や中央値
        - 木モデルにはネガティブ
        - 線形モデルで有効
    - Categorical Data
      - 欠損値を新たなカテゴリとして扱う
  - 次元削減
    - t-SNE
    - PCA
      - 決定木系のアルゴリズム(XGBoost等)が軸に斜めの表現を不得手としているため
    - word embedding
    - NMF
      - 木モデルではPCAより優秀?
    - LDA


# モデルの選定と学習
- アンサンブル学習(回帰)
  - Stacking (決定木，NN，k-近傍法)
  - Bagging (横に並べる)
    - **Random Forest**
  - Boosting (縦に並べる)
    - **AdaBoost**
    - Gradient Boosting
      - Gradient Boosting Tree
        - **XGBoost**
        - **LightGBM**
- RGF (Regularized Greedy Forest)
  - かなり遅い
  - FastRGF はちょっと早い
- Neural Network
- tSNE
  - sklearn の tSNE は遅いので，tsne パッケージ推奨

##### Random Forest
- 決定木を複数作って多数決や平均を取る
  - max_depth: 木の深さ
  - n_estimators: 決定木の数
- feature_importance を見ると不要な特徴が分かる
  - これで最初に特徴を落とすのもあり

##### LightGBM
- LightGBM 用のデータセットにすることで学習が高速化する
- 目的関数
  - rmse を使うことが多い
    - ノイズの分布に正規分布を仮定
- 重要パラメータ
  - num_leaves
    - 葉の数，max_depth と合わせて検討
    - 多め
  - min_data_in_leaf
    - ノードの最小データ数
    - 葉の数が多いと木が深く育つのを抑えられる
    - num_leaves の影響を受けやすい
  - @max_depth
    - 決定木の深さ
    - 3 ~ 8 くらいを設定
    - 大きな値を設定しすぎるとオーバーフィッティングを起こす
    - 他のパラメータと一緒に調整
  - @max_leaves
    - 末端ノード数
    - max_depth と相関が強い (`2**max_depth >= max_leaves`)
    - `int(.7 * max_depth ** 2)`
  - min_child_samples
    - データが大きいときは大きく
  - @colsample_bytree
    - 木を作成する際に使用する特徴量
    - 絡む数が多いときは 0.4 や 0.7
  - @leg_alpha
    - 0.1p
  - @leg_lambda
    - 0.1
  - @n_estimators
    - 100 - 1000
  - bagging_fraction
  - bagging_req
  - feature_fraction
  - max_bin
  - save_binary
  - learning_rate
  - num_iterations
  - min_sum_hessian_in_leaf
  - @lambda_l1
    - 0, 5, 15, 300
  - @lambda_l2
    - 0, 5, 15, 300
  - min_gain_to_split
- CV関数: Cross Validation

##### モデルのアンサンブル学習
  - single model の精度にこだわりすぎないように
  - random seed average
    - 同じモデルでも異なる random_state で学習
    - あとでそれらの投票や確率平均をとる
- GBMと相性が良いモデル
  - Random Forest
  - Neural Network
    - tree base アルゴリズムは決定境界が特徴軸に並行な矩形になるが，NNは曲線になる
  - Glmnet (Ridge や LASSO)
  - XGBoost と LightGBM をアンサンブルしてもあまり精度をあげられない

##### XGBoost の学習テクニック
  - 同じデータセットに対しては，最適な学習回数はサンプル数に対しておよそ線形
  - 単一モデルでは，学習不足・過学習で精度が落ちるが， random seed averaging では学習回数過学習気味でもあまり精度が落ちない
  - max_depth が多いほど averaging の効果が大きい．単一モデル検証で精度が拮抗していたらより深い方を選ぶのがベター
  - 学習率は小さくすれば精度が良くなるが，averaging ではその効果が減少するので，必要以上に小さくしない方が学習時間的に有利
  - colsample_by は小さくすると学習時間が線形に減少する．精度向上にも寄与することが多い．小さめの値を試してみよう．


##### Tuning
- 下記参照
[【随時更新】Kaggleテーブルデータコンペできっと役立つTipsまとめ - ML_BearのKaggleな日常](https://naotaka1128.hatenadiary.jp/entry/kaggle-compe-tips)


# モデル・ハイパーパラメータの選択 (Validation)
- 交差検証 (クロスバリデーション)
  - 統計量を用いる場合は交差確認の中で行う
- 不要特徴を抜いて学習
  - 全パターンやるのは時間がかかるので1つ2つくらい抜く
- Normal-Kfold
- Stratified-Kfold
  - 多クラスなら必須
  - 回帰でも1変数 k-means して Stratified するのもあり
- Adversarial Validation
- クロスバリデーション後に全トレーニングセットを使って再学習


# モデルの評価
- 評価指標にとらわれなくても良い
  - 回帰問題でも target を[0, 1]に正規化し，xentropy loss を最適化する方が精度が上がる場合がある

- 真値と予測値の差で評価
  - 決定係数
  - 二乗平均平方根誤差 (RMSE: Root Mean Squared Error)
    - 外れ値の影響を受けやすい
  - 平均絶対誤差 (MAE: Mean Absolute Error)
    - 外れ値の影響を受けにくい
- [特徴/メモ]
  - 上記は差の評価値であってモデルの複雑度が考慮されていない
  - 複雑なモデルほど小さくなりやすいので注意 (シンプルなモデルほど大きくなる)
  - 全く同じデータに対して計算した場合のみ相対的な大小が比較可能
    - 異なるデータセット間での指標の比較は意味がない
  - RMSEとMAEの見方 (詳細は「精度評価指標と回帰モデルの評価」を参照)
    - 誤差が正規分布に従う場合，$\frac{RMSE}{MAE} ≒ 1.253 (\sqrt{\frac{pi}{2}})$ が理想
    - 良いモデルが構築できたときは，モデルはデータの大まかな特徴を表現している
      - そのとき正規分布に従うようなノイズのみが誤差として残ると考えられる
    - 下記の場合は注意
      - 誤差の絶対値が大きい場合
      - 誤差が正規分布に従わない場合
      - データ数が少ない場合
- AIC
- yyplot
  - 図で可視化して評価
  - 横軸に実測値，縦軸に予測値をプロット
  - プロットが対角線付近に多く存在すれば良い予測ができていると言える
    - 立体的にみたとき対角線を中心としたドーム型 (正規分布) になっていると良い

- 過学習の確認
  - train data と test data の誤差を確認
  - epoch ごとの誤差を確認

# 不均衡データの取り扱い
- 適切な評価指標を使用する
  - recall
  - precision
- コスト考慮型学習
  - 少数派クラスのコスト多くする
    - LightGBM では class_weight を調整する
- サンプリング
  - 少数派クラスの oversampling
  - 多数派クラスの undersampling
    - きちんとバギングすること
    - 単にサンプルを減らしただけだとバイアスがかかる
  - SMOTE (効かないことが多い)
    - 少数派クラスのインスタンス同士に先を引く
    - 直線上で新しくインスタンスを生成
    - 過学習を完全に防げるわけではない
- 異常検知


# その他メモ
- 欠損値
  - 数値変数
    - 平均値・中央値で穴埋め
    - 結果が良かった方を採用
  - カテゴリ変数
    - 欠損値を新たなカテゴリとして扱う
    - K+1個のOne Hot Encoding で学習させる
- ツール
  - XGBoostは欠損値を欠損値として扱える
  - t-SNE

- TDA Topological Data Analysys
  - データを膨らませる
  - e.g 時間あたりの点の減り具合を確認する

- アンサンブル学習
  - Stacking

Googleコラボラトリー (GPU使える)
jupyter note book


# 情報収集
- slack: kaggler-ja に参加する
- kaggle の Kernels や Discussion を参照する
- kaggle テクニックで検索


# References
- pandas でヒストグラムを一気に可視化
[Pandasでヒストグラムの作成や頻度を出力する方法 - DeepAge](https://deepage.net/features/pandas-hist.html)


- 最初の特徴探し: 探索的データ分析(EDA)
[pandas + matplotlibで描くヒストグラムいろいろ - 天色グラフィティ](https://amalog.hateblo.jp/entry/various-histograms)


- アンサンブル学習について
[【入門】アンサンブル学習の代表的な２つの手法とアルゴリズム](https://spjai.com/ensemble-learning/)


- kaggleテクニック (1)
[Kaggleで世界11位になったデータ解析手法〜Sansan高際睦起の模範コードに学ぶ - エンジニアHub｜若手Webエンジニアのキャリアを考える！](https://employment.en-japan.com/engineerhub/entry/2018/08/24/110000)

- kaggleテクニック (2)
[Kaggleのテクニック](https://www.slideshare.net/yasunoriozaki12/kaggle-79541118)


- kaggleテクニック (3)
[top2%の私が教えるKaggleの極意, Bosch Production Line Performance \| RCO Ad-Tech Lab Blog](https://www.rco.recruit.co.jp/career/engineer/blog/kaggle-bosch/)


- kaggleテクニック (4) (分かりやすくて良い)
[【随時更新】Kaggleテーブルデータコンペできっと役立つTipsまとめ - ML_BearのKaggleな日常](https://naotaka1128.hatenadiary.jp/entry/kaggle-compe-tips)


- TDA
[Deep Learningを超える!? 数学を使った新しい分析手法TDAとは? \| AI専門ニュースメディア AINOW](https://ainow.ai/2017/06/13/114560/)


- 不均衡データの扱い
[機械学習における不均衡データの扱い方 - Qiita](https://qiita.com/r-takahama/items/631a59953fc20ceaf5d9)


- 交差検証 (クロスバリデーション)
[比較的少なめのデータで機械学習する時は交差検証 (Cross Validation) をするのです - Qiita](https://qiita.com/LicaOka/items/c6725aa8961df9332cc7)


- 決定木
[【機械学習実践】決定木で回帰してみる（回帰木）【Python】](https://rin-effort.com/2019/12/25/machine-learning-4/)


- XGBoost
[Embeddingについてまとめた。 - For Your ISHIO Blog](https://ishitonton.hatenablog.com/entry/2018/11/25/200332)
[xgboost: テーブルデータに有効な機械学習モデル - Qiita](https://qiita.com/msrks/items/e3e958c04a5167575c41)

- LightGBM のインストール
[Python: LightGBM を使ってみる - CUBE SUGAR CONTAINER](https://blog.amedama.jp/entry/2018/05/01/081842)


- LightGBM
<https://blog.amedama.jp/entry/2018/05/01/081842>
[【機械学習実践】LightGBMで回帰してみる【Python】](https://rin-effort.com/2019/12/29/machine-learning-6/)
[LightGBM 徹底入門 – LightGBMの使い方や仕組み、XGBoostとの違いについて](https://www.codexa.net/lightgbm-beginner/)
[NIPS2017読み会 LightGBM: A Highly Efficient Gradient Boosting Decision T…](https://www.slideshare.net/tkm2261/nips2017-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)

- LightGBM の GridSearchCV と .cv() の使い方
[GBDT系の機械学習モデルのパラメータチューニング奮闘記 ~ CatBoost vs LightGBM vs XGBoost vs Random Forests ~ その1 - Qiita](https://qiita.com/KROYO/items/6607bc77bb465f5e9a3a)

- LightGBM のパラメータチューニング
[勾配ブースティングで大事なパラメータの気持ち - nykergoto’s blog](https://nykergoto.hatenablog.jp/entry/2019/03/29/%E5%8B%BE%E9%85%8D%E3%83%96%E3%83%BC%E3%82%B9%E3%83%86%E3%82%A3%E3%83%B3%E3%82%B0%E3%81%A7%E5%A4%A7%E4%BA%8B%E3%81%AA%E3%83%91%E3%83%A9%E3%83%A1%E3%83%BC%E3%82%BF%E3%81%AE%E6%B0%97%E6%8C%81%E3%81%A1)
[Parameters Tuning — LightGBM 2.3.2 documentation](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html)
[LightGBM 徹底入門 – LightGBMの使い方や仕組み、XGBoostとの違いについて](https://www.codexa.net/lightgbm-beginner/)

- RandomForest
[【機械学習実践】ランダムフォレストで回帰してみる【Python】](https://rin-effort.com/2019/12/26/machine-learning-5/)
[Random Forest \| Instruction of chemoinformatics by funatsu-lab](https://funatsu-lab.github.io/open-course-ware/machine-learning/random-forest/)

- データ前処理
[機械学習のためのデータ前処理: オプションと推奨事項  \|  ソリューション  |  Google Cloud](https://cloud.google.com/solutions/machine-learning/data-preprocessing-for-ml-with-tf-transform-pt1?hl=ja)
[特徴量抽出 － カテゴリ変数と数値変数の取り扱い方 - 川雲さんの分析ブログ](http://rio-cloud.hatenablog.com/entry/2018/05/02/175319)

- 回帰モデルの評価指標
[回帰分析の評価指標 \| 決定係数や二乗平均平方根誤差などを利用して回帰モデルを評価](https://stats.biopapyrus.jp/glm/lm-evaluation.html)


- 回帰モデルの評価指標 (分かりやすくて良い)
[精度評価指標と回帰モデルの評価 \| Instruction of chemoinformatics by funatsu-lab](https://funatsu-lab.github.io/open-course-ware/basic-theory/accuracy-index/)


- AIC (Akaike information criterion) モデルの複雑さも考慮している評価指標
[AIC - 機械学習の「朱鷺の杜Wiki」](http://ibisforest.org/index.php?AIC)


- モデル開発にはあまり関係ないが，AIエンジニアになるためのロードマップ的なもの
[【保存版・初心者向け】独学でAIエンジニアになりたい人向けのオススメの勉強方法 - Qiita](https://qiita.com/tani_AI_Academy/items/4da02cb056646ba43b9d)


- 機械学習のコードスニペット的記事
[【コピペでOK】機械学習によく使うPythonのコード一覧まとめ \| AI入門ブログ](https://ai-kenkyujo.com/2019/09/09/kikaigakusyu-python/)


- 問題別の良く使われる機械学習手法
[代表的な機械学習手法一覧 - Qiita](https://qiita.com/tomomoto/items/b3fd1ec7f9b68ab6dfe2)


- PolynomialFeatures 使い方
[scikit-learnのPolynomialFeaturesで多項式と交互作用項の特徴量を作る - 静かなる名辞](https://www.haya-programming.com/entry/2019/07/14/043223)


- Undersampling + Bagging
[Python: Under-sampling + Bagging なモデルを簡単に作れる K-Fold を実装してみた - CUBE SUGAR CONTAINER](https://blog.amedama.jp/entry/under-bagging-kfold)
[Python: LightGBM で Under-sampling + Bagging したモデルを Probability Calibration してみる - CUBE SUGAR CONTAINER](https://blog.amedama.jp/entry/lgbm-under-bagging-proba-calibration)
[[Python]不均衡データ分類問題に対する定番アプローチ：under sampling + baggingを実装したよ - Qiita](https://qiita.com/nekoumei/items/6448a86a8d255619c4f4)

- 不均衡データの取り扱い
[【ML Tech RPT. 】第4回 不均衡データ学習 (Learning from Imbalanced Data) を学ぶ(1) - Sansan Builders Box](https://buildersbox.corp-sansan.com/entry/2019/03/05/110000)


- sklearn scoering 一覧
[【翻訳】scikit-learn 0.18 User Guide 3.3. モデル評価：予測の質を定量化する - Qiita](https://qiita.com/nazoking@github/items/958426da6448d74279c7)


- pickleで学習データの保存
[Python - SVM：グリッドサーチで最適化したモデルを保存するには？｜teratail](https://teratail.com/questions/219515)


- neg_mean_squred_error が負の理由
[半歩ずつ進める機械学習　～scikit-learn 回帰モデルの評価指標～ - Qiita](https://qiita.com/Mukomiz/items/fcdf1f6c2bc1e89bbc8b)


- クロスバリデーションについて
[scikit learn の Kfold, StratifiedKFold, ShuffleSplit の違い - 中野智文のブログ](http://nakano-tomofumi.hatenablog.com/entry/2018/01/15/172427)
[交差検証（cross validation／クロスバリデーション）の種類を整理してみた \| AIZINE（エーアイジン）](https://aizine.ai/cross-validation0910/)


- scatter_matrix について
[カテゴリデータがあってもpandasのscatter_matrixで表示するといいよって話 - EnsekiTT Blog](https://ensekitt.hatenablog.com/entry/2018/06/03/200000)


- imblearn について
[imbalanced-learn – サンプリング中に実行するPythonモジュールと、さまざまな手法でオーバーサンプリング – GitHubじゃ！Pythonじゃ！](https://githubja.com/scikit-learn-contrib/imbalanced-learn)