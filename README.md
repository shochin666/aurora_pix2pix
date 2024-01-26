# AURORA_PIX2PIX

pix2pix を用いて [NenuFAR](https://pparc.gp.tohoku.ac.jp/hfvhf-20190921/) から提供されたデータからオーロラ電波を抽出するモデルを作成し、ノイズを含むオーロラ電波画像をノイズを除去した画像に再生成します。
![NenuFAR](https://pparc.gp.tohoku.ac.jp/wp-content/uploads/1NeneFAR.png)

このリポジトリではデータの前処理・訓練済みモデルの適用を行います。
学習は [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) のオリジナルのリポジトリを改造して **Ubuntu22.04** を使用しました。

このリポジトリはいわば実行ファイルの集合なので、データを格納するディレクトリを別に準備する必要があります。

## STRUCTURE

### **.**

ルートディレクトリ直下にスクリプトファイルを配置しています。主にこれらのファイルを実行することでリポジトリを利用可能です。

### **./pix2pix**

配下にモデルの学習で得られたパラメータのバイナリファイル(latest_net_G.pth)を配置します。ファイル名は latest_net_G.pth 以外は設定しないでください。

### **./src**

配下で自作モジュールを管理します。ここで作成したモジュールはスクリプトファイルでインポートされモジュールとして利用されます。もし、新たに追加したいモジュールがあれば **/src** 以下に作成し、それをスクリプトファイルで実行するようにするのが良いです。

## TRAINING

まず、データセットを作成する前にパッケージのインストールを以下のコマンドから行います。
**env_name** に任意の仮想環境名を指定し、ルートディレクトリ上で以下を実行します。

```
$ conda create -n [env_name] --file requirements.txt
```

Ubuntu22.04 を [WSL2](https://qiita.com/matarillo/items/61a9ead4bfe2868a0b86) 上で起動して訓練を行います。環境構築は [miniconda](https://docs.conda.io/projects/miniconda/en/latest/) を使って仮想環境を用いて行います。以下にパッケージリストを掲載予定です。

## SETTINGS

実際に画像を再生成する際は抽出したニューラルネットワークのパラメータを格納したバイナリファイル(latest\*net_G.pth)を参照し、**pix2pix/networks.py** にパラメータを適用してモデルを利用します。
まず、/pix2pix 直下にモデルの訓練で作成した **latest_net_G.pth** を配置します。これでモデルのパスが通りました。

次に、データを格納するための **data** ディレクトリをルートディレクトリと同じ階層に作成します。その後、以下の階層をもとに各サブディレクトリを作成していきます。

```
.
├── cdf
├── fits
├── out
    ├── cdf
    ├── fits
    ├── random
    │   ├── cdf
    │   ├── cdf1
    │   ├── cdf2
    │   ├── fits
    │   └── noise_jpg
    ├── separate
    ├── test
    │   ├── A
    │   └── B
    └── train
        ├── A
        └── B
```

用いるデータをセットします。**./cdf** もしくは **./fits** 配下に **[日付]/[その日付の CDF もしくは FITS ファイル]** としてファイルを以下のようにセットします。

```
├── 19910101
    └── srn_nda_routine_jup_edr_199101012204_199101020603_V12.cdf
```

以上でセッティングは終了です。

## APPLY

スクリプトファイルを実行し、モデルを使用します。それぞれのファイルの役割は以下の通りです。

- **main.py**
  基本的に使うことはありませんが、実験的に行いたいことがあれば活用してください。
- **calculate_filter.py**
  シグナルとノイズの切り分けの基準となる(filter)を見つけるためのグラフを描画するファイル。
- **calculate_rsn.py**
  calculate_filter.py で算出した filter を用いて SN 比を強度に直して計算するファイル。
- **prepare_for_training.py**
  pix2pix の学習のためのデータを準備するファイル。
- **reconstruct.py**
  filter をかけて画像のコントラストを調整するファイル。そのまま/data/out に保存される。
- **sanitize_training_data.py**
  prepare_for_training.py によって生成された画像を削除するファイル。
- **show_current_data.py**
  生データを画像として表示したい時に実行するファイル。
