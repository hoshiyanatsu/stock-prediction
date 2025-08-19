# 📈 株価予測アプリ

Prophetを使用した株価予測Webアプリケーションです。ユーザーが株式銘柄コードを入力すると、1ヶ月後、3ヶ月後、6ヶ月後、1年後、3年後、5年後の株価予測をグラフで表示します。

## 🚀 特徴

- **AI予測**: Meta開発のProphet時系列予測ライブラリを使用
- **インタラクティブグラフ**: Plotlyによる美しい可視化
- **国際対応**: 米国株・日本株の両方に対応
- **95%信頼区間**: 予測の不確実性を可視化
- **キャッシュ機能**: 1時間のキャッシュで高速レスポンス
- **レスポンシブ**: モバイル対応のデザイン

## 📋 技術スタック

- **フロントエンド**: Streamlit
- **予測モデル**: Prophet (Meta)
- **データソース**: yfinance (Yahoo Finance API)
- **グラフ**: Plotly
- **キャッシュ**: Streamlit @st.cache_data

## 🛠️ セットアップ

### 必要な環境

- Python 3.8+

### インストール

```bash
# リポジトリをクローン
git clone <repository-url>
cd stock-prediction

# 依存関係をインストール
pip install -r requirements.txt
```

### 実行

```bash
# アプリケーションを起動
streamlit run app.py
```

ブラウザで `http://localhost:8501` にアクセスしてください。

## 🐳 Docker Compose での実行

### 必要な環境
- Docker
- Docker Compose

### アプリケーションの起動

```bash
# バックグラウンドでサービスを起動
docker compose up -d

# ログを確認したい場合
docker compose logs -f

# サービスの状態を確認
docker compose ps
```

### アプリケーションの停止・削除

```bash
# サービスを停止
docker compose stop

# サービスを停止してコンテナを削除
docker compose down

# イメージも削除する場合
docker compose down --rmi all

# ボリュームも削除する場合
docker compose down -v

# すべて（イメージ、ボリューム、ネットワーク）を削除
docker compose down --rmi all -v --remove-orphans
```

### その他の便利なコマンド

```bash
# サービスを再起動
docker compose restart

# 特定のサービスのログを確認
docker compose logs stock-prediction

# コンテナ内でコマンドを実行
docker compose exec stock-prediction bash

# サービスをリビルドして起動
docker compose up --build -d
```

## 📊 使用方法

1. サイドバーで銘柄コードを入力
   - 米国株: `AAPL`, `GOOGL`, `MSFT` など
   - 日本株: `7203.T`, `9984.T` など

2. 「予測開始」ボタンをクリック

3. グラフと予測結果を確認
   - 青線: 実績値
   - オレンジ破線: 予測値
   - 薄いオレンジ領域: 95%信頼区間
   - 赤いマーカー: 各予測時点の価格

## 🎯 予測期間

- 1ヶ月後
- 3ヶ月後
- 6ヶ月後
- 1年後
- 3年後
- 5年後

## 📝 予測モデルの詳細

### Prophet設定

- `changepoint_prior_scale`: 0.10 (データが十分な場合)
- `yearly_seasonality`: True (年次季節性)
- `weekly_seasonality`: True (週次季節性)
- `interval_width`: 0.95 (95%信頼区間)

### データ処理

- 過去5年分の株価データを使用
- 外れ値の自動処理
- タイムゾーンの正規化

## ⚠️ 免責事項

**重要**: この予測は統計的な分析に基づく参考情報です。実際の投資判断は自己責任で行ってください。

- 予測結果は保証されるものではありません
- 市場の急激な変動には対応できない場合があります
- 投資にはリスクが伴います

## 🔧 開発者向け情報

### プロジェクト構成

```
stock-prediction/
├── app.py                 # メインアプリケーション
├── requirements.txt       # 依存ライブラリ
├── README.md             # プロジェクト説明
├── Dockerfile            # Docker設定
├── docker-compose.yml    # Docker Compose設定
├── logs/                 # ログファイル（Docker使用時）
└── .gitignore            # Git除外設定
```

### 主な関数

- `get_stock_data()`: 株価データ取得
- `predict_stock_price()`: Prophet予測実行
- `create_prediction_chart()`: グラフ作成
- `calculate_prediction_summary()`: 予測サマリー計算

### カスタマイズ

Prophet モデルのパラメータは `predict_stock_price()` 関数で調整できます：

```python
model = Prophet(
    changepoint_prior_scale=0.10,  # トレンド変化の感度
    yearly_seasonality=True,       # 年次季節性
    weekly_seasonality=True,       # 週次季節性
    interval_width=0.95           # 信頼区間
)
```

## 🚀 デプロイ

### Streamlit Cloud

1. GitHubにプッシュ
2. [Streamlit Cloud](https://streamlit.io/cloud)でデプロイ
3. 環境変数の設定は不要（完全無料で動作）

### Heroku

```bash
# Procfileを作成
echo "web: streamlit run app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile

# デプロイ
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

## 📈 動作確認済み銘柄

### 米国株
- AAPL (Apple Inc.)
- GOOGL (Alphabet Inc.)
- MSFT (Microsoft Corporation)
- AMZN (Amazon.com Inc.)
- TSLA (Tesla Inc.)

### 日本株
- 7203.T (トヨタ自動車)
- 9984.T (ソフトバンクグループ)
- 6758.T (ソニーグループ)
- 9434.T (ソフトバンク)
- 8306.T (三菱UFJフィナンシャル・グループ)

## 🤝 コントリビューション

プルリクエストやイシューの報告を歓迎します。

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。