import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
from typing import Optional, Tuple, Dict
import numpy as np

warnings.filterwarnings('ignore')


@st.cache_data(ttl=3600)
def get_stock_data(symbol: str) -> Optional[Tuple[pd.DataFrame, str, float]]:
    """
    株価データを取得する関数
    
    Args:
        symbol: 株式銘柄コード
        
    Returns:
        データフレーム、企業名、現在の実際の株価のタプル、エラー時はNone
    """
    try:
        ticker = yf.Ticker(symbol)
        
        # 過去5年分のデータを取得
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5*365)
        
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            return None
            
        # 企業名を取得
        info = ticker.info
        company_name = info.get('longName', symbol)
        
        # Prophet用にデータを整形
        data = data.reset_index()
        data = data[['Date', 'Close']].copy()
        data.columns = ['ds', 'y']
        data['ds'] = data['ds'].dt.tz_localize(None)
        
        # 現在の実際の株価を保存（ログ変換前）
        current_actual_price = data['y'].iloc[-1]
        
        # 元の値を保存
        data['y_original'] = data['y'].copy()
        
        # log1pを使用して安定性を向上
        data['y'] = np.log1p(data['y'])
        
        return data, company_name, current_actual_price
        
    except Exception as e:
        st.error(f"データ取得エラー: {str(e)}")
        return None


@st.cache_data(ttl=3600)
def predict_stock_price(data: pd.DataFrame, periods: int = 1825) -> Optional[pd.DataFrame]:
    """
    Prophetを使用して株価を予測する関数
    
    Args:
        data: 株価データ
        periods: 予測期間（日数）
        
    Returns:
        予測結果のデータフレーム
    """
    try:
        # データが少ない場合の調整
        changepoint_prior_scale = 0.10 if len(data) > 500 else 0.05
        
        # データのコピーを作成
        df = data.copy()
        
        # Prophetモデルの設定
        model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.95
        )
        
        # モデルの学習
        model.fit(df[['ds', 'y']])
        
        # 予測実行
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        
        # expm1 = exp(x) - 1 を使用（log1pの逆変換）
        forecast['yhat'] = np.expm1(forecast['yhat'])
        forecast['yhat_lower'] = np.expm1(forecast['yhat_lower'])
        forecast['yhat_upper'] = np.expm1(forecast['yhat_upper'])
        
        # 非負制約を確実に適用
        forecast['yhat'] = forecast['yhat'].clip(lower=0)
        forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
        forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
        
        # 元の実績データも追加
        if 'y_original' in data.columns:
            forecast = forecast.merge(
                data[['ds', 'y_original']], 
                on='ds', 
                how='left'
            )
        
        return forecast
        
    except Exception as e:
        st.error(f"予測エラー: {str(e)}")
        return None


def create_prediction_chart(historical_data: pd.DataFrame, forecast: pd.DataFrame, 
                          company_name: str, symbol: str, current_actual_price: float) -> go.Figure:
    """
    予測結果のグラフを作成する関数
    
    Args:
        historical_data: 実績データ
        forecast: 予測データ
        company_name: 企業名
        symbol: 銘柄コード
        
    Returns:
        Plotlyグラフオブジェクト
    """
    fig = go.Figure()
    
    # 実績データの分離点を設定
    last_actual_date = historical_data['ds'].max()
    
    # 実績データをプロット（元の値を使用）
    if 'y_original' in historical_data.columns:
        actual_values = historical_data['y_original']
    else:
        actual_values = np.expm1(historical_data['y'])  # ログ変換を戻す
    
    fig.add_trace(go.Scatter(
        x=historical_data['ds'],
        y=actual_values,
        mode='lines',
        name='実績値',
        line=dict(color='blue', width=2)
    ))
    
    # 予測データをプロット（オレンジ破線）
    prediction_data = forecast[forecast['ds'] > last_actual_date]
    fig.add_trace(go.Scatter(
        x=prediction_data['ds'],
        y=prediction_data['yhat'],
        mode='lines',
        name='予測値',
        line=dict(color='orange', width=2, dash='dash')
    ))
    
    # 95%信頼区間をプロット
    fig.add_trace(go.Scatter(
        x=prediction_data['ds'],
        y=prediction_data['yhat_upper'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=prediction_data['ds'],
        y=prediction_data['yhat_lower'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(255, 165, 0, 0.2)',
        name='95%信頼区間',
        hoverinfo='skip'
    ))
    
    # 特定の予測時点にマーカーを追加
    prediction_points = {
        '1ヶ月後': 30,
        '3ヶ月後': 90,
        '6ヶ月後': 180,
        '1年後': 365,
        '3年後': 1095,
        '5年後': 1825
    }
    
    today = last_actual_date
    
    for label, days in prediction_points.items():
        target_date = today + timedelta(days=days)
        target_forecast = forecast[forecast['ds'].dt.date == target_date.date()]
        
        if not target_forecast.empty:
            price = target_forecast.iloc[0]['yhat']
            fig.add_trace(go.Scatter(
                x=[target_date],
                y=[price],
                mode='markers+text',
                marker=dict(size=10, color='red'),
                text=[f'{label}<br>¥{price:.2f}'],
                textposition='top center',
                name=label,
                showlegend=False
            ))
    
    min_act = float(np.nanmin(actual_values))
    max_act = float(np.nanmax(actual_values))

    pad_ratio = 0.25
    pad = (max_act - min_act) * pad_ratio

    y_min = max(0.0, min_act - pad)
    y_max = float(np.nanmax(np.r_[max_act, forecast['yhat'].values])) * 1.2
    
    # グラフのレイアウト設定
    fig.update_layout(
        title=f'{company_name} ({symbol}) - 株価予測',
        xaxis_title='日付',
        yaxis_title='株価 (¥)',
        yaxis=dict(
            range=[y_min, y_max],
            rangemode='tozero'  # 0を必ず含む
        ),
        hovermode='x unified',
        width=800,
        height=500,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig


def calculate_prediction_summary(historical_data: pd.DataFrame, 
                                forecast: pd.DataFrame, 
                                current_actual_price: float) -> Dict:
    """
    予測サマリーを計算する関数
    
    Args:
        historical_data: 実績データ
        forecast: 予測データ
        current_actual_price: 現在の実際の株価
        
    Returns:
        予測サマリーの辞書
    """
    last_actual_date = historical_data['ds'].max()
    
    prediction_points = {
        '1ヶ月後': 30,
        '3ヶ月後': 90,
        '6ヶ月後': 180,
        '1年後': 365,
        '3年後': 1095,
        '5年後': 1825
    }
    
    summary = {}
    
    for label, days in prediction_points.items():
        target_date = last_actual_date + timedelta(days=days)
        target_forecast = forecast[forecast['ds'].dt.date == target_date.date()]
        
        if not target_forecast.empty:
            predicted_price = target_forecast.iloc[0]['yhat']
            change_rate = ((predicted_price - current_actual_price) / current_actual_price) * 100
            
            summary[label] = {
                'predicted_price': predicted_price,
                'change_rate': change_rate,
                'upper_bound': target_forecast.iloc[0]['yhat_upper'],
                'lower_bound': target_forecast.iloc[0]['yhat_lower']
            }
    
    return summary


def main():
    """メイン関数"""
    st.set_page_config(
        page_title="株価予測アプリ",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 株価予測アプリ")
    st.markdown("**Prophet**を使用して株価の将来予測を行います")
    
    # サイドバーで銘柄入力
    with st.sidebar:
        st.header("銘柄設定")
        symbol = st.text_input(
            "銘柄コード",
            placeholder="例: AAPL, 7203.T",
            help="米国株: AAPL, GOOGL など / 日本株: 7203.T, 9984.T など"
        )
        
        predict_button = st.button("予測開始", type="primary")
        
        st.markdown("---")
        st.markdown("### 📝 使用方法")
        st.markdown("""
        1. 銘柄コードを入力
        2. 「予測開始」をクリック
        3. グラフと予測結果を確認
        """)
        
        st.markdown("### ⚠️ 免責事項")
        st.markdown("""
        この予測は統計的な分析に基づく参考情報です。
        投資判断は自己責任で行ってください。
        """)
    
    # メインエリア
    if predict_button and symbol:
        with st.spinner("AIが分析中..."):
            # データ取得
            result = get_stock_data(symbol)
            if result is None:
                st.error("❌ 銘柄が見つかりません。正しい銘柄コードを入力してください。")
                return
            
            historical_data, company_name, current_actual_price = result
            
            # 予測実行
            forecast = predict_stock_price(historical_data)
            if forecast is None:
                st.error("❌ 予測処理でエラーが発生しました。")
                return
        
        # 結果表示
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 予測グラフ")
            chart = create_prediction_chart(historical_data, forecast, company_name, symbol, current_actual_price)
            st.plotly_chart(chart, use_container_width=True)
        
        with col2:
            st.subheader("📋 予測サマリー")
            summary = calculate_prediction_summary(historical_data, forecast, current_actual_price)
            
            st.metric("現在価格", f"¥{current_actual_price:.2f}")
            
            for period, data in summary.items():
                predicted_price = data['predicted_price']
                change_rate = data['change_rate']
                
                delta_color = "normal" if change_rate >= 0 else "inverse"
                st.metric(
                    period,
                    f"¥{predicted_price:.2f}",
                    f"{change_rate:+.1f}%",
                    delta_color=delta_color
                )
            
            st.markdown("---")
            st.markdown("**信頼区間について**")
            st.markdown("予測値の周りの薄いオレンジ色の範囲は95%信頼区間を示しています。")
    
    elif predict_button and not symbol:
        st.warning("⚠️ 銘柄コードを入力してください。")
    
    else:
        # 初期画面
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🎯 予測対象期間")
            st.markdown("""
            - 1ヶ月後
            - 3ヶ月後  
            - 6ヶ月後
            - 1年後
            - 3年後
            - 5年後
            """)
        
        with col2:
            st.markdown("### 📈 対応銘柄の例")
            st.markdown("""
            **米国株**
            - AAPL (Apple)
            - GOOGL (Google)
            - MSFT (Microsoft)
            
            **日本株**
            - 6037.T (楽待)
            - 7203.T (トヨタ)
            - 9984.T (ソフトバンク)
            """)
        
        with col3:
            st.markdown("### 🔧 技術仕様")
            st.markdown("""
            - **予測モデル**: Prophet
            - **データソース**: Yahoo Finance
            - **信頼区間**: 95%
            - **キャッシュ**: 1時間
            """)


if __name__ == "__main__":
    main()