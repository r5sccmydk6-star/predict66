import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import sqlite3
from datetime import date
import math
import os
import joblib

# --- Database Setup ---
DB_FILE = "stock_data.db"
from sqlalchemy import create_engine

db_engine = create_engine(f'sqlite:///{DB_FILE}')

# --- Keras Imports ---
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import AdamW  # <-- NEW: Using AdamW optimizer

# --- Dash App Imports ---
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input as dInput, Output as dOutput, State, no_update

# ===============================================================
# 📂 1. App Setup & Configuration
# ===============================================================

# --- Model & Feature Config ---
MODEL_DIR = "models_multi_step_v6_mc"  # <-- NEW: Version 6
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# <-- NEW: Monte Carlo Dropout samples for uncertainty
N_MC_SAMPLES = 50

app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
app.title = "Persistent Stock Predictor (INR)"
server = app.server

tickers_list = ['AAPL', 'MSFT', 'GOOG', 'TSLA', 'AMZN', 'META', 'NVDA', 'JPM', 'V', 'BTC-USD', 'ETH-USD']

# !!!!!!!!!!!!!!!!!! FEATURES LIST !!!!!!!!!!!!!!!!!!
FEATURES_LIST = [
    'Close',
    'Volume',
    'RSI',
    'MACD',
    'EMA',
    'ATR',
    'BB_UPPER',
    'BB_LOWER',
    'Pct_Change',
    'SMA_7',  # <-- NEW: 7-day Moving Average
    'SMA_30',  # <-- NEW: 30-day Moving Average
    'Close_Lag_1',  # <-- NEW: Lag features
    'Close_Lag_2',
    'Close_Lag_3',
    'Close_Lag_4',
    'Close_Lag_5'
]
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

TARGET_COL = 'Pct_Change'


# ===============================================================
# ⚙️ 2. Helper Functions (Data, Modeling, Plotting)
# ===============================================================

def add_features(df):
    """
    Calculates all technical indicators.
    NOW INCLUDES Lags and Moving Averages.
    """
    print("Calculating technical indicators (RSI, MACD, MAs, Lags...)...")
    try:
        df.loc[df['High'] == df['Low'], 'High'] = df['High'] + 1e-6
        df['RSI'] = df.ta.rsi(length=14)
        macd = df.ta.macd(fast=12, slow=26, signal=9)
        macd_col_name = None
        for col in macd.columns:
            if col.startswith('MACD_'):
                macd_col_name = col
                break
        if macd_col_name is None: raise ValueError("Could not find MACD column")
        df['MACD'] = macd[macd_col_name]

        df['EMA'] = df.ta.ema(length=20)
        df['ATR'] = df.ta.atr(length=14)

        bbands = df.ta.bbands(length=20, std=2)
        upper_col_name = None
        lower_col_name = None
        for col in bbands.columns:
            if col.startswith('BBU_'):
                upper_col_name = col
            if col.startswith('BBL_'):
                lower_col_name = col
        if upper_col_name is None or lower_col_name is None:
            raise ValueError("Could not parse Bollinger Band column names.")
        df['BB_UPPER'] = bbands[upper_col_name]
        df['BB_LOWER'] = bbands[lower_col_name]

        df['Pct_Change'] = df['Close'].pct_change()

        # <-- NEW: Add Moving Averages
        df['SMA_7'] = df.ta.sma(length=7)
        df['SMA_30'] = df.ta.sma(length=30)

        # <-- NEW: Add Lag Features
        for i in range(1, 6):
            df[f'Close_Lag_{i}'] = df['Close'].shift(i)

        df.dropna(inplace=True)  # This will drop NaNs from MAs and Lags
        return df

    except Exception as e:
        print(f"An error occurred in add_features: {e}")
        raise e


def get_stock_data_from_db(ticker):
    """
    Checks the database first, then downloads *only* what's needed.
    Auto-rebuilds if new features (Lags, MAs) are missing.
    """
    table_name = ''.join(e for e in ticker if e.isalnum())

    def load_and_check_db(t_name):
        print("Loading full dataset from database...")
        df = pd.read_sql(f"SELECT * FROM {t_name}", db_engine, index_col='Date')
        df.index = pd.to_datetime(df.index)
        # <-- NEW: This check will now fail if Lags/MAs are missing
        missing_cols = [col for col in FEATURES_LIST if col not in df.columns]
        if missing_cols:
            print(f"Database is stale. Missing columns: {missing_cols}")
            raise ValueError("Database schema mismatch. Forcing rebuild.")
        print("Database schema is up-to-date.")
        return df

    try:
        query = f"SELECT * FROM {table_name} ORDER BY Date DESC LIMIT 1"
        last_entry_df = pd.read_sql(query, db_engine)
        last_date = pd.to_datetime(last_entry_df['Date'].iloc[0])
        print(f"Found existing data for {ticker}. Last entry: {last_date.date()}")
        start_date = last_date + pd.Timedelta(days=1)
        today = date.today()

        if start_date.date() >= today:
            print("Data is up-to-date.")
            return load_and_check_db(table_name)

        print(f"Downloading new data for {ticker} from {start_date.date()} to {today}...")
        new_data_df = yf.download(ticker, start=start_date, auto_adjust=True)
        if isinstance(new_data_df.columns, pd.MultiIndex):
            new_data_df.columns = new_data_df.columns.get_level_values(0)
        if new_data_df.empty:
            print("No new data found.")
            return load_and_check_db(table_name)

        new_data_with_features = add_features(new_data_df)
        new_data_with_features.index.name = 'Date'
        data_to_save = new_data_with_features.reset_index()
        data_to_save.to_sql(table_name, db_engine, if_exists='append', index=False)
        print("New data saved to database.")
        return load_and_check_db(table_name)

    except Exception as e:
        print(f"No table found or rebuild forced: {e}. Downloading full history...")
        full_data_df = yf.download(ticker, start='2015-01-01', auto_adjust=True)
        if isinstance(full_data_df.columns, pd.MultiIndex):
            full_data_df.columns = full_data_df.columns.get_level_values(0)
        if full_data_df.empty:
            print(f"No data found for new ticker: {ticker}")
            return None

        full_data_with_features = add_features(full_data_df)
        full_data_with_features.index.name = 'Date'
        data_to_save = full_data_with_features.reset_index()
        data_to_save.to_sql(table_name, db_engine, if_exists='replace', index=False)
        print(f"Full history for {ticker} saved to database.")
        return full_data_with_features


def create_sequences_multi_step(data, seq_len, forecast_horizon, num_features):
    """
    Creates input sequences (X) and target sequences (y).
    (Unchanged)
    """
    X, y = [], []

    try:
        target_idx = FEATURES_LIST.index(TARGET_COL)
    except ValueError:
        print(f"FATAL ERROR: Target column '{TARGET_COL}' not in FEATURES_LIST.")
        return np.array(X), np.array(y)

    for i in range(seq_len, len(data) - forecast_horizon + 1):
        X.append(data[i - seq_len:i, :])
        y.append(data[i:i + forecast_horizon, target_idx])

    return np.array(X), np.array(y)


def get_or_train_model(ticker, feature_data, seq_len, forecast_horizon):
    """
    Loads or trains an LSTM-ONLY model.
    NOW uses AdamW optimizer.
    """
    num_features = len(feature_data.columns)  # <-- NEW: Will be 16

    safe_ticker = ''.join(e for e in ticker if e.isalnum())

    scaler_path = f"{MODEL_DIR}/{safe_ticker}_{seq_len}seq_{forecast_horizon}hor_{num_features}feat_scaler.joblib"
    lstm_path = f"{MODEL_DIR}/{safe_ticker}_{seq_len}seq_{forecast_horizon}hor_{num_features}feat_lstm.keras"

    if os.path.exists(scaler_path) and os.path.exists(lstm_path):
        print(f"Loading cached multi-step models for {ticker}...")
        scaler = joblib.load(scaler_path)
        model_lstm = load_model(lstm_path, compile=False)  # compile=False is faster
        # Re-compile with the optimizer (AdamW objects don't save well)
        model_lstm.compile(optimizer=AdamW(learning_rate=0.001), loss='mse')
        return scaler, model_lstm

    print(f"No cached models found. Training new multi-step model for {ticker}...")

    split_idx = int(len(feature_data) * 0.8)
    data_train = feature_data.iloc[:split_idx]
    data_test = feature_data.iloc[split_idx:]

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(data_train)
    scaled_test = scaler.transform(data_test)

    X_train, y_train = create_sequences_multi_step(scaled_train, seq_len, forecast_horizon, num_features)
    X_test, y_test = create_sequences_multi_step(scaled_test, seq_len, forecast_horizon, num_features)

    if X_train.size == 0 or X_test.size == 0:
        raise ValueError("Not enough data for train/test sequences. Try a smaller seq_len or forecast_horizon.")

    # These are good defaults. Use KerasTuner to find optimal ones.
    UNITS = 64
    DROPOUT_RATE = 0.2
    LEARNING_RATE = 0.001

    model_lstm = Sequential([
        Input(shape=(seq_len, num_features)),  # <-- NEW: Shape includes new features
        LSTM(UNITS, return_sequences=True),
        Dropout(DROPOUT_RATE),  # This Dropout is used for MC Uncertainty
        LSTM(UNITS),
        Dropout(DROPOUT_RATE),  # This Dropout is used for MC Uncertainty
        Dense(32),
        Dense(forecast_horizon)
    ])

    # <-- NEW: Using AdamW optimizer
    model_lstm.compile(optimizer=AdamW(learning_rate=LEARNING_RATE), loss='mse')

    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_lstm.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[es],
        verbose=0
    )

    print(f"Saving new multi-step models for {ticker}...")
    joblib.dump(scaler, scaler_path)
    model_lstm.save(lstm_path)

    return scaler, model_lstm


def evaluate_model(model_lstm, scaler, test_data, seq_len, forecast_horizon):
    """
    Generates predictions on the test set and calculates metrics.
    Metrics are based on the MEAN prediction.
    """
    num_features = len(test_data.columns)
    scaled_test = scaler.transform(test_data)

    X_test, y_test_scaled = create_sequences_multi_step(scaled_test, seq_len, forecast_horizon, num_features)

    if X_test.size == 0:
        return {}, pd.DataFrame()

    # --- Get the MEAN prediction for the test set ---
    # We run MC Dropout here too for a more robust "mean" estimate
    all_preds_scaled = []
    for _ in range(N_MC_SAMPLES):
        preds = model_lstm(X_test, training=True).numpy()
        all_preds_scaled.append(preds)

    # Get the mean prediction across all samples
    mean_preds_scaled = np.mean(all_preds_scaled, axis=0)

    # We only care about the first day's prediction for metrics
    y_test_scaled_day1 = y_test_scaled[:, 0]
    mean_preds_scaled_day1 = mean_preds_scaled[:, 0]

    target_idx = FEATURES_LIST.index(TARGET_COL)
    target_scale = scaler.scale_[target_idx]
    target_mean = scaler.mean_[target_idx]

    preds_real_pct_day1 = (mean_preds_scaled_day1 * target_scale) + target_mean

    # --- Convert Pct_Change predictions back to Price predictions ---
    actual_close_prices_day1 = test_data['Close'].iloc[seq_len:len(X_test) + seq_len]
    base_close_prices = test_data['Close'].iloc[seq_len - 1:len(X_test) + seq_len - 1].values

    predicted_close_prices_day1 = base_close_prices * (1 + preds_real_pct_day1)

    # --- Create results df with prices ---
    test_dates = test_data.index[seq_len:len(X_test) + seq_len]
    test_results_df = pd.DataFrame({
        'Date': test_dates,
        'Actual': actual_close_prices_day1,
        'Predicted': predicted_close_prices_day1  # This is the MEAN prediction
    })

    # --- Calculate metrics on PRICE (USD) ---
    mse = mean_squared_error(actual_close_prices_day1, predicted_close_prices_day1)
    rmse = math.sqrt(mse)
    mape = mean_absolute_percentage_error(actual_close_prices_day1, predicted_close_prices_day1)

    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape
    }

    return metrics, test_results_df


def make_future_forecast(model_lstm, scaler, full_feature_data, seq_len, forecast_horizon):
    """
    Generates ONE 90-day forecast using Monte Carlo Dropout for UNCERTAINTY.
    """
    num_features = len(full_feature_data.columns)

    try:
        target_idx = FEATURES_LIST.index(TARGET_COL)
    except ValueError as e:
        print(f"Error finding column indices: {e}")
        return pd.DataFrame()

    # 1. Get the last 60-day sequence from REAL data
    last_sequence_unscaled = full_feature_data.iloc[-seq_len:]
    last_sequence_scaled = scaler.transform(last_sequence_unscaled)
    current_sequence = last_sequence_scaled.reshape(1, seq_len, num_features)

    print(f"Starting stable multi-step forecast with {N_MC_SAMPLES} MC samples...")

    # 2. <-- NEW: Predict N_MC_SAMPLES times with Dropout ON
    all_preds_scaled = []
    for _ in range(N_MC_SAMPLES):
        # Call the model with training=True to enable dropout
        pred_scaled = model_lstm(current_sequence, training=True)
        all_preds_scaled.append(pred_scaled[0].numpy())  # Get the (1, 90) -> (90,)

    all_preds_scaled = np.array(all_preds_scaled)  # Shape is (50, 90)

    # 3. Inverse transform all 50x90 'Pct_Change' predictions
    target_scale = scaler.scale_[target_idx]
    target_mean = scaler.mean_[target_idx]

    all_preds_real_pct = (all_preds_scaled * target_scale) + target_mean  # Shape (50, 90)

    # 4. Convert all 50x90 'Pct_Change' preds to 'Close' prices
    all_preds_close_price = []
    last_close_price = full_feature_data['Close'].iloc[-1]

    for i in range(N_MC_SAMPLES):  # Loop over each sample
        sample_forecast = []
        current_last_price = last_close_price
        for j in range(forecast_horizon):  # Loop over each day
            pct_change = all_preds_real_pct[i, j]
            next_price = current_last_price * (1 + pct_change)
            sample_forecast.append(next_price)
            current_last_price = next_price  # Use the new price for the next calculation
        all_preds_close_price.append(sample_forecast)

    all_preds_close_price = np.array(all_preds_close_price)  # Shape (50, 90)

    # 5. <-- NEW: Calculate Mean, Upper, and Lower bounds
    mean_forecast = np.mean(all_preds_close_price, axis=0)
    lower_bound = np.percentile(all_preds_close_price, 2.5, axis=0)
    upper_bound = np.percentile(all_preds_close_price, 97.5, axis=0)

    print("Forecast loop complete.")

    last_date = full_feature_data.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon)

    # <-- NEW: Return a df with bounds
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecast': mean_forecast,
        'Lower_Bound': lower_bound,
        'Upper_Bound': upper_bound
    })

    return forecast_df


def create_plot(ticker, full_data_ohlcv, test_results_df, forecast_df, usd_to_inr_rate):
    """
    Creates the main Plotly figure.
    NOW includes UNCERTAINTY bounds.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # --- 1. Add Price Traces (CONVERTED TO INR) ---
    fig.add_trace(go.Scatter(
        x=full_data_ohlcv.index,
        y=full_data_ohlcv['Close'] * usd_to_inr_rate,
        mode='lines', name='Actual Price',
        line=dict(color='deepskyblue', width=2),
        hovertemplate='Date: %{x}<br>Actual Price: ₹%{y:,.2f}<extra></extra>'
    ), secondary_y=False)

    if not test_results_df.empty:
        fig.add_trace(go.Scatter(
            x=test_results_df['Date'],
            y=test_results_df['Predicted'] * usd_to_inr_rate,
            mode='lines', name='Test Prediction (Validation)',
            line=dict(color='orange', width=2, dash='dot'),
            hovertemplate='Date: %{x}<br>Test Predict: ₹%{y:,.2f}<extra></extra>'
        ), secondary_y=False)

    # --- 2. <-- NEW: Add Uncertainty Bounds ---
    # Add the Upper Bound first
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Upper_Bound'] * usd_to_inr_rate,
        mode='lines',
        line=dict(width=0, color='rgba(0, 255, 0, 0.2)'),
        name='95% Upper Bound',
        showlegend=False,
        hovertemplate='Date: %{x}<br>Upper: ₹%{y:,.2f}<extra></extra>'
    ), secondary_y=False)

    # Add the Lower Bound, and fill the area *up* to the Upper Bound
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Lower_Bound'] * usd_to_inr_rate,
        mode='lines',
        line=dict(width=0, color='rgba(0, 255, 0, 0.2)'),
        fillcolor='rgba(0, 255, 0, 0.2)',
        fill='tonexty',  # Fills the area between this trace and the one above (Upper)
        name='95% Confidence Interval',
        hovertemplate='Date: %{x}<br>Lower: ₹%{y:,.2f}<extra></extra>'
    ), secondary_y=False)
    # --- END NEW ---

    # Add the main (mean) forecast line
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Forecast'] * usd_to_inr_rate,
        mode='lines', name='Future Forecast (Mean)',
        line=dict(color='limegreen', width=2, dash='dash'),
        hovertemplate='Date: %{x}<br>Forecast: ₹%{y:,.2f}<extra></extra>'
    ), secondary_y=False)

    # --- 3. Add Volume Trace (No conversion) ---
    fig.add_trace(go.Bar(
        x=full_data_ohlcv.index, y=full_data_ohlcv['Volume'],
        name='Volume', marker_color='rgba(255, 255, 255, 0.2)',
        hovertemplate='Date: %{x}<br>Volume: %{y:,.0f}<extra></extra>'
    ), secondary_y=True)

    # --- 4. Add Shaded Regions ---
    test_start = test_results_df['Date'].min() if not test_results_df.empty else None
    test_end = test_results_df['Date'].max() if not test_results_df.empty else None
    forecast_start = forecast_df['Date'].min()
    forecast_end = forecast_df['Date'].max()

    if test_start:
        fig.add_vrect(
            x0=test_start, x1=test_end,
            fillcolor="rgba(255, 165, 0, 0.1)", line_width=0,
            annotation_text="Test Set", annotation_position="top left",
        )
    fig.add_vrect(
        x0=forecast_start, x1=forecast_end,
        fillcolor="rgba(0, 255, 0, 0.1)", line_width=0,
        annotation_text="Forecast", annotation_position="top left",
    )

    # --- 5. Style Layout (NOW IN INR) ---
    fig.update_layout(
        title=f"📈 {ticker} Advanced Prediction with Uncertainty (Price in INR)",
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Price (INR)",
        yaxis2_title="Volume",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=True,
        hovermode="x unified"
    )
    fig.update_yaxes(secondary_y=True, showgrid=False)
    fig.update_yaxes(type="linear", secondary_y=False)
    fig.update_yaxes(type="linear", secondary_y=True)

    return fig


def create_metrics_display(metrics, usd_to_inr_rate):
    """
    Creates an HTML component to display evaluation metrics.
    (Unchanged)
    """
    if not metrics:
        return html.Div("Evaluation metrics will appear here.", style={'textAlign': 'center'})

    mse_inr = metrics.get("MSE", 0) * (usd_to_inr_rate ** 2)
    rmse_inr = metrics.get("RMSE", 0) * usd_to_inr_rate
    mape = metrics.get("MAPE", 0)

    return html.Div([
        html.H4("📊 Model Performance (on 20% Test Set)"),
        html.Table([
            html.Tr([html.Th("Metric"), html.Th("Value")]),
            html.Tr([html.Td("RMSE (INR)"), html.Td(f"₹{rmse_inr:,.2f}")]),
            html.Tr([html.Td("MSE (INR)"), html.Td(f"₹{mse_inr:,.2f}")]),
            html.Tr([html.Td("Mean Abs. Percentage Error (MAPE)"), html.Td(f"{mape:.2%}")]),
        ], style={'margin': 'auto', 'width': '50%'})
    ], style={'textAlign': 'center', 'marginTop': '20px'})


# ===============================================================
# 📋 3. Dash App Layout
# ===============================================================
app.layout = html.Div([
    html.H1("🤖 Advanced Stock Predictor (with Database & INR)", style={'textAlign': 'center'}),
    html.P("V6: More Features, AdamW Optimizer, and MC Dropout Uncertainty",
           style={'textAlign': 'center', 'marginBottom': '30px'}),
    html.Hr(),
    html.Div([
        html.Div([
            html.Label("Stock Ticker:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='ticker-dropdown',
                options=[{'label': t, 'value': t} for t in tickers_list],
                value='AAPL',
                clearable=False
            ),
        ], style={'width': '20%', 'display': 'inline-block', 'padding': '10px'}),
        html.Div([
            html.Label("Sequence Length (Days):", style={'fontWeight': 'bold'}),
            dcc.Input(
                id='seq-len-input',
                type='number',
                value=60,
                min=10,
                max=120,
                step=10,
                style={'width': '100%'}
            ),
        ], style={'width': '20%', 'display': 'inline-block', 'padding': '10px'}),
        html.Div([
            html.Label("Forecast Horizon (Days):", style={'fontWeight': 'bold'}),
            dcc.Input(
                id='forecast-horizon-input',
                type='number',
                value=90,
                min=7,
                max=365,
                step=1,
                style={'width': '100%'}
            ),
        ], style={'width': '20%', 'display': 'inline-block', 'padding': '10px'}),
        html.Div([
            html.Button('Run Analysis', id='run-button', n_clicks=0,
                        style={'width': '100%', 'height': '40px', 'marginTop': '27px', 'backgroundColor': '#007BFF',
                               'color': 'white', 'border': 'none'})
        ], style={'width': '20%', 'display': 'inline-block', 'padding': '10px'}),
    ], style={'display': 'flex', 'justifyContent': 'center'}),
    html.Hr(),
    dcc.Loading(
        id="loading-spinner",
        type="circle",
        children=[
            html.Div(id='error-output', style={'color': 'red', 'textAlign': 'center', 'fontSize': '20px'}),
            html.Div(id='rate-info-output', style={'color': 'grey', 'textAlign': 'center', 'fontSize': '16px'}),
            dcc.Graph(id='stock-graph', style={'height': '600px'}),
            html.Div(id='metrics-output')
        ]
    )
], style={'width': '95%', 'margin': 'auto', 'fontFamily': 'Arial'})


# ===============================================================
# 📞 4. Dash Callback
# ===============================================================

@app.callback(
    [dOutput('stock-graph', 'figure'),
     dOutput('metrics-output', 'children'),
     dOutput('error-output', 'children'),
     dOutput('rate-info-output', 'children')],
    [dInput('run-button', 'n_clicks')],
    [State('ticker-dropdown', 'value'),
     State('seq-len-input', 'value'),
     State('forecast-horizon-input', 'value')]
)
def update_graph(n_clicks, ticker, seq_len, forecast_horizon):
    if n_clicks == 0:
        return go.Figure().update_layout(
            title="Select parameters and click 'Run Analysis'",
            template="plotly_dark"
        ), html.Div(), None, None

    # --- 1. Get INR Exchange Rate ---
    try:
        is_
        crypto = "-USD" in ticker.upper()
        is_usd_pair = "USD" in ticker.upper() and not is_crypto
        if is_crypto or is_usd_pair:
            usd_to_inr_rate = 1.0
            rate_info = f"{ticker} is a crypto/forex pair, displaying in base currency (USD)."
        else:
            rate_ticker = yf.Ticker("USDINR=X")
            rate_data = rate_ticker.history(period="1d")
            if rate_data.empty or rate_data['Close'].iloc[-1] is None:
                print("history() failed for USDINR=X, trying fast_info...")
                usd_to_inr_rate = rate_ticker.fast_info.get('lastPrice')
                if usd_to_inr_rate is None: raise Exception("fast_info rate was None")
            else:
                usd_to_inr_rate = rate_data['Close'].iloc[-1]
            rate_info = f"Using exchange rate: 1 USD = {usd_to_inr_rate:.2f} INR"
    except Exception as e:
        print(f"Could not fetch INR rate: {e}. Defaulting to 1.0 (USD).")
        usd_to_inr_rate = 1.0
        rate_info = "Could not fetch INR exchange rate. Displaying in USD."

    try:
        # --- 2. Get Data ---
        full_data_with_features = get_stock_data_from_db(ticker)
        if full_data_with_features is None:
            error_msg = f"Error: Could not retrieve or process data for ticker '{ticker}'."
            return no_update, no_update, error_msg, rate_info

        # <-- NEW: Make sure we only pass the features in the list
        feature_data = full_data_with_features[FEATURES_LIST]

        # --- 3. Load or Train Model ---
        scaler, model_lstm = get_or_train_model(ticker, feature_data, seq_len, forecast_horizon)

        # --- 4. Evaluate Model on Test Set ---
        split_idx = int(len(feature_data) * 0.8)
        data_test = feature_data.iloc[split_idx:]
        metrics, test_results_df = evaluate_model(model_lstm, scaler, data_test, seq_len, forecast_horizon)

        # --- 5. Make Future Forecast ---
        # <-- NEW: Pass the full feature_data, not the sliced one
        forecast_df = make_future_forecast(model_lstm, scaler, feature_data, seq_len, forecast_horizon)

        # --- 6. Create Plot (pass rate) ---
        fig = create_plot(ticker, full_data_with_features, test_results_df, forecast_df, usd_to_inr_rate)

        # --- 7. Create Metrics Display (pass rate) ---
        metrics_display = create_metrics_display(metrics, usd_to_inr_rate)

        return fig, metrics_display, None, rate_info  # No error

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()  # Use print_exc() for full traceback
        error_msg = f"An unexpected error occurred: {e}. Check console for details."
        return no_update, no_update, error_msg, rate_info


# ===============================================================
# 🚀 5. Run App
# ===============================================================
if __name__ == '__main__':
    print("🚀 Starting Advanced Stock Predictor (V6: MC Dropout, Lags, MAs, AdamW)")
    print(f"Database file is: {DB_FILE}")
    print(f"Model directory is: {MODEL_DIR}")
    print("Access it at: http://127.0.0.1:8050/")
    app.run(debug=True)