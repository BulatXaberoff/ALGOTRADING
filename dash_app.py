import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import trino
import pandas as pd

COLORS = {
    "bg_main": "#0a0f1e",
    "bg_panel": "#0d1526",
    "bg_header": "#0f1a2e",
    "border": "#1a2a4a",
    "blue": "#4fc3f7",
    "green": "#00e676",
    "red": "#ef5350",
    "muted": "#546e7a",
    "white": "#e0e0e0",
    "amber": "#ffb300",
    "purple": "#ce93d8",
}


def get_candle_data(date_from, date_to):
    conn = trino.dbapi.connect(host="trino", port=8080, user="admin", catalog="minio", schema="moex_data")
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT start_time, open_price, high_price, low_price, close_price, volume
        FROM minio.moex_data.gazp_view
        WHERE start_time >= TIMESTAMP '{date_from} 00:00:00'
          AND start_time <= TIMESTAMP '{date_to} 23:59:59'
        ORDER BY start_time ASC
    """)
    rows = cursor.fetchall()
    conn.close()
    return pd.DataFrame(rows, columns=["begin", "open", "high", "low", "close", "volume"])


def get_indicator_data(date_from, date_to):
    conn = trino.dbapi.connect(host="trino", port=8080, user="admin", catalog="minio", schema="feature_data_gazp_agg")
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT start_time, close_price, sma_5, sma_10, sma_20,
               bb_upper, bb_lower, rsi_14, rsi_7,
               macd, macd_signal, macd_hist,
               atr_14, vol_ratio_20, rsi_zone,
               volume, is_high_volume
        FROM minio.feature_data_gazp_agg.v_superset_dashboard
        WHERE start_time >= TIMESTAMP '{date_from} 00:00:00'
          AND start_time <= TIMESTAMP '{date_to} 23:59:59'
        ORDER BY start_time ASC
    """)
    rows = cursor.fetchall()
    conn.close()
    return pd.DataFrame(rows, columns=[
        "time", "close", "sma_5", "sma_10", "sma_20",
        "bb_upper", "bb_lower", "rsi_14", "rsi_7",
        "macd", "macd_signal", "macd_hist",
        "atr_14", "vol_ratio_20", "rsi_zone",
        "volume", "is_high_volume"
    ])


def get_last_info():
    conn = trino.dbapi.connect(host="trino", port=8080, user="admin", catalog="minio", schema="feature_data_gazp_agg")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT start_time, close_price, rsi_14, macd, rsi_zone, atr_14
        FROM minio.feature_data_gazp_agg.v_superset_dashboard
        ORDER BY start_time DESC
        LIMIT 1
    """)
    row = cursor.fetchone()
    conn.close()
    return row


app = dash.Dash(__name__, suppress_callback_exceptions=True)

HEADER = html.Div([
    html.Div([
        html.Span("GAZP RX", style={"color": COLORS["blue"], "fontSize": "20px", "fontWeight": "500", "fontFamily": "monospace"}),
        html.Span("MOEX · Газпром", style={"color": COLORS["muted"], "fontSize": "12px", "marginLeft": "12px", "fontFamily": "monospace"}),
    ]),
    html.Div([
        html.Button("СВЕЧИ", id="tab-candle", n_clicks=0, style={
            "background": "#1a3a6a", "border": f"1px solid {COLORS['blue']}",
            "color": COLORS["blue"], "fontFamily": "monospace", "fontSize": "11px",
            "padding": "4px 16px", "cursor": "pointer", "borderRadius": "2px", "marginRight": "4px",
        }),
        html.Button("ИНДИКАТОРЫ", id="tab-indicator", n_clicks=0, style={
            "background": "transparent", "border": f"1px solid {COLORS['border']}",
            "color": COLORS["muted"], "fontFamily": "monospace", "fontSize": "11px",
            "padding": "4px 16px", "cursor": "pointer", "borderRadius": "2px",
        }),
    ]),
], style={
    "background": COLORS["bg_header"],
    "borderBottom": f"1px solid {COLORS['border']}",
    "padding": "12px 24px",
    "display": "flex",
    "justifyContent": "space-between",
    "alignItems": "center",
})

CONTROLS = html.Div([
    html.Div([
        html.Span("ПЕРИОД:", style={"color": COLORS["muted"], "fontSize": "11px", "fontFamily": "monospace", "marginRight": "8px"}),
        dcc.DatePickerRange(
            id="date-range",
            start_date="2026-05-01",
            end_date="2026-05-16",
            display_format="DD.MM.YYYY",
        ),
    ], style={"display": "flex", "alignItems": "center"}),
    html.Div([
        *[html.Button(label, id=f"btn-{val}", n_clicks=0, style={
            "background": "#1a2a4a", "border": f"1px solid {COLORS['border']}",
            "color": COLORS["muted"], "fontFamily": "monospace", "fontSize": "11px",
            "padding": "4px 10px", "cursor": "pointer", "borderRadius": "2px",
        }) for label, val in [("1W", "1w"), ("1M", "1m"), ("3M", "3m"), ("6M", "6m"), ("1Y", "1y"), ("ALL", "all")]],
    ], style={"display": "flex", "gap": "4px"}),
], style={
    "background": COLORS["bg_panel"],
    "borderBottom": f"1px solid {COLORS['border']}",
    "padding": "10px 24px",
    "display": "flex",
    "justifyContent": "space-between",
    "alignItems": "center",
})

app.layout = html.Div([
    HEADER,
    html.Div(id="stat-bar", style={
        "background": COLORS["bg_panel"],
        "borderBottom": f"1px solid {COLORS['border']}",
        "display": "flex",
        "padding": "10px 24px",
        "gap": "8px",
    }),
    CONTROLS,
    html.Div(id="page-content"),
    html.Div([
        html.Span("GAZP · MOEX · Данные из Trino/MinIO", style={"color": COLORS["muted"], "fontSize": "10px", "fontFamily": "monospace"}),
    ], style={"background": COLORS["bg_header"], "borderTop": f"1px solid {COLORS['border']}", "padding": "8px 24px"}),
], style={"background": COLORS["bg_main"], "minHeight": "100vh"})


def stat_block(label, value, color=None):
    color = color or COLORS["blue"]
    return html.Div([
        html.Div(label, style={"color": COLORS["muted"], "fontSize": "9px", "fontFamily": "monospace"}),
        html.Div(value, style={"color": color, "fontSize": "13px", "fontFamily": "monospace", "fontWeight": "500"}),
    ], style={
        "background": COLORS["bg_main"],
        "border": f"1px solid {COLORS['border']}",
        "padding": "6px 10px",
        "borderRadius": "2px",
        "flex": "1",
    })


def info_card(title, value, subtitle="", color=None):
    color = color or COLORS["blue"]
    return html.Div([
        html.Div(title, style={"color": COLORS["muted"], "fontSize": "10px", "fontFamily": "monospace", "marginBottom": "4px"}),
        html.Div(value, style={"color": color, "fontSize": "22px", "fontFamily": "monospace", "fontWeight": "500"}),
        html.Div(subtitle, style={"color": COLORS["muted"], "fontSize": "10px", "fontFamily": "monospace", "marginTop": "4px"}),
    ], style={
        "background": COLORS["bg_panel"],
        "border": f"1px solid {COLORS['border']}",
        "borderLeft": f"3px solid {color}",
        "padding": "16px 20px",
        "borderRadius": "2px",
        "flex": "1",
    })


@app.callback(
    Output("date-range", "start_date"),
    Output("date-range", "end_date"),
    Input("btn-1w", "n_clicks"),
    Input("btn-1m", "n_clicks"),
    Input("btn-3m", "n_clicks"),
    Input("btn-6m", "n_clicks"),
    Input("btn-1y", "n_clicks"),
    Input("btn-all", "n_clicks"),
    prevent_initial_call=True,
)
def quick_select(w, m, m3, m6, y, all_):
    from dash import ctx
    today = pd.Timestamp.now().normalize()
    mapping = {
        "btn-1w": today - pd.Timedelta(weeks=1),
        "btn-1m": today - pd.DateOffset(months=1),
        "btn-3m": today - pd.DateOffset(months=3),
        "btn-6m": today - pd.DateOffset(months=6),
        "btn-1y": today - pd.DateOffset(years=1),
        "btn-all": pd.Timestamp("2020-01-01"),
    }
    start = mapping.get(ctx.triggered_id, today - pd.DateOffset(months=1))
    return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")


@app.callback(
    Output("stat-bar", "children"),
    Output("page-content", "children"),
    Input("tab-candle", "n_clicks"),
    Input("tab-indicator", "n_clicks"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
)
def render_page(tab_c, tab_i, start_date, end_date):
    from dash import ctx
    if not start_date or not end_date:
        return [], html.Div()

    s, e = start_date[:10], end_date[:10]
    active = ctx.triggered_id

    # последняя запись
    last_info = get_last_info()

    if active == "tab-indicator":
        df = get_indicator_data(s, e)
        if df.empty:
            return [], html.Div("Нет данных", style={"color": COLORS["muted"], "padding": "24px", "fontFamily": "monospace"})

        last = df.iloc[-1]
        rsi_color = COLORS["red"] if last["rsi_14"] >= 70 else COLORS["green"] if last["rsi_14"] <= 30 else COLORS["blue"]

        stats = [
            stat_block("CLOSE", f"{last['close']:.2f}"),
            stat_block("SMA5", f"{last['sma_5']:.2f}", COLORS["amber"]),
            stat_block("SMA20", f"{last['sma_20']:.2f}", COLORS["purple"]),
            stat_block("BB UP", f"{last['bb_upper']:.2f}", COLORS["blue"]),
            stat_block("BB LOW", f"{last['bb_lower']:.2f}", COLORS["blue"]),
            stat_block("RSI14", f"{last['rsi_14']:.1f}", rsi_color),
            stat_block("MACD", f"{last['macd']:.4f}", COLORS["green"] if last["macd"] > 0 else COLORS["red"]),
            stat_block("ATR14", f"{last['atr_14']:.2f}", COLORS["muted"]),
        ]

        # SMA + BB chart
        fig_sma = go.Figure()
        fig_sma.add_trace(go.Scatter(x=df["time"], y=df["close"], name="Close", line=dict(color=COLORS["white"], width=1)))
        fig_sma.add_trace(go.Scatter(x=df["time"], y=df["sma_5"], name="SMA5", line=dict(color=COLORS["amber"], width=1, dash="dot")))
        fig_sma.add_trace(go.Scatter(x=df["time"], y=df["sma_10"], name="SMA10", line=dict(color=COLORS["green"], width=1, dash="dot")))
        fig_sma.add_trace(go.Scatter(x=df["time"], y=df["sma_20"], name="SMA20", line=dict(color=COLORS["purple"], width=1, dash="dot")))
        fig_sma.add_trace(go.Scatter(x=df["time"], y=df["bb_upper"], name="BB Upper", line=dict(color=COLORS["blue"], width=1, dash="dash"), fill=None))
        fig_sma.add_trace(go.Scatter(x=df["time"], y=df["bb_lower"], name="BB Lower", line=dict(color=COLORS["blue"], width=1, dash="dash"), fill="tonexty", fillcolor="rgba(79,195,247,0.05)"))
        fig_sma.update_layout(
            paper_bgcolor=COLORS["bg_main"], plot_bgcolor=COLORS["bg_panel"],
            font=dict(color=COLORS["muted"], family="monospace", size=10),
            legend=dict(bgcolor=COLORS["bg_panel"], bordercolor=COLORS["border"], borderwidth=1, font=dict(size=9)),
            xaxis=dict(gridcolor=COLORS["border"], linecolor=COLORS["border"]),
            yaxis=dict(gridcolor=COLORS["border"], linecolor=COLORS["border"], side="right"),
            margin=dict(l=8, r=60, t=8, b=8), height=300,
            title=dict(text="Цена · SMA · Bollinger Bands", font=dict(color=COLORS["muted"], size=11), x=0.01),
        )

        # RSI chart
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df["time"], y=df["rsi_14"], name="RSI14", line=dict(color=COLORS["blue"], width=1.5)))
        fig_rsi.add_trace(go.Scatter(x=df["time"], y=df["rsi_7"], name="RSI7", line=dict(color=COLORS["amber"], width=1, dash="dot")))
        fig_rsi.add_hline(y=70, line_color=COLORS["red"], line_width=0.8, line_dash="dash", annotation_text="70", annotation_font_color=COLORS["red"])
        fig_rsi.add_hline(y=30, line_color=COLORS["green"], line_width=0.8, line_dash="dash", annotation_text="30", annotation_font_color=COLORS["green"])
        fig_rsi.add_hrect(y0=70, y1=100, fillcolor=COLORS["red"], opacity=0.05, line_width=0)
        fig_rsi.add_hrect(y0=0, y1=30, fillcolor=COLORS["green"], opacity=0.05, line_width=0)
        fig_rsi.update_layout(
            paper_bgcolor=COLORS["bg_main"], plot_bgcolor=COLORS["bg_panel"],
            font=dict(color=COLORS["muted"], family="monospace", size=10),
            legend=dict(bgcolor=COLORS["bg_panel"], bordercolor=COLORS["border"], borderwidth=1, font=dict(size=9)),
            xaxis=dict(gridcolor=COLORS["border"], linecolor=COLORS["border"]),
            yaxis=dict(gridcolor=COLORS["border"], linecolor=COLORS["border"], side="right", range=[0, 100]),
            margin=dict(l=8, r=60, t=8, b=8), height=200,
            title=dict(text="RSI", font=dict(color=COLORS["muted"], size=11), x=0.01),
        )

        # MACD chart
        macd_colors = [COLORS["green"] if v >= 0 else COLORS["red"] for v in df["macd_hist"]]
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Bar(x=df["time"], y=df["macd_hist"], name="MACD Hist", marker_color=macd_colors, opacity=0.7))
        fig_macd.add_trace(go.Scatter(x=df["time"], y=df["macd"], name="MACD", line=dict(color=COLORS["blue"], width=1.5)))
        fig_macd.add_trace(go.Scatter(x=df["time"], y=df["macd_signal"], name="Signal", line=dict(color=COLORS["amber"], width=1, dash="dot")))
        fig_macd.add_hline(y=0, line_color=COLORS["muted"], line_width=0.5)
        fig_macd.update_layout(
            paper_bgcolor=COLORS["bg_main"], plot_bgcolor=COLORS["bg_panel"],
            font=dict(color=COLORS["muted"], family="monospace", size=10),
            legend=dict(bgcolor=COLORS["bg_panel"], bordercolor=COLORS["border"], borderwidth=1, font=dict(size=9)),
            xaxis=dict(gridcolor=COLORS["border"], linecolor=COLORS["border"]),
            yaxis=dict(gridcolor=COLORS["border"], linecolor=COLORS["border"], side="right"),
            margin=dict(l=8, r=60, t=8, b=8), height=200,
            title=dict(text="MACD (12, 26, 9)", font=dict(color=COLORS["muted"], size=11), x=0.01),
        )

        # Info cards
        last_time = last_info[0].strftime("%d.%m.%Y %H:%M") if last_info else "—"
        last_price = f"{last_info[1]:.2f} ₽" if last_info else "—"
        last_rsi = f"{last_info[2]:.1f}" if last_info else "—"
        last_zone = last_info[4] if last_info else "—"
        last_atr = f"{last_info[5]:.2f}" if last_info else "—"
        rsi_c = COLORS["red"] if last_info and last_info[2] >= 70 else COLORS["green"] if last_info and last_info[2] <= 30 else COLORS["blue"]

        cards = html.Div([
            info_card("ПОСЛЕДНИЙ ТАЙМФРЕМ", last_time, "последняя свеча в данных", COLORS["blue"]),
            info_card("АКТУАЛЬНАЯ ЦЕНА", last_price, "close_price последней свечи", COLORS["green"]),
            info_card("RSI-14", last_rsi, f"зона: {last_zone}", rsi_c),
            info_card("ATR-14", last_atr, "средний истинный диапазон", COLORS["amber"]),
        ], style={"display": "flex", "gap": "8px", "padding": "12px 24px"})

        content = html.Div([
            cards,
            html.Div([dcc.Graph(figure=fig_sma, config={"displayModeBar": False})], style={"padding": "0 8px"}),
            html.Div([dcc.Graph(figure=fig_rsi, config={"displayModeBar": False})], style={"padding": "0 8px"}),
            html.Div([dcc.Graph(figure=fig_macd, config={"displayModeBar": False})], style={"padding": "0 8px"}),
        ])
        return stats, content

    else:
        df = get_candle_data(s, e)
        if df.empty:
            return [], html.Div("Нет данных", style={"color": COLORS["muted"], "padding": "24px", "fontFamily": "monospace"})

        last = df.iloc[-1]
        first = df.iloc[0]
        change = ((last["close"] - first["open"]) / first["open"]) * 100
        change_color = COLORS["green"] if change >= 0 else COLORS["red"]
        change_sign = "▲" if change >= 0 else "▼"
        vol_b = last["volume"] / 1e9

        stats = [
            stat_block("OPEN", f"{last['open']:.2f}"),
            stat_block("HIGH", f"{df['high'].max():.2f}", COLORS["green"]),
            stat_block("LOW", f"{df['low'].min():.2f}", COLORS["red"]),
            stat_block("CLOSE", f"{last['close']:.2f}", COLORS["blue"]),
            stat_block("VOLUME", f"{vol_b:.2f}B"),
            stat_block("CHANGE", f"{change_sign} {abs(change):.2f}%", change_color),
        ]

        candle = go.Figure()
        candle.add_trace(go.Candlestick(
            x=df["begin"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            increasing_line_color=COLORS["green"], increasing_fillcolor=COLORS["green"],
            decreasing_line_color=COLORS["red"], decreasing_fillcolor=COLORS["red"],
            line_width=1,
        ))
        candle.update_layout(
            paper_bgcolor=COLORS["bg_main"], plot_bgcolor=COLORS["bg_panel"],
            font=dict(color=COLORS["muted"], family="monospace", size=10),
            xaxis=dict(range=[df["begin"].min(), df["begin"].max()], gridcolor=COLORS["border"], linecolor=COLORS["border"]),
            yaxis=dict(range=[df["low"].min() * 0.995, df["high"].max() * 1.005], gridcolor=COLORS["border"], linecolor=COLORS["border"], side="right"),
            xaxis_rangeslider_visible=False,
            margin=dict(l=8, r=60, t=8, b=8),
            showlegend=False,
        )

        vol_colors = [COLORS["green"] if c >= o else COLORS["red"] for c, o in zip(df["close"], df["open"])]
        volume = go.Figure()
        volume.add_trace(go.Bar(x=df["begin"], y=df["volume"], marker_color=vol_colors, marker_opacity=0.7))
        volume.update_layout(
            paper_bgcolor=COLORS["bg_main"], plot_bgcolor=COLORS["bg_panel"],
            font=dict(color=COLORS["muted"], family="monospace", size=10),
            xaxis=dict(range=[df["begin"].min(), df["begin"].max()], gridcolor=COLORS["border"], linecolor=COLORS["border"], showgrid=False),
            yaxis=dict(gridcolor=COLORS["border"], linecolor=COLORS["border"], side="right"),
            margin=dict(l=8, r=60, t=4, b=8), showlegend=False,
        )

        content = html.Div([
            html.Div([dcc.Graph(id="candle-chart", figure=candle, style={"height": "500px"}, config={"displayModeBar": False})], style={"padding": "0 8px"}),
            html.Div([dcc.Graph(id="volume-chart", figure=volume, style={"height": "150px"}, config={"displayModeBar": False})], style={"padding": "0 8px"}),
        ])
        return stats, content


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)