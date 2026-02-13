from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_processing import apply_filters, clean_sales_data, normalize_and_map_columns, read_uploaded_file
from utils.forecast import forecast_moving_average
from utils.insights import build_insights
from utils.metrics import compute_kpis
from utils.sample_data import generate_sample_data

st.set_page_config(page_title="Dashboard de Vendas", page_icon=":bar_chart:", layout="wide")

DAY_ORDER = [
    "Segunda-feira",
    "Terca-feira",
    "Quarta-feira",
    "Quinta-feira",
    "Sexta-feira",
    "Sabado",
    "Domingo",
]

DAY_NAMES_PT = {
    0: "Segunda-feira",
    1: "Terca-feira",
    2: "Quarta-feira",
    3: "Quinta-feira",
    4: "Sexta-feira",
    5: "Sabado",
    6: "Domingo",
}

UF_COORDINATES = {
    "AC": (-8.77, -70.55),
    "AL": (-9.62, -36.82),
    "AP": (1.41, -51.77),
    "AM": (-3.47, -65.10),
    "BA": (-12.96, -38.51),
    "CE": (-3.72, -38.54),
    "DF": (-15.79, -47.88),
    "ES": (-20.31, -40.34),
    "GO": (-16.64, -49.31),
    "MA": (-2.55, -44.30),
    "MT": (-15.60, -56.10),
    "MS": (-20.44, -54.64),
    "MG": (-19.92, -43.94),
    "PA": (-1.46, -48.49),
    "PB": (-7.12, -34.86),
    "PR": (-25.43, -49.27),
    "PE": (-8.05, -34.88),
    "PI": (-5.09, -42.80),
    "RJ": (-22.90, -43.20),
    "RN": (-5.79, -35.21),
    "RS": (-30.03, -51.23),
    "RO": (-8.76, -63.90),
    "RR": (2.82, -60.67),
    "SC": (-27.59, -48.55),
    "SP": (-23.55, -46.63),
    "SE": (-10.91, -37.07),
    "TO": (-10.25, -48.25),
}


def format_currency_br(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"R$ {float(value):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def format_integer(value: int | float | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{int(value):,}".replace(",", ".")


def format_delta(delta_pct: float | None) -> str | None:
    if delta_pct is None or (isinstance(delta_pct, float) and np.isnan(delta_pct)):
        return None
    return f"{delta_pct:+.1f}%"


@st.cache_data(show_spinner=False)
def load_uploaded_and_prepare(file_bytes: bytes, filename: str) -> tuple[pd.DataFrame, dict[str, str]]:
    raw_df = read_uploaded_file(file_bytes, filename)
    mapped_df, mapping = normalize_and_map_columns(raw_df)
    clean_df = clean_sales_data(mapped_df)
    return clean_df, mapping


@st.cache_data(show_spinner=False)
def load_sample_and_prepare() -> tuple[pd.DataFrame, dict[str, str]]:
    sample_df = generate_sample_data()
    mapped_df, mapping = normalize_and_map_columns(sample_df)
    clean_df = clean_sales_data(mapped_df)
    return clean_df, mapping


@st.cache_data(show_spinner=False)
def aggregate_revenue_over_time(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    aggregated = df.set_index("date").resample(freq)["revenue"].sum().reset_index()
    return aggregated


@st.cache_data(show_spinner=False)
def aggregate_dimension_revenue(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame(columns=["dimension", "revenue"])
    grouped = (
        df.dropna(subset=[column])
        .groupby(column, as_index=False)["revenue"]
        .sum()
        .sort_values("revenue", ascending=False)
        .rename(columns={column: "dimension"})
    )
    return grouped


@st.cache_data(show_spinner=False)
def aggregate_channel_orders(df: pd.DataFrame) -> pd.DataFrame:
    if "channel" not in df.columns:
        return pd.DataFrame(columns=["channel", "orders"])

    valid_df = df.dropna(subset=["channel"]).copy()
    if valid_df.empty:
        return pd.DataFrame(columns=["channel", "orders"])

    if "order_id" in valid_df.columns:
        grouped = (
            valid_df.groupby("channel", as_index=False)["order_id"]
            .nunique()
            .rename(columns={"order_id": "orders"})
            .sort_values("orders", ascending=False)
        )
    else:
        grouped = (
            valid_df.groupby("channel", as_index=False)
            .size()
            .rename(columns={"size": "orders"})
            .sort_values("orders", ascending=False)
        )
    return grouped


@st.cache_data(show_spinner=False)
def aggregate_top_products(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if "product" not in df.columns:
        return pd.DataFrame(columns=["product", "revenue"])
    grouped = (
        df.dropna(subset=["product"])
        .groupby("product", as_index=False)["revenue"]
        .sum()
        .sort_values("revenue", ascending=False)
        .head(top_n)
    )
    return grouped


@st.cache_data(show_spinner=False)
def build_heatmap_matrix(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    if df["date"].dt.hour.nunique() <= 1 and int(df["date"].dt.hour.max()) == 0:
        return pd.DataFrame()

    temp = df.copy()
    temp["weekday"] = temp["date"].dt.weekday.map(DAY_NAMES_PT)
    temp["hour"] = temp["date"].dt.hour
    temp["orders"] = 1

    matrix = temp.pivot_table(index="weekday", columns="hour", values="orders", aggfunc="sum", fill_value=0)
    matrix = matrix.reindex(DAY_ORDER).fillna(0)
    return matrix


def apply_global_search(df: pd.DataFrame, query: str) -> pd.DataFrame:
    if not query:
        return df

    mask = pd.Series(False, index=df.index)
    for column in df.columns:
        mask |= df[column].astype(str).str.contains(query, case=False, na=False)
    return df[mask]


def safe_date_range_input(min_date: pd.Timestamp, max_date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    selected = st.sidebar.date_input(
        "Intervalo de datas",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
    )

    if isinstance(selected, tuple):
        start_date, end_date = selected
    else:
        start_date, end_date = selected, selected

    if start_date > end_date:
        start_date, end_date = end_date, start_date

    return pd.Timestamp(start_date), pd.Timestamp(end_date)


def sidebar_multiselect(df: pd.DataFrame, column: str, label: str) -> list[str]:
    if column not in df.columns:
        return []
    options = sorted(df[column].dropna().astype(str).unique().tolist())
    return st.sidebar.multiselect(label, options=options, default=[])


def render_uf_chart(df: pd.DataFrame) -> None:
    st.subheader("Mapa de receita por UF/Regiao")

    uf_revenue = aggregate_dimension_revenue(df, "uf")
    if uf_revenue.empty:
        st.info("Coluna de UF/Regiao nao disponivel para este conjunto filtrado.")
        return

    uf_revenue = uf_revenue.rename(columns={"dimension": "uf"})
    uf_revenue["uf"] = uf_revenue["uf"].astype(str).str.strip().str.upper()
    uf_revenue["lat"] = uf_revenue["uf"].map(lambda uf: UF_COORDINATES.get(uf, (np.nan, np.nan))[0])
    uf_revenue["lon"] = uf_revenue["uf"].map(lambda uf: UF_COORDINATES.get(uf, (np.nan, np.nan))[1])

    geo_df = uf_revenue.dropna(subset=["lat", "lon"])
    if not geo_df.empty:
        fig = px.scatter_geo(
            geo_df,
            lat="lat",
            lon="lon",
            size="revenue",
            color="revenue",
            hover_name="uf",
            hover_data={"revenue": ":.2f", "lat": False, "lon": False},
            color_continuous_scale="Blues",
        )
        fig.update_geos(
            projection_type="natural earth",
            showland=True,
            landcolor="rgb(245,245,245)",
            showcountries=True,
            countrycolor="rgb(200,200,200)",
            lataxis_range=[-35, 6],
            lonaxis_range=[-75, -31],
        )
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
        return

    # Fallback to bars when values are not UF codes.
    fallback_bar = px.bar(
        uf_revenue.sort_values("revenue", ascending=False),
        x="uf",
        y="revenue",
        labels={"uf": "UF/Regiao", "revenue": "Receita"},
        color="revenue",
        color_continuous_scale="Blues",
    )
    fallback_bar.update_layout(height=420)
    st.plotly_chart(fallback_bar, use_container_width=True)


def main() -> None:
    st.title("Dashboard de Vendas")
    st.caption("Upload de dados CSV/XLSX, analise de KPIs e previsao baseline de receita.")

    st.sidebar.header("Entrada de dados")
    uploaded_file = st.sidebar.file_uploader("Envie um arquivo CSV ou XLSX", type=["csv", "xlsx", "xls"])

    try:
        if uploaded_file is not None:
            df, mapping = load_uploaded_and_prepare(uploaded_file.getvalue(), uploaded_file.name)
            st.sidebar.success(f"Arquivo carregado: {uploaded_file.name}")
        else:
            df, mapping = load_sample_and_prepare()
            st.sidebar.info("Nenhum arquivo enviado. Dataset sintetico carregado automaticamente.")
    except Exception as exc:
        st.error(f"Erro ao carregar dados: {exc}")
        st.stop()

    if df.empty:
        st.warning("Nao ha dados disponiveis para exibir.")
        st.stop()

    mapped_columns = ", ".join([f"{canonical} <- {source}" for canonical, source in mapping.items()])
    if mapped_columns:
        st.sidebar.caption(f"Mapeamento de colunas: {mapped_columns}")

    st.sidebar.header("Filtros")
    start_date, end_date = safe_date_range_input(df["date"].min(), df["date"].max())

    selected_products = sidebar_multiselect(df, "product", "Produto")
    selected_categories = sidebar_multiselect(df, "category", "Categoria")
    selected_channels = sidebar_multiselect(df, "channel", "Canal")
    selected_regions = sidebar_multiselect(df, "uf", "Regiao/UF")

    current_filters = {
        "start_date": start_date,
        "end_date": end_date,
        "product": selected_products,
        "category": selected_categories,
        "channel": selected_channels,
        "uf": selected_regions,
    }
    filtered_df = apply_filters(df, current_filters)

    period_days = max((end_date - start_date).days + 1, 1)
    previous_end = start_date - timedelta(days=1)
    previous_start = previous_end - timedelta(days=period_days - 1)

    previous_filters = dict(current_filters)
    previous_filters["start_date"] = previous_start
    previous_filters["end_date"] = previous_end
    previous_df = apply_filters(df, previous_filters)

    download_bytes = filtered_df.to_csv(index=False).encode("utf-8-sig")
    st.sidebar.download_button(
        label="Baixar dados filtrados",
        data=download_bytes,
        file_name="dados_filtrados.csv",
        mime="text/csv",
    )

    if filtered_df.empty:
        st.warning("Nenhum registro encontrado para os filtros selecionados.")
        st.stop()

    kpis = compute_kpis(filtered_df, previous_df)
    col_1, col_2, col_3, col_4 = st.columns(4)
    col_1.metric(
        "Receita total",
        format_currency_br(kpis["revenue"]["value"]),
        format_delta(kpis["revenue"]["delta_pct"]),
    )
    col_2.metric(
        "Numero de pedidos",
        format_integer(kpis["orders"]["value"]),
        format_delta(kpis["orders"]["delta_pct"]),
    )
    col_3.metric(
        "Ticket medio",
        format_currency_br(kpis["ticket"]["value"]),
        format_delta(kpis["ticket"]["delta_pct"]),
    )
    customer_value = format_integer(kpis["customers"]["value"]) if kpis["customers"]["available"] else "N/A"
    customer_delta = format_delta(kpis["customers"]["delta_pct"]) if kpis["customers"]["available"] else None
    col_4.metric("Clientes unicos", customer_value, customer_delta)

    st.subheader("Receita ao longo do tempo")
    granularity = st.selectbox("Granularidade", ["Diario", "Semanal", "Mensal"], index=0)
    freq_map = {"Diario": "D", "Semanal": "W-MON", "Mensal": "MS"}
    time_df = aggregate_revenue_over_time(filtered_df[["date", "revenue"]], freq=freq_map[granularity])
    time_chart = px.line(
        time_df,
        x="date",
        y="revenue",
        markers=True,
        labels={"date": "Data", "revenue": "Receita"},
    )
    time_chart.update_layout(height=360)
    st.plotly_chart(time_chart, use_container_width=True)

    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Top 10 produtos por receita")
        top_products = aggregate_top_products(filtered_df)
        if top_products.empty:
            st.info("Coluna de produto nao disponivel para os dados filtrados.")
        else:
            chart_data = top_products.sort_values("revenue", ascending=True)
            top_chart = px.bar(
                chart_data,
                x="revenue",
                y="product",
                orientation="h",
                labels={"product": "Produto", "revenue": "Receita"},
                color="revenue",
                color_continuous_scale="Tealgrn",
            )
            top_chart.update_layout(height=420)
            st.plotly_chart(top_chart, use_container_width=True)

    with right_col:
        st.subheader("Distribuicao de pedidos por canal")
        channel_data = aggregate_channel_orders(filtered_df)
        if channel_data.empty:
            st.info("Coluna de canal nao disponivel para os dados filtrados.")
        else:
            chart_type = st.radio("Tipo de grafico", ["Pizza", "Barras"], horizontal=True)
            if chart_type == "Pizza":
                channel_chart = px.pie(
                    channel_data,
                    names="channel",
                    values="orders",
                    hole=0.35,
                    labels={"channel": "Canal", "orders": "Pedidos"},
                )
            else:
                channel_chart = px.bar(
                    channel_data.sort_values("orders", ascending=False),
                    x="channel",
                    y="orders",
                    labels={"channel": "Canal", "orders": "Pedidos"},
                    color="orders",
                    color_continuous_scale="Blues",
                )
            channel_chart.update_layout(height=420)
            st.plotly_chart(channel_chart, use_container_width=True)

    map_col, heat_col = st.columns(2)
    with map_col:
        render_uf_chart(filtered_df)

    with heat_col:
        st.subheader("Heatmap: dia da semana x hora")
        heatmap_matrix = build_heatmap_matrix(filtered_df[["date"]])
        if heatmap_matrix.empty:
            st.info("Sem variacao horaria suficiente para montar o heatmap.")
        else:
            heatmap_chart = px.imshow(
                heatmap_matrix,
                labels={"x": "Hora", "y": "Dia da semana", "color": "Pedidos"},
                color_continuous_scale="YlOrRd",
                aspect="auto",
            )
            heatmap_chart.update_layout(height=420)
            st.plotly_chart(heatmap_chart, use_container_width=True)

    st.subheader("Tabela de dados")
    query = st.text_input("Buscar na tabela")
    table_df = apply_global_search(filtered_df, query)
    st.caption(f"Registros exibidos: {len(table_df):,}".replace(",", "."))
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    st.subheader("Insights automaticos")
    insights = build_insights(filtered_df, previous_df)
    for line in insights:
        st.markdown(f"- {line}")

    st.subheader("Previsao de receita - proximos 30 dias (baseline)")
    daily_revenue = filtered_df.set_index("date").resample("D")["revenue"].sum()
    forecast_result = forecast_moving_average(daily_revenue, horizon=30, window=7)
    history = forecast_result["history"]
    forecast = forecast_result["forecast"]

    if isinstance(history, pd.Series) and isinstance(forecast, pd.Series) and not history.empty:
        forecast_chart = go.Figure()
        forecast_chart.add_trace(
            go.Scatter(
                x=history.index,
                y=history.values,
                mode="lines",
                name="Historico",
                line=dict(color="#1f77b4"),
            )
        )
        forecast_chart.add_trace(
            go.Scatter(
                x=forecast.index,
                y=forecast.values,
                mode="lines",
                name="Previsao",
                line=dict(color="#d62728", dash="dash"),
            )
        )
        forecast_chart.update_layout(height=360, xaxis_title="Data", yaxis_title="Receita")
        st.plotly_chart(forecast_chart, use_container_width=True)

        metrics = forecast_result.get("metrics")
        if isinstance(metrics, dict):
            mae_col, mape_col = st.columns(2)
            mae_col.metric("MAE (teste)", format_currency_br(metrics.get("mae")))
            mape_value = metrics.get("mape")
            mape_col.metric("MAPE (teste)", "N/A" if mape_value is None else f"{mape_value:.2f}%")
            st.caption(f"Avaliacao realizada com {metrics.get('test_size')} dias no conjunto de teste.")
        else:
            st.info("Sem dados suficientes para validar erro da previsao. Exibindo apenas baseline projetado.")
    else:
        st.info("Sem dados suficientes para gerar previsao.")


if __name__ == "__main__":
    main()
