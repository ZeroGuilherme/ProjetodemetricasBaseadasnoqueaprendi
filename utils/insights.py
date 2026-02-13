from __future__ import annotations

import pandas as pd

DAY_NAMES_PT = {
    0: "Segunda-feira",
    1: "Terca-feira",
    2: "Quarta-feira",
    3: "Quinta-feira",
    4: "Sexta-feira",
    5: "Sabado",
    6: "Domingo",
}


def _format_currency_br(value: float) -> str:
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def build_insights(df: pd.DataFrame, previous_df: pd.DataFrame) -> list[str]:
    """Generate lightweight business insights from filtered sales data."""
    if df.empty:
        return ["Sem dados suficientes para gerar insights."]

    insights: list[str] = []

    if "product" in df.columns and df["product"].notna().any():
        product_ranking = (
            df.dropna(subset=["product"])
            .groupby("product", as_index=False)["revenue"]
            .sum()
            .sort_values("revenue", ascending=False)
        )
        if not product_ranking.empty:
            champion = product_ranking.iloc[0]
            insights.append(
                f"Produto campeao: {champion['product']} com {_format_currency_br(float(champion['revenue']))}."
            )

    weekday_revenue = df.groupby(df["date"].dt.weekday)["revenue"].sum()
    if not weekday_revenue.empty:
        best_weekday = int(weekday_revenue.idxmax())
        best_weekday_revenue = float(weekday_revenue.max())
        insights.append(
            f"Melhor dia da semana: {DAY_NAMES_PT[best_weekday]} ({_format_currency_br(best_weekday_revenue)})."
        )

    if "channel" in df.columns and df["channel"].notna().any():
        channel_ranking = (
            df.dropna(subset=["channel"])
            .groupby("channel", as_index=False)["revenue"]
            .sum()
            .sort_values("revenue", ascending=False)
        )
        if not channel_ranking.empty:
            best_channel = channel_ranking.iloc[0]
            insights.append(
                f"Melhor canal: {best_channel['channel']} com {_format_currency_br(float(best_channel['revenue']))}."
            )

    weekly_revenue = df.set_index("date").resample("W-MON")["revenue"].sum()
    if len(weekly_revenue) >= 3:
        weekly_change = weekly_revenue.pct_change().dropna()
        if not weekly_change.empty:
            max_var_date = weekly_change.abs().idxmax()
            max_var_value = float(weekly_change.loc[max_var_date] * 100)
            insights.append(
                f"Maior variacao semanal recente: {max_var_value:+.1f}% na semana encerrada em "
                f"{max_var_date.strftime('%d/%m/%Y')}."
            )
    else:
        monthly_revenue = df.set_index("date").resample("MS")["revenue"].sum()
        monthly_change = monthly_revenue.pct_change().dropna()
        if not monthly_change.empty:
            max_var_date = monthly_change.abs().idxmax()
            max_var_value = float(monthly_change.loc[max_var_date] * 100)
            insights.append(
                f"Maior variacao mensal recente: {max_var_value:+.1f}% no mes de {max_var_date.strftime('%m/%Y')}."
            )

    if previous_df is not None and not previous_df.empty:
        current_revenue = float(df["revenue"].sum())
        previous_revenue = float(previous_df["revenue"].sum())
        if previous_revenue > 0:
            delta_pct = (current_revenue - previous_revenue) / previous_revenue * 100
            if delta_pct <= -15:
                insights.append(
                    f"Alerta: queda de {abs(delta_pct):.1f}% na receita em relacao ao periodo anterior."
                )

    if not insights:
        return ["Nao foi possivel gerar insights suficientes com os dados filtrados."]

    return insights
