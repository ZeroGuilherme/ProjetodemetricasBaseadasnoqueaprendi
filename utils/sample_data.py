from __future__ import annotations

import numpy as np
import pandas as pd

PRODUCT_CATALOG = [
    ("Notebook Pro 14", "Eletronicos", 5200.0),
    ("Notebook Air 13", "Eletronicos", 4100.0),
    ("Smartphone Max", "Eletronicos", 3200.0),
    ("Smartphone Lite", "Eletronicos", 1800.0),
    ("Monitor 27", "Perifericos", 1450.0),
    ("Mouse Sem Fio", "Perifericos", 160.0),
    ("Teclado Mecanico", "Perifericos", 390.0),
    ("Headset Gamer", "Perifericos", 520.0),
    ("Cadeira Ergonomica", "Moveis", 2200.0),
    ("Mesa Home Office", "Moveis", 1400.0),
    ("Impressora Laser", "Escritorio", 980.0),
    ("Webcam Full HD", "Perifericos", 320.0),
]

CHANNELS = ["E-commerce", "Loja Fisica", "Marketplace", "Televendas"]
CHANNEL_PROBABILITIES = [0.42, 0.28, 0.20, 0.10]
CHANNEL_FACTORS = {
    "E-commerce": 0.98,
    "Loja Fisica": 1.00,
    "Marketplace": 0.95,
    "Televendas": 1.03,
}

UF_CODES = [
    "SP",
    "RJ",
    "MG",
    "RS",
    "PR",
    "SC",
    "BA",
    "PE",
    "GO",
    "CE",
    "DF",
    "PA",
    "ES",
]
UF_PROBABILITIES = [0.25, 0.12, 0.11, 0.08, 0.08, 0.05, 0.07, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03]


def generate_sample_data(n_days: int = 365, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic sales data for dashboard fallback."""
    if n_days <= 0:
        raise ValueError("n_days deve ser maior que zero.")

    rng = np.random.default_rng(seed)
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.Timedelta(days=n_days - 1)
    days = pd.date_range(start=start_date, end=end_date, freq="D")

    promo_count = max(4, n_days // 45)
    promo_days = {pd.Timestamp(day).normalize() for day in rng.choice(days, size=promo_count, replace=False)}

    customer_pool = np.array([f"CUST-{index:05d}" for index in range(1, 3501)])
    order_counter = 100000
    rows: list[dict[str, object]] = []

    for current_day in days:
        weekday_factor = [0.95, 1.00, 1.04, 1.10, 1.20, 1.30, 1.06][current_day.weekday()]
        trend_factor = 1 + (0.15 * (current_day - start_date).days / max(n_days - 1, 1))
        seasonality = 1 + 0.08 * np.sin(2 * np.pi * current_day.dayofyear / 365.25)
        promo_factor = 1.28 if current_day.normalize() in promo_days else 1.00

        expected_orders = 24 * weekday_factor * trend_factor * seasonality * promo_factor
        order_count = max(6, int(rng.poisson(expected_orders)))

        for _ in range(order_count):
            product, category, base_price = PRODUCT_CATALOG[int(rng.integers(0, len(PRODUCT_CATALOG)))]
            quantity = int(rng.choice([1, 2, 3, 4], p=[0.64, 0.24, 0.09, 0.03]))
            channel = str(rng.choice(CHANNELS, p=CHANNEL_PROBABILITIES))
            uf = str(rng.choice(UF_CODES, p=UF_PROBABILITIES))
            customer_id = str(rng.choice(customer_pool))

            hour = int(rng.integers(8, 22))
            minute = int(rng.integers(0, 60))
            timestamp = current_day + pd.Timedelta(hours=hour, minutes=minute)

            price_noise = float(rng.normal(loc=1.0, scale=0.10))
            revenue = max(12.0, base_price * quantity * CHANNEL_FACTORS[channel] * price_noise)

            rows.append(
                {
                    "date": timestamp,
                    "revenue": round(float(revenue), 2),
                    "product": product,
                    "category": category,
                    "channel": channel,
                    "uf": uf,
                    "order_id": f"ORD-{order_counter}",
                    "customer_id": customer_id,
                    "quantity": quantity,
                }
            )
            order_counter += 1

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
