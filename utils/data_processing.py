from __future__ import annotations

import io
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

COLUMN_ALIASES: dict[str, list[str]] = {
    "date": ["date", "data", "order_date", "created_at"],
    "revenue": ["revenue", "valor", "total", "total_amount", "amount"],
    "product": ["product", "produto", "item", "sku"],
    "category": ["category", "categoria"],
    "channel": ["channel", "canal", "source"],
    "uf": ["state", "uf", "region", "regiao"],
    "order_id": ["order_id", "pedido", "id_pedido"],
    "customer_id": ["customer_id", "cliente_id", "id_cliente"],
    "quantity": ["qty", "quantidade", "quantity"],
}

VALID_UFS = {
    "AC",
    "AL",
    "AP",
    "AM",
    "BA",
    "CE",
    "DF",
    "ES",
    "GO",
    "MA",
    "MT",
    "MS",
    "MG",
    "PA",
    "PB",
    "PR",
    "PE",
    "PI",
    "RJ",
    "RN",
    "RS",
    "RO",
    "RR",
    "SC",
    "SP",
    "SE",
    "TO",
}

STATE_NAME_TO_UF = {
    "acre": "AC",
    "alagoas": "AL",
    "amapa": "AP",
    "amazonas": "AM",
    "bahia": "BA",
    "ceara": "CE",
    "distrito_federal": "DF",
    "espirito_santo": "ES",
    "goias": "GO",
    "maranhao": "MA",
    "mato_grosso": "MT",
    "mato_grosso_do_sul": "MS",
    "minas_gerais": "MG",
    "para": "PA",
    "paraiba": "PB",
    "parana": "PR",
    "pernambuco": "PE",
    "piaui": "PI",
    "rio_de_janeiro": "RJ",
    "rio_grande_do_norte": "RN",
    "rio_grande_do_sul": "RS",
    "rondonia": "RO",
    "roraima": "RR",
    "santa_catarina": "SC",
    "sao_paulo": "SP",
    "sergipe": "SE",
    "tocantins": "TO",
}


def _normalize_column_name(name: Any) -> str:
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^0-9A-Za-z]+", "_", text.strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _deduplicate_column_names(columns: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    deduplicated: list[str] = []
    for column in columns:
        base = column or "col"
        count = seen.get(base, 0)
        if count == 0:
            deduplicated.append(base)
        else:
            deduplicated.append(f"{base}_{count + 1}")
        seen[base] = count + 1
    return deduplicated


def read_uploaded_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Read CSV or XLSX from uploaded file bytes."""
    extension = Path(filename).suffix.lower()
    buffer = io.BytesIO(file_bytes)

    if extension == ".csv":
        last_error: Exception | None = None
        for encoding in ("utf-8-sig", "utf-8", "latin1"):
            buffer.seek(0)
            try:
                return pd.read_csv(buffer, sep=None, engine="python", encoding=encoding)
            except Exception as exc:  # pragma: no cover - defensive fallback
                last_error = exc
        raise ValueError(f"Nao foi possivel ler o CSV enviado: {last_error}") from last_error

    if extension in {".xlsx", ".xls"}:
        buffer.seek(0)
        try:
            return pd.read_excel(buffer)
        except Exception as exc:
            raise ValueError(f"Nao foi possivel ler o XLSX enviado: {exc}") from exc

    raise ValueError("Formato nao suportado. Envie um arquivo CSV ou XLSX.")


def normalize_and_map_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Normalize column names and map aliases to canonical schema."""
    if df.empty:
        raise ValueError("O arquivo enviado nao possui dados.")

    normalized_columns = [_normalize_column_name(column) for column in df.columns]
    deduplicated_columns = _deduplicate_column_names(normalized_columns)
    renamed_df = df.copy()
    renamed_df.columns = deduplicated_columns

    mapping: dict[str, str] = {}
    used_source_columns: set[str] = set()

    for canonical, aliases in COLUMN_ALIASES.items():
        candidates = [_normalize_column_name(alias) for alias in [canonical, *aliases]]
        for candidate in candidates:
            if candidate in renamed_df.columns and candidate not in used_source_columns:
                mapping[canonical] = candidate
                used_source_columns.add(candidate)
                break

    rename_map = {source: canonical for canonical, source in mapping.items() if source != canonical}
    standardized_df = renamed_df.rename(columns=rename_map)
    return standardized_df, mapping


def _parse_numeric_value(value: Any) -> float:
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()
    if not text:
        return np.nan

    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[Rr]\$", "", text)
    text = re.sub(r"[^\d,.\-]", "", text)
    if text in {"", "-", ".", ",", "-.", "-,"}:
        return np.nan

    has_comma = "," in text
    has_dot = "." in text

    if has_comma and has_dot:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif has_comma:
        if re.fullmatch(r"-?\d{1,3}(,\d{3})+", text):
            text = text.replace(",", "")
        else:
            text = text.replace(".", "").replace(",", ".")
    elif text.count(".") > 1 and re.fullmatch(r"-?\d{1,3}(\.\d{3})+", text):
        text = text.replace(".", "")

    try:
        return float(text)
    except ValueError:
        return np.nan


def _normalize_uf_value(value: Any) -> Any:
    if pd.isna(value):
        return np.nan

    normalized = _normalize_column_name(value)
    if len(normalized) == 2 and normalized.upper() in VALID_UFS:
        return normalized.upper()
    if normalized in STATE_NAME_TO_UF:
        return STATE_NAME_TO_UF[normalized]

    fallback = str(value).strip().upper()
    return fallback if fallback else np.nan


def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply type conversions and remove invalid or duplicated rows."""
    cleaned = df.copy()

    missing_required = [column for column in ("date", "revenue") if column not in cleaned.columns]
    if missing_required:
        missing_list = ", ".join(missing_required)
        raise ValueError(f"Nao foi possivel identificar colunas obrigatorias: {missing_list}.")

    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce", dayfirst=True)
    cleaned["revenue"] = cleaned["revenue"].apply(_parse_numeric_value)

    if "quantity" in cleaned.columns:
        cleaned["quantity"] = cleaned["quantity"].apply(_parse_numeric_value)

    for id_column in ("order_id", "customer_id"):
        if id_column in cleaned.columns:
            cleaned[id_column] = cleaned[id_column].astype(str).str.strip()
            cleaned.loc[cleaned[id_column].isin(["", "nan", "None"]), id_column] = np.nan

    for text_column in ("product", "category", "channel"):
        if text_column in cleaned.columns:
            cleaned[text_column] = cleaned[text_column].astype(str).str.strip()
            cleaned.loc[cleaned[text_column].isin(["", "nan", "None"]), text_column] = np.nan

    if "uf" in cleaned.columns:
        cleaned["uf"] = cleaned["uf"].apply(_normalize_uf_value)

    cleaned = cleaned.dropna(subset=["date", "revenue"])
    cleaned = cleaned.drop_duplicates()
    cleaned = cleaned.sort_values("date").reset_index(drop=True)
    return cleaned


def apply_filters(df: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    """Apply date and dimension filters to a sales dataframe."""
    filtered = df.copy()

    start_date = filters.get("start_date")
    end_date = filters.get("end_date")

    if start_date is not None:
        filtered = filtered[filtered["date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        end_bound = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        filtered = filtered[filtered["date"] < end_bound]

    for column in ("product", "category", "channel", "uf"):
        selected_values = filters.get(column)
        if selected_values and column in filtered.columns:
            selected_values_str = {str(value) for value in selected_values}
            filtered = filtered[filtered[column].astype(str).isin(selected_values_str)]

    return filtered.copy()
