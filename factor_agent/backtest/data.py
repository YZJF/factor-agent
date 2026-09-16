"""Public A-share data ingestion and deterministic Qlib binary export."""

from __future__ import annotations

import hashlib
import json
import os
import time
from calendar import monthrange
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Protocol

from pydantic import BaseModel, Field


SUPPORTED_UNIVERSES = {"CSI300": "hs300", "CSI500": "zz500"}
RAW_FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,"
    "pctChg,tradestatus,isST"
)


class MarketDataManifest(BaseModel):
    source: str = "baostock"
    created_at: str = Field(default_factory=lambda: datetime.now().astimezone().isoformat())
    start: str
    end: str
    warmup_start: str
    universes: list[str]
    instruments: int
    rows: int
    membership_snapshots: dict[str, int]
    fields: list[str]
    files: dict[str, str]
    content_hashes: dict[str, str]
    qlib_uri: str = ""
    notes: list[str] = Field(default_factory=list)


class PublicMarketDataSource(Protocol):
    """Canonical boundary for future AkShare or licensed data adapters."""

    name: str

    def open(self) -> None: ...

    def close(self) -> None: ...

    def memberships(self, start: str, end: str, universes: Iterable[str]): ...

    def industries(self, start: str, end: str): ...

    def prices(
        self,
        instruments: Iterable[str],
        start: str,
        end: str,
        *,
        cache_dir: Path | None = None,
    ): ...


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - dependency guidance
        raise RuntimeError('market-data support requires: pip install -e ".[backtest]"') from exc
    return pd


def _require_baostock():
    try:
        import baostock as bs
    except ImportError as exc:  # pragma: no cover - dependency guidance
        raise RuntimeError('public download requires: pip install -e ".[backtest]"') from exc
    return bs


def _frame(result: Any):
    pd = _require_pandas()
    if result.error_code != "0":
        raise RuntimeError(f"BaoStock error {result.error_code}: {result.error_msg}")
    rows: list[list[str]] = []
    while result.next():
        rows.append(result.get_row_data())
    return pd.DataFrame(rows, columns=result.fields)


def _instrument(code: str) -> str:
    market, number = code.lower().split(".", 1)
    return f"{market}{number}"


def _baostock_code(instrument: str) -> str:
    value = instrument.lower()
    return f"{value[:2]}.{value[2:]}"


def _month_ends(start: str, end: str) -> list[str]:
    first = date.fromisoformat(start)
    last = date.fromisoformat(end)
    year, month = first.year, first.month
    values: list[str] = []
    while (year, month) <= (last.year, last.month):
        values.append(date(year, month, monthrange(year, month)[1]).isoformat())
        month = month % 12 + 1
        year += 1 if month == 1 else 0
    return values


def _index_adjustment_dates(start: str, end: str) -> list[str]:
    """CSI300/500 regular reviews are semiannual; include both coverage edges."""

    first = date.fromisoformat(start)
    last = date.fromisoformat(end)
    values = {first.isoformat(), last.isoformat()}
    for year in range(first.year, last.year + 1):
        for month, day in ((6, 30), (12, 31)):
            current = date(year, month, day)
            if first <= current <= last:
                values.add(current.isoformat())
    return sorted(values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_csv(frame: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _query_memberships(bs: Any, start: str, end: str, universes: Iterable[str]):
    pd = _require_pandas()
    frames = []
    counts: dict[str, int] = {}
    for universe in universes:
        query_name = SUPPORTED_UNIVERSES[universe]
        query = getattr(bs, f"query_{query_name}_stocks")
        snapshots = []
        for requested_date in _index_adjustment_dates(start, end):
            print(f"BaoStock {universe} membership: {requested_date}", flush=True)
            result = _frame(query(date=requested_date))
            if result.empty:
                continue
            date_column = "date" if "date" in result else "updateDate"
            snapshot = result[["code", date_column]].copy()
            snapshot.columns = ["code", "date"]
            snapshot["date"] = snapshot["date"].replace("", requested_date)
            snapshot["universe"] = universe
            snapshots.append(snapshot)
        if not snapshots:
            raise RuntimeError(f"no historical constituent snapshots returned for {universe}")
        joined = pd.concat(snapshots, ignore_index=True).drop_duplicates()
        joined["instrument"] = joined["code"].map(_instrument)
        joined = joined[["universe", "date", "instrument"]]
        counts[universe] = int(joined["date"].nunique())
        frames.append(joined)
    return pd.concat(frames, ignore_index=True).sort_values(["universe", "date", "instrument"]), counts


def _query_industries(bs: Any, start: str, end: str):
    pd = _require_pandas()
    frames = []
    for requested_date in _month_ends(start, end)[::12]:
        print(f"BaoStock industry snapshot: {requested_date}", flush=True)
        result = _frame(bs.query_stock_industry(date=requested_date))
        if result.empty:
            continue
        date_column = "updateDate" if "updateDate" in result else "date"
        frame = result[["code", date_column, "industry"]].copy()
        frame.columns = ["code", "date", "industry"]
        frame["date"] = frame["date"].replace("", requested_date)
        frame["instrument"] = frame["code"].map(_instrument)
        frames.append(frame[["date", "instrument", "industry"]])
    if not frames:
        raise RuntimeError("no industry snapshots returned by BaoStock")
    return pd.concat(frames, ignore_index=True).drop_duplicates().sort_values(["instrument", "date"])


def _query_prices(
    bs: Any,
    instruments: Iterable[str],
    start: str,
    end: str,
    *,
    cache_dir: Path | None = None,
):
    pd = _require_pandas()
    selected = sorted(set(instruments))
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        missing = [item for item in selected if not (cache_dir / f"{item}.csv").exists()]
        workers = max(1, int(os.getenv("FACTOR_DATA_WORKERS", "4")))
        if missing and workers > 1:
            chunks = [missing[index : index + 10] for index in range(0, len(missing), 10)]
            completed = len(selected) - len(missing)
            failures: list[str] = []
            with ProcessPoolExecutor(max_workers=min(workers, len(chunks))) as executor:
                futures = [
                    executor.submit(
                        _download_price_batch,
                        chunk,
                        start,
                        end,
                        str(cache_dir),
                    )
                    for chunk in chunks
                ]
                for future in as_completed(futures):
                    batch_count, batch_failures = future.result()
                    completed += batch_count
                    failures.extend(batch_failures)
                    print(f"BaoStock prices cached: {completed}/{len(selected)}", flush=True)
            if failures:
                raise RuntimeError(
                    f"BaoStock failed for {len(failures)} instruments; first errors: {failures[:5]}"
                )

    frames = []
    for index, instrument in enumerate(selected, start=1):
        cache_path = cache_dir / f"{instrument}.csv" if cache_dir else None
        if cache_path and cache_path.exists():
            result = pd.read_csv(cache_path, dtype={"code": str})
        else:
            result = _frame(
                bs.query_history_k_data_plus(
                    _baostock_code(instrument),
                    RAW_FIELDS,
                    start_date=start,
                    end_date=end,
                    frequency="d",
                    adjustflag="2",
                )
            )
            if cache_path and not result.empty:
                _atomic_csv(result, cache_path)
        if result.empty:
            continue
        result["instrument"] = instrument
        frames.append(result)
        if index == 1 or index % 25 == 0 or index == len(selected):
            print(f"BaoStock prices: {index}/{len(selected)}", flush=True)
    if not frames:
        raise RuntimeError("no daily prices returned by BaoStock")
    prices = pd.concat(frames, ignore_index=True)
    numeric = ["open", "high", "low", "close", "preclose", "volume", "amount", "pctChg"]
    for column in numeric:
        prices[column] = pd.to_numeric(prices[column], errors="coerce")
    prices["return"] = prices["pctChg"] / 100.0
    prices["tradestatus"] = pd.to_numeric(prices["tradestatus"], errors="coerce").fillna(0).astype(int)
    prices["isST"] = pd.to_numeric(prices["isST"], errors="coerce").fillna(0).astype(int)
    return prices[
        [
            "date",
            "instrument",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "return",
            "tradestatus",
            "isST",
        ]
    ].drop_duplicates(["date", "instrument"])


def _download_price_batch(
    instruments: list[str],
    start: str,
    end: str,
    cache_dir: str,
) -> tuple[int, list[str]]:
    """Worker process: one independent BaoStock session per small batch."""

    bs = _require_baostock()

    def connect() -> bool:
        for attempt in range(4):
            login = bs.login()
            if login.error_code == "0":
                return True
            time.sleep(2**attempt)
        return False

    if not connect():
        return 0, [f"{instrument}: login failed" for instrument in instruments]
    completed = 0
    failures: list[str] = []
    try:
        for instrument in instruments:
            last_error = ""
            for attempt in range(3):
                try:
                    result = _frame(
                        bs.query_history_k_data_plus(
                            _baostock_code(instrument),
                            RAW_FIELDS,
                            start_date=start,
                            end_date=end,
                            frequency="d",
                            adjustflag="2",
                        )
                    )
                    if result.empty:
                        last_error = "empty"
                    else:
                        _atomic_csv(result, Path(cache_dir) / f"{instrument}.csv")
                        completed += 1
                        last_error = ""
                        break
                except Exception as exc:
                    last_error = str(exc)
                try:
                    bs.logout()
                except Exception:
                    pass
                time.sleep(2**attempt)
                connect()
            if last_error:
                failures.append(f"{instrument}: {last_error}")
    finally:
        bs.logout()
    return completed, failures


class BaoStockDataSource:
    name = "baostock"

    def __init__(self):
        self.bs = _require_baostock()

    def open(self) -> None:
        login = self.bs.login()
        if login.error_code != "0":
            raise RuntimeError(f"BaoStock login failed: {login.error_msg}")

    def close(self) -> None:
        self.bs.logout()

    def memberships(self, start: str, end: str, universes: Iterable[str]):
        return _query_memberships(self.bs, start, end, universes)

    def industries(self, start: str, end: str):
        return _query_industries(self.bs, start, end)

    def prices(
        self,
        instruments: Iterable[str],
        start: str,
        end: str,
        *,
        cache_dir: Path | None = None,
    ):
        return _query_prices(self.bs, instruments, start, end, cache_dir=cache_dir)


def _attach_industry(prices: Any, industries: Any):
    pd = _require_pandas()
    left = prices.drop(columns=["industry"], errors="ignore").copy()
    right = industries.copy()
    left["date"] = pd.to_datetime(left["date"])
    right["date"] = pd.to_datetime(right["date"])
    merged = pd.merge_asof(
        left.sort_values(["date", "instrument"]),
        right.sort_values(["date", "instrument"]),
        on="date",
        by="instrument",
        direction="backward",
    )
    merged["industry"] = merged["industry"].fillna("UNKNOWN")
    categories = {name: index for index, name in enumerate(sorted(merged["industry"].unique()), start=1)}
    merged["industry"] = merged["industry"].map(categories).astype(float)
    merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
    return merged


def _cached_snapshots(output: Path, start: str, end: str, universes: list[str]):
    pd = _require_pandas()
    membership_path = output / "memberships.csv"
    industry_path = output / "industries.csv"
    if not membership_path.exists() or not industry_path.exists():
        return None
    memberships = pd.read_csv(membership_path)
    industries = pd.read_csv(industry_path)
    required_membership = {"universe", "date", "instrument"}
    required_industry = {"date", "instrument", "industry"}
    if not required_membership <= set(memberships) or not required_industry <= set(industries):
        return None
    if set(universes) - set(memberships["universe"].unique()):
        return None
    membership_start = pd.Timestamp(memberships["date"].min())
    membership_end = pd.Timestamp(memberships["date"].max())
    industry_start = pd.Timestamp(industries["date"].min())
    industry_end = pd.Timestamp(industries["date"].max())
    if membership_start > pd.Timestamp(start) or membership_end < pd.Timestamp(end) - pd.Timedelta(days=10):
        return None
    if industry_start > pd.Timestamp(start) + pd.Timedelta(days=31):
        return None
    if industry_end.year < pd.Timestamp(end).year:
        return None
    counts = {
        universe: int(memberships.loc[memberships["universe"] == universe, "date"].nunique())
        for universe in universes
    }
    print("BaoStock snapshots: using local checkpoint", flush=True)
    return memberships, industries, counts


def download_public_data(
    output_dir: str | Path,
    *,
    start: str = "2017-01-01",
    end: str = "2024-12-31",
    universes: Iterable[str] = ("CSI300", "CSI500"),
    source: PublicMarketDataSource | None = None,
) -> MarketDataManifest:
    """Download canonical CSV files. Network access occurs only in this command."""

    selected = [item.upper() for item in universes]
    unsupported = sorted(set(selected) - set(SUPPORTED_UNIVERSES))
    if unsupported:
        raise ValueError(f"unsupported public universes: {unsupported}")
    output = Path(output_dir)
    data_source = source or BaoStockDataSource()
    data_source.open()
    try:
        membership_path = output / "memberships.csv"
        industry_path = output / "industries.csv"
        cached = _cached_snapshots(output, start, end, selected)
        if cached:
            memberships, industries, snapshot_counts = cached
        else:
            memberships, snapshot_counts = data_source.memberships(start, end, selected)
            industries = data_source.industries(start, end)
            _atomic_csv(memberships, membership_path)
            _atomic_csv(industries, industry_path)
        prices = data_source.prices(
            memberships["instrument"].unique(),
            start,
            end,
            cache_dir=output / "prices",
        )
        market = _attach_industry(prices, industries)
    finally:
        data_source.close()

    market_path = output / "market.csv"
    _atomic_csv(market, market_path)
    files = {
        "market": str(market_path.resolve()),
        "memberships": str(membership_path.resolve()),
        "industries": str(industry_path.resolve()),
    }
    hashes = {name: _sha256(Path(path)) for name, path in files.items()}
    manifest = MarketDataManifest(
        source=data_source.name,
        start=start,
        end=end,
        warmup_start=start,
        universes=selected,
        instruments=int(market["instrument"].nunique()),
        rows=len(market),
        membership_snapshots=snapshot_counts,
        fields=[column for column in market.columns if column not in {"date", "instrument"}],
        files=files,
        content_hashes=hashes,
        notes=["adjustflag=2", "industry values are as-of annual BaoStock snapshots"],
    )
    manifest_path = output / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest


def _membership_ranges(memberships: Any, calendar: list[str], universe: str):
    pd = _require_pandas()
    source = memberships[memberships["universe"] == universe].copy()
    source["date"] = pd.to_datetime(source["date"])
    snapshots = sorted(source["date"].unique())
    rows: list[tuple[str, str, str]] = []
    for index, snapshot_date in enumerate(snapshots):
        start = max(pd.Timestamp(calendar[0]), pd.Timestamp(snapshot_date))
        next_date = (
            pd.Timestamp(snapshots[index + 1])
            if index + 1 < len(snapshots)
            else pd.Timestamp(calendar[-1]) + timedelta(days=1)
        )
        valid_calendar = [pd.Timestamp(item) for item in calendar if start <= pd.Timestamp(item) < next_date]
        if not valid_calendar:
            continue
        end = valid_calendar[-1]
        instruments = source[source["date"] == snapshot_date]["instrument"].unique()
        rows.extend((item, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")) for item in instruments)
    return rows


def export_qlib_dataset(raw_dir: str | Path, qlib_dir: str | Path) -> MarketDataManifest:
    """Export canonical CSVs into Qlib's local binary provider format."""

    pd = _require_pandas()
    import numpy as np

    raw = Path(raw_dir)
    target = Path(qlib_dir)
    manifest = MarketDataManifest.model_validate_json((raw / "manifest.json").read_text(encoding="utf-8"))
    market = pd.read_csv(manifest.files["market"])
    memberships = pd.read_csv(manifest.files["memberships"])
    if market.duplicated(["date", "instrument"]).any():
        raise ValueError("duplicate date/instrument rows in market data")
    calendar = sorted(market.loc[market["tradestatus"] == 1, "date"].astype(str).unique())
    if not calendar:
        raise ValueError("market data has no tradable calendar")
    calendar_index = {value: index for index, value in enumerate(calendar)}
    (target / "calendars").mkdir(parents=True, exist_ok=True)
    (target / "instruments").mkdir(parents=True, exist_ok=True)
    (target / "features").mkdir(parents=True, exist_ok=True)
    (target / "calendars" / "day.txt").write_text("\n".join(calendar) + "\n", encoding="utf-8")

    feature_columns = ["open", "high", "low", "close", "volume", "amount", "return", "industry", "tradestatus", "isST"]
    for instrument, frame in market.groupby("instrument"):
        frame = frame.sort_values("date")
        instrument_dir = target / "features" / str(instrument).lower()
        instrument_dir.mkdir(parents=True, exist_ok=True)
        start_index = calendar_index[str(frame.iloc[0]["date"])]
        end_index = calendar_index[str(frame.iloc[-1]["date"])]
        indexed = frame.set_index(frame["date"].astype(str)).reindex(calendar[start_index : end_index + 1])
        for field in feature_columns:
            values = pd.to_numeric(indexed[field], errors="coerce").to_numpy(dtype="<f4")
            payload = np.concatenate((np.asarray([start_index], dtype="<f4"), values))
            payload.tofile(instrument_dir / f"{field}.day.bin")

    all_rows = []
    for instrument, frame in market.groupby("instrument"):
        all_rows.append((instrument, str(frame["date"].min()), str(frame["date"].max())))
    for universe in manifest.universes:
        rows = _membership_ranges(memberships, calendar, universe)
        text = "\n".join(f"{instrument}\t{start}\t{end}" for instrument, start, end in rows) + "\n"
        (target / "instruments" / f"{universe.lower()}.txt").write_text(text, encoding="utf-8")
    (target / "instruments" / "all.txt").write_text(
        "\n".join(f"{instrument}\t{start}\t{end}" for instrument, start, end in all_rows) + "\n",
        encoding="utf-8",
    )
    manifest.qlib_uri = str(target.resolve())
    manifest_path = raw / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest


def validate_manifest(path: str | Path, *, required_start: str, required_end: str) -> MarketDataManifest:
    manifest_path = Path(path)
    manifest = MarketDataManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if manifest.start > required_start or manifest.end < required_end:
        raise ValueError(
            f"data coverage {manifest.start}..{manifest.end} does not cover "
            f"{required_start}..{required_end}"
        )
    for name, value in manifest.files.items():
        file_path = Path(value)
        if not file_path.exists():
            raise FileNotFoundError(f"manifest file missing: {file_path}")
        if _sha256(file_path) != manifest.content_hashes[name]:
            raise ValueError(f"content hash mismatch: {name}")
    if not manifest.qlib_uri or not Path(manifest.qlib_uri).exists():
        raise FileNotFoundError("Qlib dataset has not been exported")
    return manifest
