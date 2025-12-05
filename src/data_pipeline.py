#!/usr/bin/env python3
"""
Data Pipeline Module
Handles 13F loading, CRSP queries, feature generation, and caching
"""

import pandas as pd
import numpy as np
import wrds
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class DataPipeline:
    """Complete data pipeline for institutional ownership strategy"""

    def __init__(self, config: Dict):
        self.config = config
        self.wrds_username = config["data"]["wrds_username"]
        self.parquet_13f = Path(config["data"]["paths"]["13f_parquet"])
        self.cache_dir = Path(config["data"]["paths"]["cache"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.start_date = config["data"]["start_date"]
        self.end_date = config["data"]["end_date"]
        logger.info(f"DataPipeline initialized: {self.start_date} to {self.end_date}")

    def load_13f_all_quarters(self) -> pd.DataFrame:
        """Load all 13F data from partitioned parquet"""
        logger.info("Loading 13F data from partitioned parquet...")
        quarter_folders = sorted(self.parquet_13f.glob("yq=*"))
        if not quarter_folders:
            raise FileNotFoundError(f"No 13F data found in {self.parquet_13f}")
        dfs = []
        for qtr_folder in quarter_folders:
            qtr_file = qtr_folder / "data.parquet"
            if qtr_file.exists():
                dfs.append(pd.read_parquet(qtr_file))
        h13 = pd.concat(dfs, ignore_index=True)
        h13["cusip"] = h13["cusip"].str.upper().str.strip()
        h13["ncusip"] = h13["cusip"].str[:8]
        h13["report_q_end"] = pd.to_datetime(h13["report_q_end"])
        h13["filing_date"] = pd.to_datetime(h13["filing_date"])
        h13["date_q_end"] = h13["report_q_end"].dt.to_period("Q").dt.end_time
        logger.info(f"Loaded {len(h13):,} rows, {h13['date_q_end'].nunique()} quarters")
        return h13

    def download_new_13f_quarter(self, quarter_end: str) -> bool:
        """Download new quarter from WRDS"""
        qtr_date = pd.to_datetime(quarter_end)
        qtr_str = qtr_date.strftime("%Y-Q%q")
        logger.info(f"Downloading {qtr_str}...")
        try:
            db = wrds.Connection(wrds_username=self.wrds_username)
            query = f"""
            SELECT fdate AS filing_date, rdate AS report_q_end, mgrno AS cik,
                   mgrname AS manager_name, stkname AS issuer, cusip, shares,
                   (shares * prc) AS value_usd
            FROM tr_13f.s34
            WHERE rdate = '{qtr_date.strftime("%Y-%m-%d")}'
              AND shares > 0 AND prc > 0
            """
            df = db.raw_sql(query, date_cols=["filing_date", "report_q_end"])
            db.close()
            if len(df) == 0:
                logger.warning(f"No data for {qtr_str}")
                return False
            df["cusip"] = df["cusip"].str.upper().str.strip()
            qtr_folder = self.parquet_13f / f"yq={qtr_str}"
            qtr_folder.mkdir(parents=True, exist_ok=True)
            df.to_parquet(qtr_folder / "data.parquet", index=False)
            logger.info(f"✓ Downloaded {len(df):,} rows")
            return True
        except Exception as e:
            logger.error(f"Failed: {e}")
            return False

    def get_crsp_data(
        self, permnos: List[int], use_cache: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get CRSP quarterly data"""
        cache_file = self.cache_dir / "crsp_monthly.parquet"
        if use_cache and cache_file.exists():
            logger.info("Loading CRSP from cache...")
            cached = pd.read_parquet(cache_file)
            cached = cached[cached["permno"].isin(permnos)]
            px_q = cached[["permno", "date_q_end", "prc", "ret_q"]].drop_duplicates()
            shares_q = cached[["permno", "date_q_end", "shares_out"]].drop_duplicates()
            logger.info(f"Loaded {len(px_q):,} observations")
            return px_q, shares_q
        logger.info(f"Querying CRSP for {len(permnos):,} PERMNOs...")
        db = wrds.Connection(wrds_username=self.wrds_username)
        chunk_size = 1000
        msf_chunks = []
        for i in range(0, len(permnos), chunk_size):
            chunk_permnos = permnos[i : i + chunk_size]
            query = f"""
            SELECT permno, date, prc, ret, shrout
            FROM crsp.msf
            WHERE permno IN ({','.join(map(str, chunk_permnos))})
              AND date >= '{self.start_date}' AND date <= '{self.end_date}'
            """
            msf_chunks.append(db.raw_sql(query, date_cols=["date"]))
            if (i // chunk_size + 1) % 10 == 0:
                logger.info(f"  Progress: {i+chunk_size}/{len(permnos)}")
        db.close()
        msf = pd.concat(msf_chunks, ignore_index=True)
        msf["date_q_end"] = msf["date"].dt.to_period("Q").dt.end_time
        px_q = (
            msf.sort_values(["permno", "date"])
            .groupby(["permno", "date_q_end"], as_index=False)
            .agg({"prc": "last", "ret": lambda x: (1 + x).prod() - 1})
            .rename(columns={"ret": "ret_q"})
        )
        shares_q = (
            msf.sort_values(["permno", "date"])
            .groupby(["permno", "date_q_end"], as_index=False)
            .agg({"shrout": "last"})
            .rename(columns={"shrout": "shares_out"})
        )
        shares_q["shares_out"] *= 1000
        cache_data = px_q.merge(shares_q, on=["permno", "date_q_end"], how="outer")
        cache_data.to_parquet(cache_file)
        return px_q, shares_q

    def create_cusip_permno_mapping(
        self, h13: pd.DataFrame, use_cache: bool = True
    ) -> pd.DataFrame:
        """Create point-in-time CUSIP to PERMNO mapping"""
        cache_file = self.cache_dir / "cusip_permno_map.parquet"
        if use_cache and cache_file.exists():
            logger.info("Loading mapping from cache...")
            return pd.read_parquet(cache_file)
        logger.info("Creating CUSIP -> PERMNO mapping...")
        ncusips = sorted(h13["ncusip"].unique())
        quarters = sorted(h13["date_q_end"].unique())
        db = wrds.Connection(wrds_username=self.wrds_username)
        chunk_size = 5000
        msenames_chunks = []
        for i in range(0, len(ncusips), chunk_size):
            chunk_cusips = ncusips[i : i + chunk_size]
            query = f"""
            SELECT permno, ncusip, namedt, nameendt
            FROM crsp.msenames
            WHERE ncusip IN ({','.join(f"'{c}'" for c in chunk_cusips)})
              AND namedt <= '{self.end_date}' AND nameendt >= '{self.start_date}'
            """
            msenames_chunks.append(db.raw_sql(query, date_cols=["namedt", "nameendt"]))
        db.close()
        msenames = pd.concat(msenames_chunks, ignore_index=True)
        map_13f = []
        for q in quarters:
            hq = h13[h13["date_q_end"] == q][
                ["cik", "ncusip", "cusip"]
            ].drop_duplicates()
            valid = msenames[(msenames["namedt"] <= q) & (msenames["nameendt"] >= q)][
                ["permno", "ncusip"]
            ].drop_duplicates()
            mapped = hq.merge(valid, on="ncusip", how="inner")
            mapped["date_q_end"] = q
            map_13f.append(mapped)
        map_13f = pd.concat(map_13f, ignore_index=True)
        map_13f.to_parquet(cache_file)
        logger.info(f"Created {len(map_13f):,} mappings")
        return map_13f

    def compute_holdings_with_ownership(
        self, h13, map_13f, px_q, shares_q
    ) -> pd.DataFrame:
        """Compute holdings with ownership metrics"""
        logger.info("Computing holdings...")
        hold_q = (
            h13.merge(map_13f, on=["cik", "cusip", "date_q_end"], how="inner")
            .merge(shares_q, on=["permno", "date_q_end"], how="left")
            .merge(px_q, on=["permno", "date_q_end"], how="left")
        )
        hold_q["w_mgr"] = hold_q.groupby(["cik", "date_q_end"])["value_usd"].transform(
            lambda s: s / s.sum()
        )
        hold_q["shares_out"] = hold_q["shares_out"].replace(0, np.nan)
        hold_q["io_pct"] = (
            hold_q.groupby(["permno", "date_q_end"])["shares"].transform("sum")
            / hold_q["shares_out"]
        )
        total_inst = hold_q.groupby(["permno", "date_q_end"])["shares"].transform("sum")
        hold_q["own_frac_mgr"] = hold_q["shares"] / total_inst
        return hold_q

    def compute_manager_skill(self, hold_q, px_q) -> pd.DataFrame:
        """Compute 8Q trailing manager skill"""
        logger.info("Computing manager skill...")
        px_next = px_q.sort_values(["permno", "date_q_end"]).assign(
            ret_next=lambda d: d.groupby("permno")["ret_q"].shift(-1)
        )[["permno", "date_q_end", "ret_next"]]
        panel = hold_q.merge(
            px_next, on=["permno", "date_q_end"], how="left"
        ).sort_values(["permno", "date_q_end"])
        mgr_perf = (
            panel.groupby(["cik", "date_q_end"], as_index=False)
            .apply(
                lambda g: pd.Series({"mgr_ret": (g["w_mgr"] * g["ret_next"]).sum()}),
                include_groups=False,
            )
            .reset_index()
        )
        mgr_perf = mgr_perf.sort_values(["cik", "date_q_end"])
        mgr_perf["skill_8q"] = (
            mgr_perf.groupby("cik")["mgr_ret"]
            .rolling(8, min_periods=3)
            .mean()
            .reset_index(level=0, drop=True)
        )
        panel = panel.merge(
            mgr_perf[["cik", "date_q_end", "skill_8q"]],
            on=["cik", "date_q_end"],
            how="left",
        )
        return panel

    def compute_skill_weighted_features(self, panel) -> pd.DataFrame:
        """Compute skill-weighted ownership features"""
        logger.info("Computing skill-weighted features...")
        panel_copy = panel.copy()
        panel_copy["skill_quintile"] = panel_copy.groupby("date_q_end")[
            "skill_8q"
        ].transform(lambda x: pd.qcut(x, 5, labels=False, duplicates="drop"))
        panel_copy["skill_posw"] = 0.0
        panel_copy.loc[panel_copy["skill_quintile"] >= 3, "skill_posw"] = 1.0
        panel_copy = panel_copy.sort_values(["cik", "permno", "date_q_end"])
        panel_copy["own_flow_mgr"] = panel_copy.groupby(["cik", "permno"])[
            "own_frac_mgr"
        ].diff()
        skilled = panel_copy[panel_copy["skill_posw"] > 0].copy()
        skilled["w_own_contrib"] = skilled["skill_posw"] * skilled["own_frac_mgr"]
        skilled["w_flow_contrib"] = skilled["skill_posw"] * skilled[
            "own_flow_mgr"
        ].fillna(0)
        features = (
            skilled.groupby(["permno", "date_q_end"], as_index=False)
            .agg(
                {
                    "w_own_contrib": "sum",
                    "w_flow_contrib": "sum",
                    "skill_posw": "sum",
                    "cik": "count",
                }
            )
            .rename(
                columns={
                    "w_own_contrib": "sk_own",
                    "w_flow_contrib": "sk_flow",
                    "skill_posw": "n_skilled_mgrs",
                    "cik": "n_total_mgrs",
                }
            )
        )
        skilled["contrib_sq"] = skilled["w_own_contrib"] ** 2
        concentration = (
            skilled.groupby(["permno", "date_q_end"], as_index=False)
            .agg({"contrib_sq": "sum"})
            .rename(columns={"contrib_sq": "sk_hhi"})
        )
        features = features.merge(
            concentration, on=["permno", "date_q_end"], how="left"
        )
        return features

    def build_model_dataframe(
        self, use_cache: bool = True, force_refresh: bool = False
    ) -> pd.DataFrame:
        """Build complete model_df"""
        cache_file = self.cache_dir / "model_df.parquet"
        if use_cache and cache_file.exists() and not force_refresh:
            logger.info("Loading model_df from cache...")
            return pd.read_parquet(cache_file)
        logger.info("=" * 80)
        logger.info("BUILDING COMPLETE MODEL DATAFRAME")
        h13 = self.load_13f_all_quarters()
        map_13f = self.create_cusip_permno_mapping(
            h13, use_cache=use_cache and not force_refresh
        )
        permnos = sorted(map_13f["permno"].unique().tolist())
        px_q, shares_q = self.get_crsp_data(
            permnos, use_cache=use_cache and not force_refresh
        )
        hold_q = self.compute_holdings_with_ownership(h13, map_13f, px_q, shares_q)
        panel = self.compute_manager_skill(hold_q, px_q)
        features = self.compute_skill_weighted_features(panel)
        stock_level = panel[
            ["permno", "date_q_end", "io_pct", "ret_next"]
        ].drop_duplicates()
        stock_level = stock_level.sort_values(["permno", "date_q_end"])
        stock_level["io_flow"] = stock_level.groupby("permno")["io_pct"].diff()
        model_df = features.merge(stock_level, on=["permno", "date_q_end"], how="left")
        model_df = model_df.merge(
            px_q[["permno", "date_q_end", "prc", "ret_q"]],
            on=["permno", "date_q_end"],
            how="left",
        )
        model_df = model_df.merge(
            shares_q[["permno", "date_q_end", "shares_out"]],
            on=["permno", "date_q_end"],
            how="left",
        )
        model_df["mkt_cap"] = model_df["prc"].abs() * model_df["shares_out"]
        model_df = model_df.rename(columns={"ret_q": "mom_1"})
        model_df = model_df.dropna(subset=["ret_next"]).sort_values(
            ["date_q_end", "permno"]
        )
        model_df.to_parquet(cache_file)
        logger.info(f"MODEL DATAFRAME COMPLETE: {len(model_df):,} rows")
        logger.info("=" * 80)
        return model_df
