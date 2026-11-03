"""Prepare bounded raw cutoff-query inputs for frozen Two-Tower inference.

The cache binds model maps, independent rosters and legal history cutoffs. SQL
extracts model-independent profiles/history once; integer gathers preserve query
order and zero-history masks. Derived mmap files stay beneath the run root;
trainable representations are always recomputed by the current frozen model."""

from datetime import date

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from hm_recsys.core.contracts import digest, read_json
from hm_recsys.core.progress import emit
from hm_recsys.data.streaming import export_query
from hm_recsys.retrieval.two_tower.execution import ItemBank
from hm_recsys.retrieval.two_tower.model import PROFILE_KEYS
from hm_recsys.retrieval.two_tower.prepared import verify
from hm_recsys.runtime.capacity import check
from hm_recsys.runtime.files import atomic_json


class QueryInputs:
    """Persist only model-independent [Q,3], [Q,20,3], [Q,20] input arrays.

    One windowed transaction pass per partition replaces per-batch window scans.
    Full label files are never opened. Reload verifies and reuses the same raw
    inputs, while every learned user vector is computed by the loaded model.
    """

    def __init__(self, encoder, part) -> None:
        history, run = (encoder.history_store, encoder.history_store.run)
        folder = run / "prepared/queries" / part.name
        expected = {
            "version": 1,
            "run": read_json(run / "run_manifest.json")[("migration_signature")],
            ("maps"): encoder.maps,
            ("roster"): digest(run / ("queries") / f"{part.name}.json"),
            ("before"): part.meta[("start")],
            ("ordering"): ("date-item-price-channel-desc/raw20/from-2020-05-19"),
        }
        emit(
            f"query_inputs_{part.name}",
            0,
            part.count,
            unit="queries",
            status=("started"),
            message=("Preparing label-free raw profile/history inputs"),
            force=True,
        )
        if folder.exists():
            self.meta = verify(folder, expected)
            emit(
                f"query_inputs_{part.name}",
                part.count,
                part.count,
                unit="queries",
                status=("restored"),
                message=("Raw partition input cache verified and reused"),
                force=True,
            )
        else:
            check(
                run,
                f"query_inputs_{part.name}",
                memory=64 * 1024**2,
                disk=part.count * 600 * 3,
                details={("queries"): part.count},
            )
            folder.mkdir(parents=True)
            con = history.con
            if part.disk:
                roster_sql = "SELECT customer_id FROM read_parquet(?)"
                params = [str(run / part.meta["file"])]
            else:
                con.register(
                    "encoding_roster", pa.table({"customer_id": part.meta[("users")]})
                )
                roster_sql, params = ("SELECT customer_id FROM encoding_roster", [])
            con.execute(
                (
                    """CREATE OR REPLACE TEMP TABLE encoding_queries AS SELECT *,
                    row_number()
                    OVER(ORDER BY customer_id)-1 AS q FROM ("""
                )
                + roster_sql
                + (")"),
                params,
            )
            export_query(
                con,
                (
                    """SELECT q,c.* FROM encoding_queries JOIN customers c
                    USING(customer_id)
                    ORDER BY q"""
                ),
                [],
                folder / ("profiles.parquet"),
            )
            arrays = {
                "profiles": np.lib.format.open_memmap(
                    folder / "profiles.npy",
                    mode=("w+"),
                    dtype=np.int64,
                    shape=(part.count, 3),
                ),
                ("histories"): np.lib.format.open_memmap(
                    folder / ("histories.npy"),
                    mode=("w+"),
                    dtype=np.int64,
                    shape=(part.count, 20, 3),
                ),
                ("weights"): np.lib.format.open_memmap(
                    folder / ("weights.npy"),
                    mode=("w+"),
                    dtype=np.float32,
                    shape=(part.count, 20),
                ),
            }
            for value in arrays.values():
                value[:] = 0
            profile_count = 0
            for batch in pq.ParquetFile(folder / "profiles.parquet").iter_batches(512):
                records = batch.to_pylist()
                for row in records:
                    user = row["customer_id"]
                    row["age"] = float(row["age"]) if row.get("age") else None
                    encoder.sample.customers = {user: row}
                    arrays["profiles"][row["q"]] = [
                        encoder.maps[k].get(v, 0)
                        for k, v in zip(
                            PROFILE_KEYS, encoder._profile_values(user), strict=True
                        )
                    ]
                profile_count += len(records)
                emit(
                    f"query_profiles_{part.name}",
                    profile_count,
                    part.count,
                    unit=("queries"),
                    message=("Encoded customer profile IDs written"),
                )
            emit(
                f"query_profiles_{part.name}",
                profile_count,
                part.count,
                unit=("queries"),
                status=("completed"),
                message=("Raw profile IDs ready"),
                force=True,
            )
            before = date.fromisoformat(part.meta["start"])
            emit(
                f"query_history_sql_{part.name}",
                0,
                None,
                unit="history_rows",
                status=("started"),
                message=("Selecting latest-20 strictly pre-cutoff raw lines"),
                force=True,
            )
            export_query(
                con,
                (
                    """SELECT q,article_id,t_dat,
                    row_number() OVER(PARTITION BY q ORDER BY t_dat DESC,
                    article_id DESC,price DESC,sales_channel_id DESC)-1 AS
                    slot
                    FROM transactions t JOIN encoding_queries
                    USING(customer_id)
                    WHERE t_dat>=DATE '2020-05-19' AND t_dat<?
                    QUALIFY slot<20 ORDER BY q,slot"""
                ),
                [before],
                folder / ("history.parquet"),
            )
            bank = ItemBank(encoder, torch.device("cpu"))
            selected_history = pq.ParquetFile(folder / "history.parquet")
            history_total = selected_history.metadata.num_rows
            emit(
                f"query_history_sql_{part.name}",
                history_total,
                None,
                unit=("selected_history_rows"),
                status=("completed"),
                message=("Latest-20 selected; scanned transaction rows unknown"),
                force=True,
            )
            history_count = 0
            for batch in selected_history.iter_batches(512):
                ids = batch["article_id"].to_pylist()
                q = batch["q"].to_numpy()
                slot = batch["slot"].to_numpy()
                arrays["histories"][q, slot] = bank.codes[
                    [bank.position[i] for i in ids]
                ][:, [0, 2, 4]]
                days = (
                    (
                        np.datetime64(before)
                        - batch["t_dat"].to_numpy(zero_copy_only=False)
                    )
                    .astype("timedelta64[D]")
                    .astype(np.int64)
                )
                arrays["weights"][q, slot] = np.exp(-(days - 1) / 14)
                history_count += len(batch)
                emit(
                    f"query_history_slots_{part.name}",
                    history_count,
                    history_total,
                    unit=("history_rows"),
                    message=("Past-only raw history slots written"),
                )
            emit(
                f"query_history_slots_{part.name}",
                history_count,
                history_total,
                unit=("history_rows"),
                status=("completed"),
                message=("Selected raw history slots ready"),
                force=True,
            )
            for value in arrays.values():
                value.flush()
            self.meta = {
                "identity": expected,
                "queries": part.count,
                ("transaction_window_passes"): 1,
                ("files"): {
                    p.name: {("bytes"): p.stat().st_size, ("sha256"): digest(p)}
                    for p in folder.iterdir()
                },
            }
            atomic_json(folder / "manifest.json", self.meta)
            emit(
                f"query_inputs_{part.name}",
                part.count,
                part.count,
                unit="queries",
                status=("completed"),
                message=("Label-free partition raw inputs sealed"),
                metrics={("selected_history_rows"): history_total},
                force=True,
            )
        self.arrays = {
            name: np.load(folder / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in (("profiles"), ("histories"), ("weights"))
        }

    def users(self, start: int, count: int) -> tuple:
        """Gather at most 512 queries without history SQL or row encoders."""
        if (
            count < 0
            or count > 512
            or start < 0
            or (start + count > self.meta["queries"])
        ):
            raise ValueError("Invalid query-input slice")
        return tuple(
            torch.from_numpy(np.array(a[start : start + count], copy=True))
            for a in self.arrays.values()
        )
