"""Fit the one production sparse, time-bounded WALS base for every run profile.

DuckDB aggregates legal raw lines once into full-confidence CSR interactions.
Columns use purchases in the frozen vocabulary window; item/user factors are
fit only in the base interval. Bounded Arrow/mmap assembly avoids all-event
Python lists. Resource checks precede ALS, alpha=1 prevents double scaling, and
separate learned index/factor files are written beneath the supplied run root.
"""

from pathlib import Path

import numpy as np

from hm_recsys.core.contracts import (
    BASE_END,
    START,
    VOCAB_START,
    write_json,
)
from hm_recsys.core.progress import emit
from hm_recsys.data.history import History
from hm_recsys.data.streaming import export_query
from hm_recsys.retrieval.vectors import BlockIndex
from hm_recsys.runtime.capacity import check


def fit_wals(run: Path, profile: dict) -> None:
    """Fit cohort confidence CSR on represented columns of the global vocabulary."""
    from implicit.cpu.als import AlternatingLeastSquares
    from scipy.sparse import csr_matrix

    history = History(run)
    con = history.con
    try:
        con.execute(
            (
                """CREATE TEMP TABLE vocabulary AS SELECT DISTINCT article_id
                FROM transactions WHERE t_dat>=? AND t_dat<?"""
            ),
            [VOCAB_START, BASE_END],
        )
        con.execute(
            (
                """CREATE TEMP TABLE observed AS SELECT customer_id,article_id,
                1+20*ln(1+sum(exp(-date_diff('day',t_dat,DATE
                '2020-08-11')/30.0))) AS confidence
                FROM transactions JOIN cohort USING(customer_id) JOIN
                vocabulary
                USING(article_id)
                WHERE t_dat>=? AND t_dat<? GROUP BY customer_id,article_id"""
            ),
            [START, BASE_END],
        )
        nnz, user_count, item_count = con.execute(
            """SELECT count(*),count(DISTINCT customer_id),count(DISTINCT
            article_id)
            FROM observed"""
        ).fetchall()[0]
        if not nnz:
            raise ValueError("No legal WALS observations")
        estimate = (
            nnz * 40
            + (user_count + item_count) * profile["compute"][("wals_factors")] * 24
            + user_count * 32
        )
        check(
            run,
            "wals",
            memory=estimate,
            disk=estimate * 2,
            details={
                "nnz": nnz,
                ("user_count"): user_count,
                ("items"): item_count,
                ("factors"): profile[("compute")][("wals_factors")],
            },
        )
        global_ids = [
            r[0]
            for r in con.execute(
                "SELECT article_id FROM vocabulary ORDER BY article_id"
            ).fetchall()
        ]
        ids = [
            r[0]
            for r in con.execute(
                """SELECT DISTINCT article_id FROM observed ORDER BY article_id"""
            ).fetchall()
        ]
        export_query(
            con,
            ("""SELECT DISTINCT customer_id FROM observed ORDER BY customer_id"""),
            [],
            run / ("models/wals_users.parquet"),
        )
        users = (
            [
                r[0]
                for r in con.execute(
                    """SELECT DISTINCT customer_id FROM observed ORDER BY
                    customer_id"""
                ).fetchall()
            ]
            if profile[("profile")] == ("pilot")
            else []
        )
        con.execute(
            """CREATE TEMP TABLE item_map AS SELECT article_id,row_number()
            OVER(ORDER
            BY article_id)-1 AS i FROM (SELECT DISTINCT article_id FROM
            observed)"""
        )
        con.execute(
            """CREATE TEMP TABLE user_map AS SELECT customer_id,row_number()
            OVER(ORDER
            BY customer_id)-1 AS u FROM (SELECT DISTINCT customer_id FROM
            observed)"""
        )
        export_query(
            con,
            (
                """SELECT u,i,confidence FROM observed JOIN item_map
                USING(article_id) JOIN
                user_map USING(customer_id) ORDER BY u,i"""
            ),
            [],
            run / ("tmp/wals_csr.parquet"),
        )
        import pyarrow.parquet as pq

        values = np.lib.format.open_memmap(
            run / "tmp/wals_values.npy", mode="w+", dtype=np.float32, shape=(nnz,)
        )
        indices = np.lib.format.open_memmap(
            run / "tmp/wals_indices.npy", mode="w+", dtype=np.int64, shape=(nnz,)
        )
        pointers = np.zeros(user_count + 1, dtype=np.int64)
        offset = 0
        for batch in pq.ParquetFile(run / "tmp/wals_csr.parquet").iter_batches(8192):
            n = len(batch)
            values[offset : offset + n] = batch["confidence"].to_numpy()
            indices[offset : offset + n] = batch["i"].to_numpy()
            unique, counts = np.unique(batch["u"].to_numpy(), return_counts=True)
            pointers[unique + 1] += counts
            offset += n
            emit("wals_csr", offset, nnz, "nonzero_edges")
        np.cumsum(pointers, out=pointers)
        if offset != nnz or pointers[-1] != nnz:
            raise ValueError("CSR row alignment failed")
        sparse = csr_matrix(
            (values, indices, pointers), shape=(user_count, len(ids)), copy=False
        )
        c = profile["compute"]
        model = AlternatingLeastSquares(
            factors=c["wals_factors"],
            regularization=0.05,
            alpha=1.0,
            iterations=c[("wals_iterations")],
            num_threads=2,
            random_state=42,
            use_cg=True,
        )
        emit(
            "wals_matrix",
            sparse.nnz,
            sparse.nnz,
            "nonzero_edges",
            status="complete",
            metrics={
                ("users"): user_count,
                ("items"): len(ids),
                ("latent_dimensions"): model.factors,
            },
            force=True,
        )

        def report_iteration(iteration, seconds, loss):
            emit(
                "wals_iterations",
                iteration + 1,
                c["wals_iterations"],
                "iterations",
                metrics={("iteration_seconds"): seconds, ("loss"): loss},
                force=True,
            )

        model.fit(sparse, show_progress=False, callback=report_iteration)
        emit(
            "wals_iterations",
            c["wals_iterations"],
            c["wals_iterations"],
            "iterations",
            status=("complete"),
            force=True,
        )
        BlockIndex(ids, np.asarray(model.item_factors, dtype=np.float32)).save(
            run / ("indexes"), ("wals")
        )
        np.save(
            run / "models/wals_user_factors.npy", model.user_factors, allow_pickle=False
        )
        write_json(
            run / "models/wals.json",
            {
                "users": users,
                "user_count": user_count,
                ("users_file"): ("models/wals_users.parquet"),
                ("items"): ids,
                ("global_vocabulary"): global_ids,
                ("unsupported_cohort_items"): sorted(set(global_ids) - set(ids)),
                ("normalization"): ("none"),
                ("iterations"): c[("wals_iterations")],
                ("factors"): c[("wals_factors")],
                ("nnz"): nnz,
                ("library_alpha"): 1.0,
                ("confidence_alpha"): 20,
                ("regularization"): 0.05,
                ("fit_start"): str(START),
                ("fit_end_exclusive"): str(BASE_END),
                ("vocabulary_start"): str(VOCAB_START),
                ("coverage"): (
                    "only columns with legal cohort observations have trained factors"
                ),
            },
        )
    finally:
        history.close()
