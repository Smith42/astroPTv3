"""KdTreeCrossmatch extension that also surfaces unmatched right-catalog rows.

lsdb's crossmatch only supports ``how in {"left", "inner"}`` at the row
level (``AbstractCrossmatchAlgorithm.crossmatch``'s own docstring is
explicit about this): a Legacy image with no DESI spectrum within
``radius_arcsec`` is silently dropped by ``_create_crossmatch_df``, even
though the full row -- image included -- was already read into memory to
run the KDTree match against. Profiling (2026-09-02 loader-throughput
investigation) found this discards roughly 60% of total wire bytes: ~90%
of DESI rows have no Legacy match, and Legacy's full "image" column (the
big payload) is fetched for every candidate before the distance filter
runs. This subclass recovers those rows as image-only records (NA
spectrum/DESI columns) for zero extra network bytes.

Deliberately narrow scope: this only recovers Legacy rows within pixel
pairs the crossmatch ALREADY visits (i.e. near some DESI pointing). A
Legacy partition with no DESI coverage nearby is never fetched by this
join at all, regardless of this class -- that's the majority of Legacy's
sky footprint, and recovering it needs a genuinely separate stream (full
wire cost, out of scope here).

Built on lsdb internals (``_create_crossmatch_df``, ``_na_series_for_dtype``,
``apply_suffixes``) because the public API has no extension point for this;
re-verify against lsdb's crossmatch test suite behavior if lsdb is upgraded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hats.pixel_math.spatial_index import healpix_to_spatial_index

import nested_pandas as npd
from lsdb.core.crossmatch.abstract_crossmatch_algorithm import _na_series_for_dtype
from lsdb.core.crossmatch.crossmatch_args import CrossmatchArgs
from lsdb.core.crossmatch.kdtree_match import KdTreeCrossmatch
from lsdb.dask.merge_catalog_functions import apply_suffixes


class OuterKdTreeCrossmatch(KdTreeCrossmatch):
    """KdTreeCrossmatch that also emits right rows unmatched within the
    pixel pairs it visits, instead of silently dropping them.

    Pass ``how="left"`` to ``Catalog.crossmatch`` as usual -- pixel
    alignment is unaffected; this only changes row-level assembly within
    the pixel pairs "left" alignment already selects.
    """

    def crossmatch(
        self,
        crossmatch_args: CrossmatchArgs,
        how: str,
        suffixes: tuple[str, str],
        suffix_method: str = "all_columns",
    ) -> npd.NestedFrame:
        if how != "left":
            # only "left" is meaningfully extended here (that's our one use
            # case); anything else falls back to stock lsdb behaviour.
            return super().crossmatch(crossmatch_args, how, suffixes, suffix_method)

        left_join_result = super().crossmatch(crossmatch_args, "left", suffixes, suffix_method)

        right_df = crossmatch_args.right_df
        if (
            right_df is None
            or len(right_df) == 0
            or crossmatch_args.right_pixel is None
            or crossmatch_args.right_order is None
        ):
            return left_join_result

        # Re-run the KDTree match to get right-side indices (super().crossmatch
        # above already ran it once internally but doesn't expose r_inds; this
        # duplicate pass is cheap relative to the network fetch that dominates
        # this pipeline -- see the loader-throughput investigation).
        l_inds, r_inds, extra_cols = self.perform_crossmatch(crossmatch_args)
        if not len(l_inds) == len(r_inds) == len(extra_cols):
            raise ValueError(
                "Crossmatch algorithm must return left and right indices and extra columns with same length"
            )

        # right_df includes the right catalog's margin cache (candidates from
        # a NEIGHBOURING partition, included only to catch boundary-crossing
        # matches). Restrict to rows genuinely native to this pixel via the
        # spatial index, or a margin row would be double-emitted here AND
        # again when its own home partition is processed.
        healpix_order = crossmatch_args.right_catalog_info.healpix_order
        lower = healpix_to_spatial_index(
            crossmatch_args.right_order, crossmatch_args.right_pixel, spatial_index_order=healpix_order
        )
        upper = healpix_to_spatial_index(
            crossmatch_args.right_order,
            crossmatch_args.right_pixel + 1,
            spatial_index_order=healpix_order,
        )
        native_mask = ((right_df.index >= lower) & (right_df.index < upper)).to_numpy()

        matched_mask = np.zeros(len(right_df), dtype=bool)
        matched_mask[r_inds] = True

        unmatched_mask = native_mask & ~matched_mask
        if not unmatched_mask.any():
            return left_join_result

        left_df = crossmatch_args.left_df
        _, suffixed_right_df = apply_suffixes(left_df, right_df, suffixes, suffix_method, log_changes=False)
        right_unmatched = suffixed_right_df.iloc[unmatched_mask].reset_index(drop=True)
        n = len(right_unmatched)
        right_col_names = set(suffixed_right_df.columns)
        na_left = pd.DataFrame(
            {
                col: _na_series_for_dtype(left_join_result[col].dtype, n)
                for col in left_join_result.columns
                if col not in right_col_names
            }
        )
        right_unmatched_out = pd.concat([na_left.reset_index(drop=True), right_unmatched], axis=1)
        right_unmatched_out = right_unmatched_out[left_join_result.columns]
        # No left row to inherit an index from -- use the right row's own
        # (unique, genuine) spatial index. Downstream decode never reads the
        # pandas index (object identity comes from the object_id column), so
        # this mixing left- and right-index semantics in one frame is safe
        # for this pipeline but not something to rely on more generally.
        right_unmatched_out.index = right_df.index[unmatched_mask]

        combined = pd.concat([left_join_result, right_unmatched_out], axis=0)
        return npd.NestedFrame(combined)
