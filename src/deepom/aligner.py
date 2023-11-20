from enum import IntEnum
from typing import Optional

import numpy
import numpy as np
from devtools import debug
from numba import njit
from numpy import ndarray
from utils.pyutils import PydanticClass, PydanticClassConfig, PydanticClassInputs

from utils.pyutils import NDArray


class Orientation(IntEnum):
    FORWARD = 0
    REVERSE = 1


class DeepOMAligner:
    class Config(PydanticClassConfig):
        loc_factor: float = 500
        skip_r_factor: float = 10
        skip_q_factor: float = 10
        dp_band_size: int = 5
        limit_ref_len: Optional[int] = None

    class Query(PydanticClass):
        query_positions: NDArray
        query_scale: float

    class QueryToRef(Query):
        reference_id: str = None
        orientation: Orientation

    class Alignment(QueryToRef):
        score: float
        oriented_query: NDArray
        reference_indices: NDArray
        query_indices: NDArray

    def __init__(self, **config):
        self.config = self.Config(**config)
        debug(self.config)

    def align(self, query: Query, references: dict[str, np.ndarray]) -> Alignment:
        return max((
            self.align_to_ref(
                query=self.QueryToRef(
                    **query.model_dump(),
                    orientation=orientation,
                    reference_id=reference_id
                ),
                reference=reference[:self.config.limit_ref_len],
            )
            for reference_id, reference in references.items()
            if len(reference) >= 2
            for orientation in [Orientation.FORWARD, Orientation.REVERSE]
        ), key=lambda x: x.score)

    def align_to_ref(self, query: QueryToRef, reference: np.ndarray) -> Alignment:
        if query.orientation == Orientation.FORWARD:
            oriented_query = query.query_positions
        elif query.orientation == Orientation.REVERSE:
            oriented_query = -query.query_positions[::-1]
        else:
            assert False

        query_scaled = oriented_query * query.query_scale

        # send query and reference to the alignment algorithm, and get back the alignment score matrix,
        #   and matrix of previous indices
        score_matrix, prev_matrix = self.compute_score_matrix(
            qvec=query_scaled,
            rvec=reference,
            loc_factor=self.config.loc_factor,
            skip_r_factor=self.config.skip_r_factor,
            skip_q_factor=self.config.skip_q_factor,
            dp_band_size=self.config.dp_band_size,
        )

        # find best score, and the alignment path's endpoint in the alignment matrix
        r0, q0 = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
        score = score_matrix[r0, q0]

        # traverse alignment path
        path = []
        while r0 >= 0 and q0 >= 0:
            path.append((r0, q0))
            r0, q0 = prev_matrix[r0, q0]

        path = numpy.stack(path)[::-1]

        # extract indices from the path
        reference_indices, query_indices = path.T

        if query.orientation == Orientation.REVERSE:
            query_indices = len(oriented_query) - query_indices - 1

        return self.Alignment(
            **query.model_dump(),
            score=score,
            oriented_query=oriented_query,
            reference_indices=reference_indices,
            query_indices=query_indices,
        )

    @staticmethod
    @njit(nogil=True)
    def compute_score_matrix(
            qvec: ndarray,
            rvec: ndarray,
            loc_factor: float,
            skip_r_factor: float,
            skip_q_factor: float,
            dp_band_size: int
    ):
        assert 2 <= len(rvec)
        assert 2 <= len(qvec)

        len_r = len(rvec)
        len_q = len(qvec)

        S = numpy.full((len_r, len_q), numpy.nan)
        P = numpy.full((len_r, len_q, 2), -1)
        S[0, :] = 0
        S[:, 0] = 0

        for r in range(1, len_r):
            for q in range(1, len_q):
                best_score = -numpy.inf
                best_prev_node = None

                for q0 in range(max(0, q - dp_band_size), q):
                    delta_query = qvec[q] - qvec[q0]
                    skip_query = q - q0 - 1

                    for r0 in range(max(0, r - dp_band_size), r):
                        delta_ref = rvec[r] - rvec[r0]
                        skip_ref = r - r0 - 1

                        delta_cost = abs(delta_ref - delta_query)
                        edge_cost = (
                                delta_cost / loc_factor +
                                skip_ref / skip_r_factor +
                                skip_query / skip_q_factor
                        )
                        node_score = 1
                        cand_score = S[r0, q0] + node_score - edge_cost

                        if cand_score > best_score:
                            best_score = cand_score
                            best_prev_node = r0, q0

                S[r, q] = best_score
                P[r, q, :] = best_prev_node

        return S, P
