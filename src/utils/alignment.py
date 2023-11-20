from typing import Optional

import numpy as np
from scipy.interpolate import interp1d

from deepom.aligner import DeepOMAligner
from utils.pyutils import NDArray


class AlignedImage(DeepOMAligner.Alignment):
    image: NDArray
    subset: Optional[str] = None


class AlignmentInfo(DeepOMAligner.Alignment):
    references: dict

    def __repr__(self):
        return f"{self.reference_id} {self.orientation.name} " \
               f"ref_lims={[*self.ref_lims]} " \
               f"query_lims={[*self.query_lims]} " \
               f"score={self.score:.2f} " \
               f"actual_scale={self.actual_scale} "

    def reference_segment(self, start, stop):
        i, j = self.aligned_reference_segment.searchsorted([start, stop])
        return self.aligned_reference_segment[i: j]

    @property
    def aligned_reference_segment(self):
        i, j = self.reference_indices[[0, -1]]
        return self.reference[i: j]

    @property
    def aligned_reference(self):
        return self.reference[self.reference_indices]

    @property
    def aligned_query(self):
        return self.query_positions[self.query_indices]

    @property
    def ref_lims(self):
        return self.aligned_reference[[0, -1]]

    @property
    def query_lims(self):
        return sorted(self.aligned_query[[0, -1]])

    @property
    def reference(self):
        return self.references[self.reference_id]

    def transform_to_reference(self, query_positions: NDArray | list):
        return interp1d(self.aligned_query, self.aligned_reference, bounds_error=False,
                        fill_value="extrapolate")(np.asarray(query_positions))

    def transform_to_query(self, reference_positions: NDArray | list):
        return interp1d(self.aligned_reference, self.aligned_query, bounds_error=False,
                        fill_value="extrapolate")(np.asarray(reference_positions))

    @property
    def actual_scale(self):
        qry = self.query_lims
        ref = self.ref_lims
        return (ref[-1] - ref[0]) / (qry[-1] - qry[0])
