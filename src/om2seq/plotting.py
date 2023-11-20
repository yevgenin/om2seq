import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from deepom.aligner import Orientation
from deepom.localizer import DeepOMLocalizer
from om2seq.mapping import MappingResult, RefSegment
from om2seq.cropping import Cropper
from om2seq.env import ENV
from utils.alignment import AlignmentInfo, AlignedImage
from utils.pyutils import PydanticClass, PydanticClassConfig, PydanticClassInputs


def event_plot(x, color=None, pos=0, linelength=.5, **kwargs):
    plt.eventplot(x, colors=color, linelengths=linelength, lineoffsets=(pos + 1 + .5) * linelength, **kwargs)


def init_figure():
    plt.figure(figsize=(40, 2))


def set_axis_props():
    # plt.margins(0, 0)
    plt.axis("off")
    # plt.tight_layout()


class ImagePlot(PydanticClass):
    image_width: int = 5

    def plot_image(self, image: np.ndarray, y_extent=(1, 0)):
        w = image.shape[0]
        s = self.image_width
        image = image[w // 2 - s // 2:w // 2 + s // 2 + 1]
        plt.imshow(image, extent=[0, image.shape[1], *y_extent],
                   aspect='auto', cmap='gray', interpolation='none')


class AlignmentPlot(AlignedImage, AlignmentInfo, ImagePlot):
    references: dict
    lim_to_alignment: bool = False

    def plot_alignment(self):
        init_figure()

        if self.image is not None:
            self.plot_image(self.image)

        self._plot_alignment()
        set_axis_props()
        self._plot_title()

    def _plot_alignment(self):
        # event_plot(self.query_positions / ENV.BNX_SCALE, color='r', pos=1)
        ref_to_query = interp1d(self.aligned_reference[[0, -1]], self.aligned_query[[0, -1]], bounds_error=False,
                                fill_value="extrapolate")
        event_plot(ref_to_query(self.aligned_reference_segment) / ENV.BNX_SCALE, color='g', pos=1)
        if self.lim_to_alignment:
            plt.xlim(*self.query_lims)

    def _plot_title(self):
        plt.title(f'score={self.score:.2f} ref={self.reference_id}')


class LocalizerPlot(DeepOMLocalizer.LocalizerOutput, ImagePlot):

    def plot_locs(self):
        init_figure()
        self.plot_image(self.preprocessed_image)
        event_plot(self.localizations, pos=1)


class CropPlot(AlignmentPlot, Cropper.AlignedCrop):
    references: dict

    def plot(self):
        init_figure()

        plt.subplot(2, 1, 1)

        set_axis_props()
        self._plot_title()

        self.plot_image(self.image)
        self._plot_alignment()
        self.plot_lims()

        plt.subplot(2, 1, 2, sharex=plt.gca())
        set_axis_props()
        self.plot_crop()

    def plot_lims(self):
        plt.axvline(self.qry_start, color='m', lw=3, ls='-')
        plt.axvline(self.qry_stop, color='m', lw=3, ls='-')

    def plot_crop(self, y_extent=(1, 0), pos=1):
        start, stop = self.qry_start, self.qry_stop
        crop_image = self.y
        if self.crop_orientation == Orientation.REVERSE:
            crop_image = crop_image[:, ::-1]
        plt.imshow(crop_image, extent=[start - self.pad_amount, stop + self.pad_amount, *y_extent], aspect='auto',
                   cmap='gray',
                   interpolation='none')
        x = self.x
        r = np.flatnonzero(x)
        event_plot(self.transform_to_query(r * self.bin_size + self.ref_start) / ENV.BNX_SCALE, 'b', pos=pos,
                   linestyles='-')


class MappingResultPlot(PydanticClass):
    references: dict
    mapping_results: list[MappingResult]

    def plot_mapping_results(self):
        set_axis_props()
        qry = self.mapping_results[0].qry
        gt_ref_segment = RefSegment(**dict(qry), references=self.references).ref_segment
        event_plot(gt_ref_segment - gt_ref_segment[0], color='g', pos=0)
        for i, mapping_result in enumerate(self.mapping_results):
            ref = mapping_result.ref
            ref_segment = RefSegment(**dict(ref), references=self.references).ref_segment
            if qry.reference_id == ref.reference_id:
                ref_segment = ref_segment - gt_ref_segment[0]
            else:
                ref_segment = ref_segment - ref_segment[0]
            event_plot(ref_segment, color='b', pos=- i - 1)
