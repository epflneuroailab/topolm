import numpy as np

class NeuronSmoothing:
    def __init__(self, fwhm_mm, resolution_mm):
        self.fwhm_mm = fwhm_mm
        self.resolution_mm = resolution_mm

    def __call__(self, positions=None, activations=None):

        def _get_grid_coord(x, y):
            xmin, xmax = np.floor(np.min(x)), np.ceil(np.max(x))
            ymin, ymax = np.floor(np.min(y)), np.ceil(np.max(y))
            grids = np.array(np.meshgrid(np.arange(xmin, xmax + 1, self.resolution_mm),
                                         np.arange(ymin, ymax + 1, self.resolution_mm)))
            gridx = grids[0].flatten().reshape(-1, 1)
            gridy = grids[1].flatten().reshape(-1, 1)
            return gridx, gridy

        # Get voxel coordinates
        tissue_x = positions[:, 0][:, np.newaxis]
        tissue_y = positions[:, 1][:, np.newaxis]
        gridx, gridy = _get_grid_coord(tissue_x, tissue_y)

        # compute sigma from fwmh
        sigma = self.fwhm_mm / np.sqrt(8. * np.log(2))

        # define gaussian kernel
        d_square = (tissue_x - gridx.T) ** 2 + (tissue_y - gridy.T) ** 2
        gaussian_filter = 1. / (2 * np.pi * sigma ** 2) * np.exp(- d_square / (2 * sigma ** 2))

        features_smoothed = np.dot(activations, gaussian_filter)
        
        return gridx, gridy, features_smoothed
