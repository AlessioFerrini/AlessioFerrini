"""
File containing the FEniCS Expressions used throughout the simulation
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)


class VesselReconstruction:
    """
    Generate the field reconstruction of capillary network starting from an image.
    """
    def __init__(self,
                 binary_array: np.ndarray,
                 mesh_Lx,
                 mesh_Ly):

        # save Lx and Ly
        self.mesh_Lx = mesh_Lx
        self.mesh_Ly = mesh_Ly

        # check if image ratio is preserved
        if not np.isclose(mesh_Lx / mesh_Ly, binary_array.shape[1] / binary_array.shape[0]):
            raise RuntimeError("Mesh and image must have the same ratio")

        # get number of pixel
        self.n_p_x = binary_array.shape[1]
        self.n_p_y = binary_array.shape[0]

        # save binary array as property
        self.boolean_binary_array = binary_array.astype(bool)

    def eval(self, x):
        # get array indices corresponding to the x values
        i = (self.n_p_y - 1) - np.round((x[1] / self.mesh_Ly) * (self.n_p_y - 1))
        i = i.astype(int)
        j = np.round((x[0] / self.mesh_Lx) * (self.n_p_x - 1))
        j = j.astype(int)

        return np.where(self.boolean_binary_array[i, j], 1, -1)

