import logging
import numpy as np
from pathlib import Path, PurePath
from typing import List
from skimage import color, io
import skimage.morphology as morphology
from skimage.measure import regionprops, label


# set up logging
logger = logging.getLogger(__name__)


def load_mask(mask: np.ndarray or Path or str) -> np.ndarray:
    """
    Load a mask image (a segmentation ora focus area) and its intermediate values. Returns an array
    with only values 0 and 255.

    :param mask: input mask
    :return:
    """

    # load mask if path
    if isinstance(mask, str):
        logger.debug(f" Opening {Path(mask)}")
        mask_array = io.imread(Path(mask))
    elif isinstance(mask, PurePath):
        logger.debug(f" Opening {mask}")
        mask_array = io.imread(mask)
    elif isinstance(mask, np.ndarray):
        mask_array = mask
    else:
        raise TypeError(f"mask of type {type(mask)} not supported.")

    # convert to correct shape
    logger.debug(msg=f"Mask shape: {mask_array.shape}")
    if len(mask_array.shape) != 2:
        if mask_array.shape[2] == 3:
            mask_array = color.rgb2gray(mask_array)
        elif mask_array.shape[2] == 4:
            mask_array = color.rgb2gray(color.rgba2rgb(mask_array))
        else:
            raise RuntimeError(f"Found image of shape {mask_array.shape}")

    # removes shades
    mask_array = 255 * (mask_array > 0)
    # convert to uint8
    mask_array = mask_array.astype(np.uint8)

    return mask_array


def volume_fraction(input_segmentation: np.ndarray or Path or str,
                    interest_zone: np.ndarray or Path or str or None = None,
                    is_interest_zone_reverse: bool = True):
    """
    Estimate the volume fraction of a given vessel segmentation, i.e. the ratio between the white pixels and the total
    number of pixels.

    :param input_segmentation: input segmentation
    :param interest_zone: interest zone
    :param is_interest_zone_reverse: True if values == 255 in the interest zone should be excluded. Default is True.
    :return: the volume fraction.
    """
    input_segmentation = load_mask(input_segmentation)  # load segmentation

    if interest_zone is None:
        # count elements
        area = input_segmentation.shape[0] * input_segmentation.shape[1]
        # count white elements
        volume = np.count_nonzero(input_segmentation == 255)
        # compute volume fraction
        vf = volume / area
    else:
        interest_zone = load_mask(interest_zone)  # load interest zone
        interest_value = 0 if is_interest_zone_reverse else 255

        # evaluate interest area
        interest_area = np.count_nonzero(interest_zone == interest_value)
        # count all white pixels
        volume = np.count_nonzero((input_segmentation == 255) & (interest_zone == interest_value))
        # compute volume fraction
        vf = volume / interest_area

    return vf


def find_branch_points(input_segmentation: np.ndarray or Path or str):
    """
    Get array with branch points. Each branch point is a group of pixels.

    :param input_segmentation:
    :return:
    """
    input_segmentation = load_mask(input_segmentation)  # load segmentation
    skeleton = morphology.skeletonize(input_segmentation)  # skeletonize

    # get branching point
    branch_points = np.zeros(skeleton.shape, dtype=bool)
    for i in range(1, skeleton.shape[0] - 1):
        for j in range(1, skeleton.shape[1] - 1):
            # if part of the skeleton
            if skeleton[i][j]:
                # get near pixels
                near_skeleton_pixels = np.array([skeleton[i + 1][j + 1],
                                                 skeleton[i + 1][j],
                                                 skeleton[i + 1][j - 1],
                                                 skeleton[i][j + 1],
                                                 skeleton[i][j - 1],
                                                 skeleton[i - 1][j + 1],
                                                 skeleton[i - 1][j],
                                                 skeleton[i - 1][j - 1]])
                # check pixels near
                if sum(near_skeleton_pixels) > 2:
                    # select point as branch
                    branch_points[i][j] = True

    return branch_points


def n_branch_points(input_segmentation: np.ndarray or Path or str,
                    interest_zone: np.ndarray or Path or str or None = None,
                    is_interest_zone_reverse: bool = True):
    """
    Get nnumber of branch points for the given segmentation

    :param input_segmentation:
    :param interest_zone:
    :param is_interest_zone_reverse:
    :return:
    """

    branch_points = find_branch_points(input_segmentation)

    if interest_zone is None:
        number_of_branches = len(regionprops(label(branch_points)))  # count branches
    else:
        interest_zone = load_mask(interest_zone)  # load interest zone
        if is_interest_zone_reverse:
            branch_points[interest_zone == 255] = False  # remove zone out of interest
        else:
            branch_points[interest_zone == 0] = False  # remove zone out of interest
        number_of_branches = len(regionprops(label(branch_points)))  # count number of branches

    return number_of_branches


def branches_per_area(input_segmentation: np.ndarray or Path or str,
                      interest_zone: np.ndarray or Path or str or None = None,
                      is_interest_zone_reverse: bool = True):
    """
    Estimate the branches per area for a given segmentation.

    :param input_segmentation: input segmentation
    :param interest_zone: interest zone
    :param is_interest_zone_reverse: True if values == 255 in the interest zone should be excluded. Default is True.
    :return: the volume fraction.
    :return: the branches per area for the given segmentation
    """
    branch_points = find_branch_points(input_segmentation)

    if interest_zone is None:
        number_of_branches = len(regionprops(label(branch_points)))   # count branches
        focus_area = branch_points.shape[0] * branch_points.shape[1]  # get number of pixels
        bpa = number_of_branches / focus_area  # compute bpa
    else:
        interest_zone = load_mask(interest_zone)  # load interest zone
        interest_zone = interest_zone > 0  # convert in boolean
        if is_interest_zone_reverse:
            branch_points[interest_zone] = False  # remove zone out of interest
            focus_area = np.sum(~interest_zone)  # estimate area
        else:
            branch_points[~interest_zone] = False  # remove zone out of interest
            focus_area = np.sum(interest_zone)  # estimate area
        logger.debug(f"{branch_points.shape}")

        logger.debug(f"Segmentation {input_segmentation} regionprops: {regionprops(label(branch_points))}")
        number_of_branches = len(regionprops(label(branch_points)))  # count number of branches
        bpa = number_of_branches / focus_area  # eval branches per area

    logger.debug(f"Segmentation {input_segmentation} has {number_of_branches} bp with area {focus_area}."
                 f" Resulting bpa = {bpa}")

    return bpa


def branches_per_length(input_segmentation: np.ndarray or Path or str,
                        interest_zone: np.ndarray or Path or str or None = None,
                        is_interest_zone_reverse: bool = True):
    """
    Estimate branches per unit network length.

    :param input_segmentation:
    :param interest_zone:
    :param is_interest_zone_reverse:
    :return:
    """

    input_segmentation = load_mask(input_segmentation)

    skeleton = morphology.skeletonize(input_segmentation)  # skeletonize

    branch_points = find_branch_points(input_segmentation)

    # if interest zone is not none, manage it
    if interest_zone is not None:
        interest_zone = load_mask(interest_zone)  # load interest zone

        if is_interest_zone_reverse:
            skeleton[interest_zone == 255] = 0  # remove zone out of interest
            branch_points[interest_zone == 255] = 0
        else:
            skeleton[interest_zone == 0] = 0  # remove zone out of interest
            branch_points[interest_zone == 0] = 0

    total_length = np.count_nonzero(skeleton)  # compute total length
    number_of_branches = len(regionprops(label(branch_points)))  # compute number of branch points
    if total_length > 0:
        bpl = number_of_branches / total_length  # compute branches per length
    else:
        bpl = 0

    return bpl


def distance_transform(input_segmentation: np.ndarray or Path or str,
                       interest_zone: np.ndarray or Path or str or None = None,
                       is_interest_zone_reverse: bool = True):
    """
    Computes the distance transform for the given vessel segmentation.
    It is useful to estimate the width of the vessels.

    :param input_segmentation:
    :param interest_zone:
    :param is_interest_zone_reverse:
    :return:
    """
    # load segmentation
    input_segmentation = load_mask(input_segmentation)

    # get distance transform
    skeleton, dist = morphology.medial_axis(input_segmentation, return_distance=True)
    dist_transform = skeleton * dist

    # if interest zone is not none, manage it
    if interest_zone is not None:
        interest_zone = load_mask(interest_zone)  # load interest zone

        if is_interest_zone_reverse:
            dist_transform[interest_zone == 255] = 0
        else:
            dist_transform[interest_zone == 0] = 0

    return dist_transform


def vascularization_degree(input_segmentation: np.ndarray or Path or str,
                           interest_zone: np.ndarray or Path or str or None = None,
                           is_interest_zone_reverse: bool = True):
    """
    Compute vascularization degree as defined by Maugeri, A., Lombardo, G. E., Navarra, M., Cirmi, S. & Rapisarda, A.
    The chorioallantoic membrane: A novel approach to extrapolate data from a well-established method. J. Appl. Toxicol.
    42, 995–1003 (2022).

    :param input_segmentation:
    :param interest_zone:
    :param is_interest_zone_reverse:
    :return:
    """
    input_segmentation = load_mask(input_segmentation)

    skeleton = morphology.skeletonize(input_segmentation)  # skeletonize

    # if interest zone is not none, manage it
    if interest_zone is None:
        # eval area
        area = input_segmentation.shape[0] * input_segmentation.shape[1]
        # eval length
        total_length = np.count_nonzero(skeleton)  # compute total length
        # compute vascularization degree
        if total_length > 0 and area > 0:
            vd = total_length / area  # compute branches per length
        else:
            vd = 0
    else:
        interest_zone = load_mask(interest_zone)  # load interest zone
        interest_value = 0 if is_interest_zone_reverse else 255
        # evaluate interest area
        interest_area = np.count_nonzero(interest_zone == interest_value)
        # eval length
        length = np.count_nonzero(skeleton & (interest_zone == interest_value))
        # compute vascularization degree
        if length > 0 and interest_area > 0:
            vd = length / interest_area
        else:
            vd = 0

    return vd


def compute_interest_area(interest_zone: np.ndarray or Path or str or None = None,
                          is_interest_zone_reverse: bool = True):
    if interest_zone is None:
        logger.log(level=logging.WARNING, msg=f"No interest zone provided, returning NaN")
        return np.nan
    else:
        interest_zone = load_mask(interest_zone)  # load interest zone
        interest_value = 0 if is_interest_zone_reverse else 255

        # evaluate interest area
        interest_area = np.count_nonzero(interest_zone == interest_value)

        return interest_area


def network_length_per_area(input_segmentation: np.ndarray or Path or str,
                            interest_zone: np.ndarray or Path or str or None = None,
                            is_interest_zone_reverse: bool = True):
    input_segmentation = load_mask(input_segmentation)

    skeleton = morphology.skeletonize(input_segmentation)  # skeletonize

    # if interest zone is not none, manage it
    if interest_zone is not None:
        interest_zone = load_mask(interest_zone)  # load interest zone

        if is_interest_zone_reverse:
            skeleton[interest_zone == 255] = False  # remove zone out of interest
            interest_area = np.count_nonzero(interest_zone == 255)
        else:
            skeleton[interest_zone == 0] = 0  # remove zone out of interest
            interest_area = np.count_nonzero(interest_zone == 0)
    else:
        interest_area = input_segmentation.shape[0] * input_segmentation.shape[1]

    total_length = np.count_nonzero(skeleton)  # compute total length

    return total_length / interest_area


def compute_max_radius(input_segmentation: np.ndarray or Path or str,
                       interest_zone: np.ndarray or Path or str or None = None,
                       is_interest_zone_reverse: bool = True):
    # compute distance transform
    dt = distance_transform(input_segmentation, interest_zone, is_interest_zone_reverse)
    # get all radiuses
    all_radiuses = dt[dt > 0]
    # return max
    return np.amax(all_radiuses)


def compute_median_radius(input_segmentation: np.ndarray or Path or str,
                          interest_zone: np.ndarray or Path or str or None = None,
                          is_interest_zone_reverse: bool = True):
    # compute distance transform
    dt = distance_transform(input_segmentation, interest_zone, is_interest_zone_reverse)
    # get all radiuses
    all_radiuses = dt[dt > 0]
    # return median
    return np.median(all_radiuses)


def compute_mean_radius(input_segmentation: np.ndarray or Path or str,
                          interest_zone: np.ndarray or Path or str or None = None,
                          is_interest_zone_reverse: bool = True):
    # compute distance transform
    dt = distance_transform(input_segmentation, interest_zone, is_interest_zone_reverse)
    # get all radiuses
    all_radiuses = dt[dt > 0]
    # return median
    return np.mean(all_radiuses)


def compute_angiometrics(input_segmentation: np.ndarray or Path or str,
                         interest_zone: np.ndarray or Path or str or None = None,
                         is_interest_zone_reverse: bool = True,
                         include_angiometrics: List[str] or None = None):
    """
    Compute all angiometrics for the given segmentation. Returns a dictionary of all angiometrics.
    Currently contains:
    * vf (volume fraction)
    * bpa (branches per area)
    * bpl (branches per length)
    * median_radius
    * max_radius
    * mean_radius
    * vd (vascularization degree, as defined in Maugeri et al.)
    * nlpa (network length per area)
    * ia (interest area)
    * 1/ia (inverse of interest area)

    :param input_segmentation:
    :param interest_zone:
    :param is_interest_zone_reverse:
    :param include_angiometrics:
    :return:
    """

    # compute inverse area
    ia = compute_interest_area(interest_zone, is_interest_zone_reverse)

    # check if include_angiometrics is None
    if include_angiometrics is None:
        # if True, compute all angiometrics
        angiometrics_dict = {
            "vf": volume_fraction(input_segmentation, interest_zone, is_interest_zone_reverse),
            "bpa": branches_per_area(input_segmentation, interest_zone, is_interest_zone_reverse),
            "bpl": branches_per_length(input_segmentation, interest_zone, is_interest_zone_reverse),
            "median_radius": compute_median_radius(input_segmentation, interest_zone, is_interest_zone_reverse),
            "max_radius": compute_max_radius(input_segmentation, interest_zone, is_interest_zone_reverse),
            "mean_radius": compute_mean_radius(input_segmentation, interest_zone, is_interest_zone_reverse),
            "vd": vascularization_degree(input_segmentation, interest_zone, is_interest_zone_reverse),
            "ia": ia,
            "1/ia": 1 / ia,
            "nlpa": network_length_per_area(input_segmentation, interest_zone, is_interest_zone_reverse)
        }
    else:
        # compute each of the included angiometrics
        angiometrics_dict = {}
        if "vf" in include_angiometrics:
            angiometrics_dict["vf"] = volume_fraction(input_segmentation, interest_zone, is_interest_zone_reverse)
        if "bpa" in include_angiometrics:
            angiometrics_dict["bpa"] = branches_per_area(input_segmentation, interest_zone, is_interest_zone_reverse)
        if "bpl" in include_angiometrics:
            angiometrics_dict["bpl"] = branches_per_length(input_segmentation, interest_zone, is_interest_zone_reverse)
        if "median_radius" in include_angiometrics:
            angiometrics_dict["median_radius"] = compute_median_radius(input_segmentation,
                                                                       interest_zone,
                                                                       is_interest_zone_reverse)
        if "max_radius" in include_angiometrics:
            angiometrics_dict["max_radius"] = compute_max_radius(input_segmentation,
                                                                 interest_zone,
                                                                 is_interest_zone_reverse)
        if "mean_radius" in include_angiometrics:
            angiometrics_dict["mean_radius"] = compute_mean_radius(input_segmentation,
                                                                   interest_zone,
                                                                   is_interest_zone_reverse)
        if "vd" in include_angiometrics:
            angiometrics_dict["vd"] = vascularization_degree(input_segmentation,
                                                             interest_zone,
                                                             is_interest_zone_reverse)
        if "ia" in include_angiometrics:
            angiometrics_dict["ia"] = ia
        if "1/ia" in include_angiometrics:
            angiometrics_dict["1/ia"] = 1 / ia
        if "nlpa" in include_angiometrics:
            angiometrics_dict["nlpa"] = network_length_per_area(input_segmentation,
                                                                interest_zone,
                                                                is_interest_zone_reverse)

    return angiometrics_dict
