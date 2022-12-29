""" [Draft] for faults validation algorithm.

Rework all logic of this algorithm or remove it. """
import numpy as np
from scipy.ndimage import binary_dilation, find_objects
from cc3d import connected_components
from sklearn.neighbors import KDTree

from ...utils import groupby_all


def validate_fault(coords=None, fault=None, depth_step=1, length_threshold=5, depth_break_off_threshold=20,
                   distances_threshold=1, components_distances_threshold=2, projection_axis=0):
    """ Fault validation algorithm.

    The idea is to see fault changes by depths slices and to detect its parts divergence
    into branches with different directions. For this we find the closest slide, containing anchor line
    (line before splitting into branches) and compare next slides with it.
    If on the next slides components are on different sides of anchor, then the fault is invalid.

    Note, if fault is created from sticks or simplices, then fault branches can be connected and
    this method doesn't work correctly.

    Parameters
    ----------
    projection_axis : {0, 1}
        ..!!.. Use orthogonal to prediction orientation.

    ..!!..
    """
    if (fault is None) and (coords is None):
        raise ValueError("Invalid input arguments: please, provide a fault or fault coordinates!")

    if coords is None:
        coords = fault.points

    kwargs = {
        'depth_break_off_threshold': depth_break_off_threshold,
        'length_threshold': length_threshold, 
        'distances_threshold': distances_threshold,
        'components_distances_threshold': components_distances_threshold,
        'projection_axis': projection_axis
    }

    depths_ranges = (np.min(coords[:, -1]), np.max(coords[:, -1]))

    # Find anchor line and go up and down for finding invalid branching
    if (len(np.unique(coords[:, 0])) > 1) and (len(np.unique(coords[:, 1])) > 1):
        bbox = np.column_stack([np.min(coords, axis=0), np.max(coords, axis=0)])
        anchor_line_init, anchor_depth_init = _init_anchor_line(coords=coords, bbox=bbox)

        # Go up
        order = -1
        anchor_line = anchor_line_init
        for depth in range(anchor_depth_init+order*depth_step, depths_ranges[0]+1*order, order*depth_step):
            is_valid, anchor_line = _is_valid_depth_slides(coords=coords, bbox=bbox, anchor_line=anchor_line,
                                                           order=order, depth=depth, end_depth=depths_ranges[0]-1, **kwargs)
            if not is_valid:
                return False

        # Go down
        order = +1
        anchor_line = anchor_line_init
        for depth in range(anchor_depth_init+order*depth_step, depths_ranges[1]+1*order, order*depth_step):
            is_valid, anchor_line = _is_valid_depth_slides(coords=coords, bbox=bbox, anchor_line=anchor_line,
                                                           order=order, depth=depth, end_depth=depths_ranges[1]+1, **kwargs)
            if not is_valid:
                return False
    return True

def _init_anchor_line(coords, bbox): 
    """ Find initial anchor line. """
    # Find initial anchor line as a longest line on a depth-slide in the object

    # For this we iter over depths in the some range and try to find longest depthwise line which consists the smallest components amount
    # Sort depths by object points amount
    coords_ = np.zeros_like(coords)
    coords_[:, 0] = np.sort(coords[:, -1])

    res = groupby_all(coords_)
    occurrences_sort = np.argsort(res[:, 2])[::-1]

    sorted_occurrences = res[occurrences_sort, 2]
    sorted_depths = res[occurrences_sort, 0] # sort depths by occurrences

    selected_anchor_depth = None
    selected_anchor_line = None
    selected_n_objects = -1

    max_occurrences = res[occurrences_sort[0], 2]
    min_occurrences = 0.8 * max_occurrences # For early exit: too small component are not interesting

    for depth, occurrences in zip(sorted_depths, sorted_occurrences):
        if occurrences < min_occurrences:
            break

        depth_slice = coords[coords[:, -1] == depth, :-1]

        mask_fill = depth_slice - bbox[:-1, 0]
        mask_shape = np.max(mask_fill, axis=0) + 1

        mask = np.zeros(mask_shape, int)
        mask[mask_fill[:, 0], mask_fill[:, 1]] = 1

        labeled = connected_components(mask)
        n_objects = labeled.max()

        if (selected_anchor_depth is None) or (n_objects < selected_n_objects):
            selected_anchor_depth = depth
            selected_anchor_line = labeled
            selected_n_objects = n_objects

        if n_objects == 1:
            break

    return selected_anchor_line, selected_anchor_depth

def _is_valid_depth_slides(coords, bbox, anchor_line,
                           order, depth, depth_break_off_threshold, 
                           length_threshold, distances_threshold, components_distances_threshold, 
                           end_depth, projection_axis):
    """ Validate the slide or update the anchor.

    ..!!.."""
    # Validate depth slide by anchor_line contour: if new mask has componets separated by ankhor line -> then it is invalid
    if (order > 0) and (depth + depth_break_off_threshold > end_depth): # TODO properly before the function call
        return True, anchor_line
    if (order < 0) and (depth - depth_break_off_threshold < end_depth):
        return True, anchor_line

    depth_slice = coords[coords[:, -1] == depth, :-1]

    if len(depth_slice) <= length_threshold:
        return True, anchor_line

    # Find components on the depthwise slide
    mask_fill = depth_slice - bbox[:-1, 0]
    mask_shape = np.max(mask_fill, axis=0) + 1

    mask = np.zeros(mask_shape, int)
    mask[mask_fill[:, 0], mask_fill[:, 1]] = 1

    labeled = connected_components(mask)
    objects = find_objects(labeled)

    if (len(objects) == 1) and ((labeled > 0).sum() >= 0.8*(anchor_line > 0).sum()): # TEST
        anchor_line = labeled
    else:
        if (labeled > 0).sum() > (anchor_line > 0).sum():
            # new line is bigger than ankhor and consists it -> renew ankhor
            ankhor_mask = (anchor_line > 0).astype(int)
            labeled_mask_dilated = binary_dilation(labeled > 0, iterations=1).astype(int)

            shape = ankhor_mask.shape
            labeled_mask_dilated = labeled_mask_dilated[:shape[0], :shape[1]]

            background = np.zeros_like(ankhor_mask)
            background[:labeled_mask_dilated.shape[0], :labeled_mask_dilated.shape[1]] = labeled_mask_dilated

            if (ankhor_mask - background > 0).sum() < 3:
                anchor_line = labeled 
                return True, anchor_line

        if len(objects) > 1:      
            # Check objects distances (between objects)
            distances = []

            for i in range(len(objects)-1):                        
                distances_i = []

                coords_i = np.nonzero(labeled == i + 1)
                length = len(coords_i[0])

                if length <= length_threshold:
                    continue

                # coords_i = np.column_stack([coords_i[0], coords_i[1]])
                coords_i = coords_i[projection_axis].reshape(-1, 1)
                tree = KDTree(coords_i)

                for j in range(0, len(objects)): # optimize it
                    if i != j:
                        coords_j = np.nonzero(labeled == j + 1)
                        # coords_j = np.column_stack([coords_j[0], coords_j[1]])
                        coords_j = coords_j[projection_axis].reshape(-1, 1)

                        distances_ij, _ = tree.query(coords_j)
                        distances_i.append(np.min(distances_ij))

                if len(distances_i) > 0:
                    distances.append(np.min(distances_i))

            if len(distances) == 0 or np.max(distances) < components_distances_threshold:
                # objects are the one splitted object or close
                return True, anchor_line


            # Extract ankhor line coords - optimize (out of cycle)
            anchor_line_coords = np.nonzero(anchor_line)
            anchor_line_coords = np.column_stack(anchor_line_coords)

            ankhor_tree_flattened = KDTree(anchor_line_coords[:, projection_axis].reshape(-1, 1))

            dilated_ankhor_coords = np.zeros((len(anchor_line_coords)*3, 2), int)

            for idx, i in enumerate(range(-1, 2)):
                dilated_ankhor_coords[len(anchor_line_coords)*idx:len(anchor_line_coords)*(idx+1), 0] = anchor_line_coords[:, 0]
                dilated_ankhor_coords[len(anchor_line_coords)*idx:len(anchor_line_coords)*(idx+1), 1] = anchor_line_coords[:, 1] + i

            ankhor_tree = KDTree(dilated_ankhor_coords)
            anchor_line_coords = dilated_ankhor_coords

            # Check objects distances (between object and ankhor)
            distances = []

            for i in range(len(objects)):                        
                coords_i = np.nonzero(labeled == i + 1)
                length = len(coords_i[0])

                if length <= length_threshold:
                    continue

                coords_i = coords_i[projection_axis].reshape(-1, 1)
                distances_i, _ = ankhor_tree_flattened.query(coords_i)

                distances.append(np.median(distances_i)) # or 0.9 percentile? or max?

            if len(distances) == 0 or np.max(distances) <= distances_threshold:
                # objects are close to the ankhor
                return True, anchor_line

            # Find objects signs
            signs = set()

            for idx, obj in enumerate(objects):
                label = labeled == idx + 1

                coords_ = np.nonzero(label)
                coords_ = np.column_stack([coords_[0], coords_[1]])

                length = len(coords_)

                if length <= length_threshold:
                    continue

                sign = _distance_sign(ankhor_coords=anchor_line_coords, ankhor_tree=ankhor_tree,
                                      points=coords_, projection_axis=projection_axis)
                signs.add(sign)

                if len(signs) > 1:
                    break

            # Exit
            if len(signs) > 1:
#                # Visual validation, will be removed
#                 print('bad: signs: ', signs, 'depth: ', depth, 'distances ', distances)

#                 mask_shape = np.max([np.max(anchor_line_coords, axis=0), np.max(mask_fill, axis=0)], axis=0) + 1
#                 mask_ = np.zeros(mask_shape, int)

#                 mask_line = mask_.copy()
#                 mask_line[anchor_line_coords[:, 0], anchor_line_coords[:, 1]] = 1

#                 mask_[mask_fill[:, 0], mask_fill[:, 1]] = 2
#                 plot([mask_line, mask_, [mask_line + mask_]], grid='', cmap='viridis', colorbar=True)

                return False, anchor_line
    return True, anchor_line

def _distance_sign(ankhor_coords, ankhor_tree, points, projection_axis):
    """ Distance sign between ankhor and object lines. """

    # Get distances between closest points
    distances, indices = ankhor_tree.query(points)
    distances[distances <= 0.5] = 0 # close to zero

    # Early exit if lines are close enough
    if ((distances == 0).sum() >= (distances > 0).sum()):
        return 0

    # Find signed distances, as orientation-wise distances between closest points
    signed_distances = []
    for idx, coord in enumerate(points): # maybe need orientation param
        signed_distance = (coord - ankhor_coords[indices[idx]])[0][projection_axis]
        signed_distances.append(signed_distance)

    # Found sign for the line
    signed_distances = np.array(signed_distances)

    zero_amount = (signed_distances == 0).sum()
    plus_amount = (signed_distances > 0).sum()
    minus_amount = (signed_distances < 0).sum()

    if (zero_amount >= plus_amount) and (zero_amount >= minus_amount):
        return 0
    if (plus_amount >= minus_amount):
        return +1
    return -1    
