""" Faults extractor from point cloud. """
from collections import deque
import numpy as np

from cc3d import connected_components
from cv2 import dilate
from scipy.ndimage import find_objects

from batchflow import Notifier

from .base import Fault
from .postprocessing import skeletonize
from .coords_utils import (bboxes_adjacent, bboxes_embedded, bboxes_intersected, dilate_coords, find_contour,
                           depthwise_groupby_max, compute_distances, restore_coords_from_projection)
from ...utils import groupby_min, groupby_max



class FaultExtractor:
    """ Extract fault surfaces from an array with predicted and smoothed fault probabilities.

    Main naming rules, which help to understand what's going on:
    - Component is a 2d connected component on some slide.
    - Prototype is a 3d points body of merged components.
    - `coords` are spatial coordinates in format (iline, xline, depth) with (N, 3) shape.
    - `points` are coordinates and probabilities values in format (iline, xline, depth, proba) with (N, 4) shape.
      Note, that probabilities are converted into (0, 255) values for applying integer storage for points.

    The extraction algorithm is:

    1) Extract first prototype approximation as a set of similar components on neighboring slides on `direction` axis.

    For this we choose initial 2d component (first unmerged and the longest component),
    find the closest component on the next slide, and save them into one prototype.
    We repeat this operation for the new founded components until we find any closest components.
    The closest components are components which has the minimal axis-wise distances.

    We can have the situation, when we have two close components of different lengths:
    in this case we split component parts out of the overlap, save them as new components
    in the container and concatenate overlapping parts into one prototype.

    2) Extracted set of prototypes is not the targeted surfaces:
    sometimes we do extra components splitting (where prediction was lost).

    For the improvement, we concat connected prototypes (which look like puzzle details).

    For more, see the :meth:`~.concat_connected_prototypes`.

    This operation is recommended to be repeated for different axes of concatenation and different overlap thresholds.

    3) We can have the situation when we don't concat all parts of one prototype and internal
    (embedded) parts are out of the extracted surface.

    For this case we find embedded prototypes and concat them into one.
    Embedded prototypes are surfaces that are inside bboxes of other surfaces and connected
    with them at least than 2 sides.

    For more, see the :meth:`~.concat_embedded_prototypes`.

    To sum up, the whole algorithm is:

    1) Initialize container with smoothed probabilities predictions.
    2) Extract first prototype approximation with :meth:`~.extract_prototypes`.
    3) Iteratively concat connected prototypes changing concatenation axis and threshold with
    :meth:`~.concat_connected_prototypes`.
    3) Concat internal prototypes pieces with :meth:`~.concat_embedded_prototypes`.

    If you want to speed up, you can add filtering on any stage.
    As an example, you can use :meth:`~.run`.

    Parameters
    ----------
    data : np.ndarray or :class:`~.Geometry` instance, optional
        A 3d array with smoothed predictions with shape corresponds to the field shape.
        Note, that you need to provide `data` argument or `prototypes` and `shape` instead.
    direction : {0, 1}
        Extraction direction, can be 0 (ilines) or 1 (xlines) and the same as the prediction direction.
    direction_origin : int
        Data origin for the direction axis.
    prototypes : list of :class:`~.FaultPrototype` instances, optional
        Prototypes for applying :class:`~.FaultExtractor` manipulations on.
    shape : sequence of three ints, optional
        Data shape from which `prototypes` were extracted.
    skeleton_data : np.ndarray or :class:`~.Geometry` instance, optional
        Data received after the `data` skeletonize.
        Can be used for speed-up: we make skeletonize inside the extractor initialization,
        but sometimes we have this array outside the extractor for other needs.
    component_len_threshold : int
        Threshold to filter out too small connected components on data slides.
        If 0, then no filter applied (recommended for higher accuracy).
        If more than 0, then extraction will be faster.
    """
    # pylint: disable=protected-access
    def __init__(self, data=None, direction=0, direction_origin=0, prototypes=None, shape=None,
                 skeleton_data=None, component_len_threshold=0):
        # Check
        if data is None and (prototypes is None or shape is None):
            raise ValueError("`data` or `prototypes` and `shape` must be provided!")

        # Data parameters
        shape = data.shape if shape is None else shape
        self.shape = shape

        self.direction_origin = direction_origin

        self.direction = direction
        self.orthogonal_direction = 1 - self.direction

        # Internal parameters
        self.dilation = 3 # constant for internal operations
        self.component_len_threshold = component_len_threshold

        self.container = self._init_container(data=data, skeleton_data=skeleton_data) if data is not None else None
        self._unprocessed_slide_idx = self.direction_origin # variable for internal operations speed up

        self.prototypes_queue = deque() # prototypes for extension
        self.prototypes = [] if prototypes is None else prototypes # extracted prototypes

    def _init_container(self, data, skeleton_data=None):
        """ Extract connected components on each slide and save them into container. """
        dilation_structure = np.ones((1, self.dilation), np.uint8)
        container = {}

        # Process data slides: extract connected components and their info
        for slide_idx in Notifier('t')(range(self.shape[self.direction])):
            # Get smoothed data slide
            smoothed = data.take(slide_idx, axis=self.direction)

            # Get skeletonized slide
            if skeleton_data is None:
                mask = skeletonize(smoothed, width=3)
                mask = dilate(mask, (1, 3))
            else:
                mask = skeleton_data.take(slide_idx, axis=self.direction)

            # Extract connected components from the slide
            labeled = connected_components(mask > 0)
            objects = find_objects(labeled)

            # Get components info
            components, lengths = [], []

            for idx, object_bbox in enumerate(objects, start=1):
                # Extract component coords: we refine skeletonize effects by applying it on limited area
                dilation_axis = self.orthogonal_direction
                dilation_ranges = (max(0, object_bbox[0].start - self.dilation // 2),
                                   min(object_bbox[0].stop + self.dilation // 2, self.shape[dilation_axis]))

                object_mask = labeled[dilation_ranges[0]:dilation_ranges[1], object_bbox[-1]] == idx

                length = np.count_nonzero(object_mask)

                if length <= self.component_len_threshold:
                    continue

                # Get component neighboring area coords for probabilities extraction
                object_mask = dilate(object_mask.astype(np.uint8), dilation_structure)

                dilated_coords_2d = np.nonzero(object_mask)

                dilated_coords = np.zeros((len(dilated_coords_2d[0]), 3), dtype=np.int32)

                dilated_coords[:, self.direction] = slide_idx + self.direction_origin
                dilated_coords[:, dilation_axis] = dilated_coords_2d[0].astype(np.int32) + dilation_ranges[0]
                dilated_coords[:, 2] = dilated_coords_2d[1].astype(np.int32) + object_bbox[1].start

                smoothed_values = smoothed[dilated_coords[:, dilation_axis], dilated_coords[:, -1]]

                coords, probas = depthwise_groupby_max(coords=dilated_coords, values=smoothed_values)

                # Previous length and len(coords) are different
                # because original coords have more than one coord per depth when new are not
                length = len(coords)

                if length <= self.component_len_threshold:
                    continue

                lengths.append(length)

                # We convert probas to integer values for saving them in points array with 3d-coordinates
                probas = np.round(probas * 255).astype(coords.dtype)
                points = np.hstack((coords, probas.reshape(-1, 1)))

                # Bbox
                bbox = np.column_stack([np.min(coords, axis=0), np.max(coords, axis=0)])

                component = Component(points=points, slide_idx=slide_idx+self.direction_origin, bbox=bbox)
                components.append(component)

            container[slide_idx + self.direction_origin] = {
                'components': components,
                'lengths': lengths
            }

        return container


    # Prototypes extraction
    def extract_prototypes(self):
        """ Extract all fault prototypes from the point cloud. """
        prototype = self.extract_prototype()

        while prototype is not None:
            self.prototypes.append(prototype)
            prototype = self.extract_prototype()

        return self.prototypes

    def extract_prototype(self):
        """ Extract one fault prototype from the point cloud. """
        if len(self.prototypes_queue) == 0:
            component, component_idx = self._find_next_component()

            if component is None:
                return None

            self.container[component.slide_idx]['lengths'][component_idx] = -1 # mark as merged

            prototype = FaultPrototype(points=component.points, direction=self.direction, last_component=component)
        else:
            prototype = self.prototypes_queue.popleft()
            component = prototype.last_component

        # Find closest components on next slides
        for _ in range(component.slide_idx + 1, self.shape[self.direction] + self.direction_origin):
            # Find the closest component on the slide_idx_ to the current
            component, split_indices = self._find_closest_component(component=component)

            # Postprocess prototype - it need to be splitted if it is out of component ranges
            if component is not None:
                prototype, new_prototypes = prototype.split(split_indices=split_indices, axis=2)
                self.prototypes_queue.extend(new_prototypes)

                prototype.append(component)
            else:
                break

        return prototype

    def _find_next_component(self):
        """ Find the longest not merged component on the minimal slide. """
        for slide_idx in range(self._unprocessed_slide_idx, self.shape[self.direction] + self.direction_origin):
            slide_info = self.container[slide_idx]

            if len(slide_info['lengths']) > 0:
                max_component_idx = np.argmax(slide_info['lengths'])

                if slide_info['lengths'][max_component_idx] != -1:
                    self._unprocessed_slide_idx = slide_idx
                    component = self.container[slide_idx]['components'][max_component_idx]
                    return component, max_component_idx

        return None, None

    def _find_closest_component(self, component, distances_threshold=None,
                                depth_iteration_step=10, depths_threshold=20):
        """ Find the closest component to the provided on next slide, get splitting indices for prototype if needed.

        Parameters
        ----------
        component : instance of :class:`~.Component`
            Component for which find the closest on the next slide.
        distances_threshold : int, optional
            Threshold for the max possible axis-wise distance between components.
        depth_iteration_step : int
            The depth iteration step to find distances between components.
            Value 1 is recommended for higher accuracy.
            Value more than 1 is less accurate but speeds up the finding.
        depths_threshold : int
            Depth-length threshold to decide to split closest component or prototype.
            If one component is longer than another more than on depths_threshold,
            then we need to split the longest one into parts:
             - one part corresponds to the closest component;
             - another corresponds to the different component, which is not allowed for merge with the current.
        """
        # Dilate component bbox for detecting close components: component on next slide can be shifted
        dilated_bbox = component.bbox.copy()
        dilated_bbox[self.orthogonal_direction, :] += (-self.dilation // 2, self.dilation // 2)
        dilated_bbox[self.orthogonal_direction, 0] = max(0, dilated_bbox[self.orthogonal_direction, 0])
        dilated_bbox[self.orthogonal_direction, 1] = min(dilated_bbox[self.orthogonal_direction, 1],
                                                         self.shape[self.orthogonal_direction])

        min_distance = distances_threshold if distances_threshold is not None else 100

        # Init returned values
        closest_component = None
        prototype_split_indices = [None, None]

        component_split_indices = [None, None]

        # Iter over components and find the closest one
        for other_component_idx, other_component in enumerate(self.container[component.slide_idx + 1]['components']):
            if self.container[component.slide_idx + 1]['lengths'][other_component_idx] == -1:
                continue

            # Check bboxes intersection
            if not bboxes_intersected(dilated_bbox, other_component.bbox, axes=(self.orthogonal_direction, 2)):
                continue

            # Check closeness of some points (as depth-wise distances)
            # Faster then component overlap, but not so accurate
            overlap_depths = (max(component.bbox[2, 0], other_component.bbox[2, 0]),
                                    min(component.bbox[2, 1], other_component.bbox[2, 1]))

            step = min(depth_iteration_step, (overlap_depths[1]-overlap_depths[0])//3)
            step = max(step, 1)

            indices_1 = np.in1d(component.coords[:, -1], np.arange(overlap_depths[0], overlap_depths[1]+1, step))
            indices_2 = np.in1d(other_component.coords[:, -1], np.arange(overlap_depths[0], overlap_depths[1]+1, step))

            coords_1 = component.coords[indices_1, self.orthogonal_direction]
            coords_2 = other_component.coords[indices_2, self.orthogonal_direction]

            components_distances = compute_distances(coords_1, coords_2, max_threshold=min_distance)

            if (components_distances[0] == -1) or (components_distances[0] > 1):
                # Components are not close
                continue

            if components_distances[1] >= min_distance:
                # `other_component` is not the closest
                continue

            # The most depthwise distant points in components are close enough -> we can combine components
            min_distance = components_distances[1]

            closest_component = other_component
            merged_idx = other_component_idx
            overlap_borders = overlap_depths

            if min_distance == 0:
                # The closest component is founded
                break

        if closest_component is not None:
            # Process (split if needed) founded component and get split indices for prototype
            self.container[closest_component.slide_idx]['lengths'][merged_idx] = -1 # mark component as merged

            # Get prototype split indices:
            # check that the new component is smaller than the previous one (for each border)
            if overlap_borders[0] - component.bbox[2, 0] > depths_threshold:
                prototype_split_indices[0] = overlap_borders[0]

            if component.bbox[2, 1] - overlap_borders[1] > depths_threshold:
                prototype_split_indices[1] = overlap_borders[1]

            # Split new component: check that the new component is bigger than the previous one (for each border)
            # Create splitted items and save them as new elements for merge
            if overlap_borders[0] - closest_component.bbox[2, 0] > depths_threshold:
                component_split_indices[0] = overlap_borders[0]

            if closest_component.bbox[2, 1] - overlap_borders[1] > depths_threshold:
                component_split_indices[1] = overlap_borders[1]

            closest_component, new_components = closest_component.split(split_indices=component_split_indices)
            self._add_new_components(new_components)

        return closest_component, prototype_split_indices

    def _add_new_components(self, components):
        """ Add new items into the container.

        New items are creating after components splitting.
        """
        for component in components:
            if len(component) > self.component_len_threshold:
                self.container[component.slide_idx]['components'].append(component)
                self.container[component.slide_idx]['lengths'].append(len(component))


    # Prototypes concatenation
    def concat_connected_prototypes(self, overlap_ratio_threshold=None, axis=2,
                                    border_threshold=20, width_split_threshold=100):
        """ Concat prototypes which are connected.

        Under the hood, we compare prototypes with each other and find which are connected as puzzles.
        For this we get neighboring borders and compare them.
        If borders are almost overlapped after spatial shift then we merge prototypes.

        Parameters
        ----------
        overlap_ratio_threshold : float or None
            Prototypes contours overlap ratio to decide that prototypes are not close.
            Possible values are float numbers in the (0, 1] interval or None.
            If None, then default values are used: 0.5 for the depth axis and 0.9 for others.
        axis : {0, 1, 2}
            Axis along which to find prototypes connections.
            Recommended values are 2 (for depths) or self.direction.
        border_threshold : int
            Minimal amount of points out of border contours overlap to decide that prototypes are not close.
        width_split_threshold : int or None
            Contour widths (along self.direction axis) difference threshold to decide that prototypes need
            to be splitted by self.direction axis.
            If value is None, then no splitting applied. But there are the risk of interpenetration of
            triangulated surfaces in this case.
            With lower value more splitting applied and smaller prototypes are extracted.
            With higher value there are less prototypes with higher areas extracted.
            So, if you want more detailing, then provide smaller width_split_threshold (near to 10).
            If you want to extract bigger surfaces, then provide higher width_split_threshold (near to 100 or None).
        """
        margin = 1 # local constant for code prettifying

        if overlap_ratio_threshold is None:
            overlap_ratio_threshold = 0.5 if axis in (-1, 2) else 0.9

        overlap_axis = self.direction if axis in (-1, 2) else 2

        # Under the hood, we check borders connectivity (as puzzles)
        borders_to_check = ('up', 'down') if axis in (-1, 2) else ('left', 'right')

        # Presort objects by overlap axis for early stopping
        sort_axis = overlap_axis
        prototypes_starts = np.array([prototype.bbox[sort_axis, 0] for prototype in self.prototypes])
        prototypes_order = np.argsort(prototypes_starts)
        reodered_prototypes = [self.prototypes[idx] for idx in prototypes_order]

        new_prototypes = []

        for i, prototype_1 in enumerate(reodered_prototypes):
            for prototype_2 in reodered_prototypes[i+1:]:
                # Exit if we out of sort_axis ranges for prototype_1
                if (prototype_1.bbox[sort_axis, 1] < prototype_2.bbox[sort_axis, 0]):
                    break

                adjacent_borders = bboxes_adjacent(prototype_1.bbox, prototype_2.bbox)

                if adjacent_borders is None:
                    continue

                # Check that bboxes overlap is enough
                overlap_threshold = min(prototype_1.bbox[overlap_axis, 1] - prototype_1.bbox[overlap_axis, 0],
                                        prototype_2.bbox[overlap_axis, 1] - prototype_2.bbox[overlap_axis, 0])
                overlap_threshold *= overlap_ratio_threshold

                overlap_length = adjacent_borders[overlap_axis][1] - adjacent_borders[overlap_axis][0]

                if overlap_length < overlap_threshold:
                    continue

                # Find object contours on close borders
                is_first_upper = prototype_1.bbox[axis, 0] < prototype_2.bbox[axis, 0]

                contour_1 = prototype_1.get_border(border=borders_to_check[is_first_upper],
                                                   projection_axis=self.orthogonal_direction)
                contour_2 = prototype_2.get_border(border=borders_to_check[~is_first_upper],
                                                   projection_axis=self.orthogonal_direction)

                # Get border contours in the area of interest
                overlap_range = (min(adjacent_borders[axis]) - margin, max(adjacent_borders[axis]) + margin)

                contour_1 = contour_1[(contour_1[:, axis] >= overlap_range[0]) & \
                                      (contour_1[:, axis] <= overlap_range[1])]
                contour_2 = contour_2[(contour_2[:, axis] >= overlap_range[0]) & \
                                      (contour_2[:, axis] <= overlap_range[1])]

                # If one data contour is much longer than other, then we can't connect them as puzzle details
                if len(contour_1) == 0 or len(contour_2) == 0:
                    continue

                length_ratio = min(len(contour_1), len(contour_2)) / max(len(contour_1), len(contour_2))

                if length_ratio < overlap_ratio_threshold:
                    continue

                # Correct border_threshold for too short borders
                if (1 - overlap_ratio_threshold) * min(len(contour_1), len(contour_2)) < border_threshold:
                    corrected_border_threshold = 2 * margin
                else:
                    corrected_border_threshold = border_threshold

                # Shift one of the objects, making their contours intersected
                shift = 1 if is_first_upper else -1
                contour_1[:, axis] += shift

                # Check that one component contour is inside another (for both)
                # TODO: think about reordering: smaller inside the bigger
                if not (self._is_contour_inside(contour_1, contour_2, border_threshold=corrected_border_threshold) or \
                        self._is_contour_inside(contour_2, contour_1, border_threshold=corrected_border_threshold)):
                    continue

                # Split by self.direction for avoiding wrong prototypes shapes (like C or T-likable, etc.)
                if axis in (-1, 2):
                    lower_is_wider = (is_first_upper and prototype_2.width - prototype_1.width > 20) or \
                                     (not is_first_upper and prototype_1.width - prototype_2.width > 20)

                    too_big_width_diff = (width_split_threshold is not None) and \
                                         (np.abs(prototype_1.width - prototype_2.width) > width_split_threshold)

                    if (lower_is_wider or too_big_width_diff):
                        split_indices = (max(prototype_1.bbox[self.direction, 0], prototype_2.bbox[self.direction, 0]),
                                         min(prototype_1.bbox[self.direction, 1], prototype_2.bbox[self.direction, 1]))

                        prototype_1, new_prototypes_ = prototype_1.split(split_indices, axis=self.direction)
                        new_prototypes.extend(new_prototypes_)

                        prototype_2, new_prototypes_ = prototype_2.split(split_indices, axis=self.direction)
                        new_prototypes.extend(new_prototypes_)

                prototype_2.concat(prototype_1)
                prototype_1._already_merged = True
                break

        self.prototypes = [prototype for prototype in self.prototypes
                           if not getattr(prototype, '_already_merged', False)]
        self.prototypes.extend(new_prototypes)
        return self.prototypes

    def _is_contour_inside(self, contour_1, contour_2, border_threshold):
        """ Check that `contour_1` is almost inside the dilated `contour_2`.

        We apply dilation for `contour_2` because the fault can be a shifted on neighboring slides.

        Parameters
        ----------
        contour_1, contour_2 : np.ndarrays of (N, 3) shape
            Contours coordinates for check. Sorting is not required.
            Also, contours created by :meth:`.~get_border` will be sorted.
        border_threshold : int
            Minimal amount of points out of contours overlap to decide that `contour_1` is not inside `contour_2`.
        """
        contour_1_set = set(tuple(x) for x in contour_1)

        # Objects can be shifted on `self.orthogonal_direction`, so apply dilation for coords
        contour_2_dilated = dilate_coords(coords=contour_2, dilate=self.dilation,
                                          axis=self.orthogonal_direction,
                                          max_value=self.shape[self.orthogonal_direction])

        contour_2_dilated = set(tuple(x) for x in contour_2_dilated)

        contours_intersected = len(contour_1_set.intersection(contour_2_dilated)) > 0
        return contours_intersected and (len(contour_1_set - contour_2_dilated) < border_threshold)

    def concat_embedded_prototypes(self, border_threshold=100):
        """ Concat embedded prototypes with 2 or more closed borders.

        Under the hood, we compare different prototypes to find pairs in which one prototype is inside another.
        If more than two borders of internal prototype is connected with other (main) prototype, then we merge them.

        Internal logic looks similar to `.concat_connected_prototypes`,
        but now we find embedded bboxes and need two borders coincidence instead of one.

        Embedded prototypes examples:

        ||||||  or  |||||||  or  ||||||  etc.
        ...|||      |...|||      |||...
           |||      ||||||
        ||||||

         - where | means one prototype points, and . - other prototype points.

        Parameters
        ----------
        border_threshold : int
            Minimal amount of points out of borders overlap to decide that prototypes are not close.
        """
        # Presort objects by other valuable axis for early stopping
        sort_axis = self.direction
        prototypes_starts = np.array([prototype.bbox[sort_axis, 0] for prototype in self.prototypes])
        prototypes_order = np.argsort(prototypes_starts)
        reodered_prototypes = [self.prototypes[idx] for idx in prototypes_order]

        margin = 3 # local constant

        for i, prototype_1 in enumerate(reodered_prototypes):
            for prototype_2 in reodered_prototypes[i+1:]:
                # Check that prototypes are embedded
                if (prototype_1.bbox[sort_axis, 1] < prototype_2.bbox[sort_axis, 0]):
                    break

                is_embedded, swap = bboxes_embedded(prototype_1.bbox, prototype_2.bbox, margin=margin)

                if not is_embedded:
                    continue

                coords = prototype_1.coords if swap is False else prototype_2.coords
                other = prototype_2 if swap is False else prototype_1

                # Check borders connections
                close_borders_counter = 0

                for border in ('up', 'down', 'left', 'right'): # TODO: get more optimal order depend on bboxes
                    # Find internal object border contour
                    contour = other.get_border(border=border, projection_axis=self.orthogonal_direction)

                    # Shift contour to make it intersected with another object
                    shift = -1 if border in ('up', 'left') else 1
                    shift_axis = self.direction if border in ('left', 'right') else 2
                    contour[:, shift_axis] += shift

                    # Get main object coords in the area of the interest for speeding up evaluations
                    slices = other.bbox.copy()
                    slices[:, 0] -= margin
                    slices[:, 1] += margin

                    coords_sliced = coords[(coords[:, 0] >= slices[0, 0]) & (coords[:, 0] <= slices[0, 1]) & \
                                           (coords[:, 1] >= slices[1, 0]) & (coords[:, 1] <= slices[1, 1]) & \
                                           (coords[:, 2] >= slices[2, 0]) & (coords[:, 2] <= slices[2, 1])]

                    # Check that the shifted border is inside the main_object area
                    corrected_border_threshold = min(border_threshold, len(contour)//2)

                    if self._is_contour_inside(contour, coords_sliced,
                                               border_threshold=corrected_border_threshold):
                        close_borders_counter += 1

                    if close_borders_counter >= 2:
                        break

                # If objects have more than 2 closed borders then they are parts of the same prototype -> merge them
                if close_borders_counter >= 2:
                    prototype_2.concat(prototype_1)
                    prototype_1._already_merged = True
                    break

        self.prototypes = [prototype for prototype in self.prototypes
                           if not getattr(prototype, '_already_merged', False)]
        return self.prototypes


    # Addons
    def run(self, prolongate_in_depth=False, concat_iters=20, overlap_ratio_threshold=None,
            additional_filters=False, **filtering_kwargs):
        """ Recommended full extracting procedure.

        The procedure scheme is:
            - extract prototypes from point cloud;
            - filter too small prototypes (for speed up);
            - concat connected prototypes (concat by depth axis, concat by self.direction axis) concat_iters times
            with changed `overlap_ratio_threshold`;
            - filter too small prototypes (for speed up);
            - concat embedded prototypes;
            - filter all unsuitable prototypes.

        Parameters
        ----------
        prolongate_in_depth : bool
            Whether to maximally prolongate faults in depth or not.
            If True, then surfaces will be tall and thin.
            If False, then surfaces will be more longer for `self.direction` than for depth axis.
        concat_iters : int
            Maximal amount of connected component concatenation operations which are include concat along
            the depth and `self.direction` axes.
        overlap_ratio_threshold : dict or None
            Prototype borders overlap ratio to decide that prototypes can be connected.
            Note, it is decrementally changed. Keys are axes and values in the (start, stop, step) format.
        additional_filters : bool
            Whether to apply additional filtering for speed up.
        filtering_kwargs
            kwargs for the `.filter_prototypes` method.
            These kwargs are applied in the filtration after whole extraction procedure.

        Returns
        -------
        prototypes: list of the :class:`~.FaultPrototype` instances
            Resulting prototypes.
        stats : dict
            Amount of prototypes after each proceeding.
        """
        stats = {}

        if overlap_ratio_threshold is None:
            overlap_ratio_threshold = {
                self.direction: (0.9, 0.7, 0.05), # (start, stop, step)
                2: (0.9, 0.5, 0.05)
            }

        depth_overlap_threshold = overlap_ratio_threshold[2][0]
        direction_overlap_threshold = overlap_ratio_threshold[self.direction][0]

        # Extract prototypes from data
        if len(self.prototypes) == 0:
            _ = self.extract_prototypes()
        stats['extracted'] = len(self.prototypes)

        # Filter for speed up
        if additional_filters:
            self.prototypes = self.filter_prototypes(min_height=3, min_width=3, min_n_points=10)
            stats['filtered_extracted'] = len(self.prototypes)

        # Concat connected (as puzzles) prototypes
        stats['after_connected_concat'] = {}

        for i in Notifier('t')(concat_iters):
            stats['after_connected_concat'][i] = []
            # Concat by depth axis
            _ = self.concat_connected_prototypes(overlap_ratio_threshold=depth_overlap_threshold,
                                                 axis=2)
            stats['after_connected_concat'][i].append(len(self.prototypes))

            # Concat by direction axis
            if (not prolongate_in_depth) or (depth_overlap_threshold <= overlap_ratio_threshold[2][1]):
                _ = self.concat_connected_prototypes(overlap_ratio_threshold=direction_overlap_threshold,
                                                    axis=self.direction)

                stats['after_connected_concat'][i].append(len(self.prototypes))

            # Early stopping
            if (depth_overlap_threshold <= overlap_ratio_threshold[2][1]) and \
               (direction_overlap_threshold <= overlap_ratio_threshold[self.direction][1]) and \
               (stats['after_connected_concat'][i][-1] == stats['after_connected_concat'][i-1][-1]):
                break

            depth_overlap_threshold = round(depth_overlap_threshold - overlap_ratio_threshold[2][-1], 2)
            depth_overlap_threshold = max(depth_overlap_threshold, overlap_ratio_threshold[2][1])

            if (not prolongate_in_depth) or (depth_overlap_threshold <= overlap_ratio_threshold[2][1]):
                direction_overlap_threshold = round(direction_overlap_threshold - \
                                                         overlap_ratio_threshold[self.direction][-1], 2)
                direction_overlap_threshold = max(direction_overlap_threshold,
                                                       overlap_ratio_threshold[self.direction][1])

        # Filter for speed up
        if additional_filters:
            self.prototypes = self.filter_prototypes(min_height=3, min_width=3, min_n_points=10)
            stats['filtered_connected_concat'] = len(self.prototypes)

        # Concat embedded
        _ = self.concat_embedded_prototypes()
        stats['after_embedded_concat'] = len(self.prototypes)

        # Filter too small prototypes
        if additional_filters or len(filtering_kwargs) > 0:
            self.prototypes = self.filter_prototypes(**filtering_kwargs)
            stats['after_last_filtering'] = len(self.prototypes)
        return self.prototypes, stats

    def filter_prototypes(self, min_height=40, min_width=20, min_n_points=100):
        """ Filer out unsuitable prototypes.

        min_height : int
            Minimal preferred prototype height (length along the depth axis).
        min_width : int
            Minimal preferred prototype width (length along the `self.direction` axis).
        min_n_points : int
            Minimal preferred points amount.
        """
        filtered_prototypes = []

        for prototype in self.prototypes:
            if (prototype.height >= min_height) and (prototype.width >= min_width) and \
               (prototype.n_points >= min_n_points):

                filtered_prototypes.append(prototype)

        return filtered_prototypes

    def prototypes_to_faults(self, field):
        """ Convert all prototypes to faults. """
        faults = [Fault(prototype.coords, field=field) for prototype in self.prototypes]
        return faults



class Component:
    """ Container for extracted connected component.

    Parameters
    ----------
    points : np.ndarray of (N, 4) shape
        Spatial coordinates and probabilities in the (ilines, xlines, depths, proba) format.
        Sorting is not required. Also usually `points` created in :class:`.~FaultExtractor` will be unsorted.
    slide_idx : int
        Index of the slide from which component was extracted.
    bbox : np.ndarray of (3, 2) shape
        3D bounding box.
    length : int
        Component length.
    """
    def __init__(self, points, slide_idx, bbox=None, length=None):
        self.points = points
        self.slide_idx = slide_idx

        self._bbox = bbox


    @property
    def coords(self):
        """ Spatial coordinates in the (ilines, xlines, depths) format."""
        return self.points[:, :-1]

    @property
    def bbox(self):
        """ 3D bounding box. """
        if self._bbox is None:
            self._bbox = np.column_stack([np.min(self.coords, axis=0), np.max(self.coords, axis=0)])
        return self._bbox

    def __len__(self):
        """ Number of points in a component. """
        return len(self.points)


    def split(self, split_indices):
        """ Depth-wise component split by indices.

        Parameters
        ----------
        split_indices : sequence of two ints or None
            Depth values (upper and lower) to split component into parts. If None, the no need in split.

        Returns
        -------
        self : `~.Component` instance
            The most closest component to the `self` after split.
        new_components : list of `~.Component` instances
            Components created from splitted parts.
        """
        new_components = []

        # Cut upper part of the component and save it as another item
        if split_indices[0] is not None:
            # Extract closest part
            mask = self.points[:, 2] >= split_indices[0]
            self, new_component = self._split(mask=mask, split_index=split_indices[0])

            new_components.append(new_component)

        # Cut lower part of the component and save it as another item
        if split_indices[1] is not None:
            # Extract closest part
            mask = self.points[:, 2] <= split_indices[1]
            self, new_component = self._split(mask=mask, split_index=split_indices[1])

            new_components.append(new_component)

        return self, new_components

    def _split(self, mask, split_index):
        """ Split component into parts by mask. """
        # Create new Component from extra part
        new_component_points = self.points[~mask]
        new_component = Component(points=new_component_points, slide_idx=self.slide_idx)

        # Extract suitable part
        self.points = self.points[mask]
        self._bbox = None
        return self, new_component



class FaultPrototype:
    """ Class for faults prototypes. Provides a necessary API for convenient prototype extraction process.

    Note, the `last_component` parameter is preferred for prototype extraction and is optional:
    it is used for finding closest components on next slides.

    Parameters
    ----------
    points : np.ndarray of (N, 4) shape
        Prototype coordinates and probabilities.
        Sorting is not required. Also usually `points` created in :class:`.~FaultExtractor` will be unsorted.
    direction : {0, 1}
        Direction along which the prototype is extracted (the same as prediction direction).
    last_component : instance of :class:`~.Component`
        The last added component into prototype. Needs for prototypes prolongation during extraction.
    """
    def __init__(self, points, direction, last_component=None):
        self.points = points
        self.direction = direction

        self._bbox = None

        self._last_component = last_component

        self._contour = None
        self._borders = {}

    @property
    def coords(self):
        """ Spatial coordinates in (ilines, xlines, depth) format. """
        return self.points[:, :3]

    @property
    def bbox(self):
        """ 3D bounding box. """
        if self._bbox is None:
            self._bbox = np.column_stack([np.min(self.coords, axis=0), np.max(self.coords, axis=0)])
        return self._bbox

    # Stats for filtering
    @property
    def height(self):
        """ Height (length along the depth axis). """
        return self.bbox[2, 1] - self.bbox[2, 0]

    @property
    def width(self):
        """ Width (length along the `self.direction` axis). """
        return self.bbox[self.direction, 1] - self.bbox[self.direction, 0]

    @property
    def n_points(self):
        """ Amount of the surface points. """
        return len(self.points)

    @property
    def proba(self):
        """ 90% percentile of approximate proba values in [0, 1] interval. """
        proba_value = np.percentile(self.points[:, 3], 90) # is integer value from 0 to 255
        proba_value /= 255
        return proba_value

    # Properties for internal needs
    @property
    def last_component(self):
        """ Last added component. """
        if self._last_component is None:
            last_slide_idx = self.points[:, self.direction].max()

            component_points = self.points[self.points[:, self.direction] == last_slide_idx]
            self._last_component = Component(points=component_points, slide_idx=last_slide_idx)
        return self._last_component

    @property
    def contour(self):
        """ Contour of 2d projection on axis, orthogonal to self.direction.

        Note, output coordinates, which corresponding to the projection axis are zeros.
        """
        if self._contour is None:
            projection_axis = 1 - self.direction
            self._contour = find_contour(coords=self.coords, projection_axis=projection_axis)
        return self._contour


    def append(self, component):
        """ Append new component into prototype.

        Parameters
        ----------
        component : instance of :class:`~.Component`
            Component to add into the prototype.
        """
        self.points = np.vstack([self.points, component.points])

        self._contour = None
        self._borders = {}

        self._last_component = component

        self._bbox = self._concat_bbox(component.bbox)

    def concat(self, other):
        """ Concatenate two prototypes. """
        self.points = np.vstack([self.points, other.points])

        self._bbox = self._concat_bbox(other.bbox)

        self._contour = None
        self._borders = {}

        self._last_component = None

    def _concat_bbox(self, other_bbox):
        """ Concat bboxes of two objects into one. """
        bbox = np.empty((3, 2), np.int32)
        bbox[:, 0] = np.min((self.bbox[:, 0], other_bbox[:, 0]), axis=0)
        bbox[:, 1] = np.max((self.bbox[:, 1], other_bbox[:, 1]), axis=0)
        return bbox

    def _separate_objects(self, points, axis):
        """ Separate points into different object points depend on their connectedness by axis.

        After split we can have the situation when splitted part has more than one connected items.
        This method separate disconnected parts into different prototypes.
        """
        # Get coordinates along the axis
        unique_direction_points = np.unique(points[:, axis])

        # Slides distance more than 1 -> different objects
        split_indices = np.nonzero(unique_direction_points[1:] - unique_direction_points[:-1] > 1)[0]

        if len(split_indices) == 0:
            return [FaultPrototype(points=points, direction=self.direction)]

        # Separate disconnected objects and create new prototypes instances
        start_indices = unique_direction_points[split_indices + 1]
        start_indices = np.insert(start_indices, 0, 0)

        end_indices = unique_direction_points[split_indices]
        end_indices = np.append(end_indices, unique_direction_points[-1])

        prototypes = []

        for start_idx, end_idx in zip(start_indices, end_indices):
            points_ = points[(start_idx <= points[:, axis]) & (points[:, axis] <= end_idx)]
            prototype = FaultPrototype(points=points_, direction=self.direction)
            prototypes.append(prototype)

        return prototypes


    def split(self, split_indices, axis=2):
        """ Axis-wise prototypes split by indices.

        Parameters
        ----------
        split_indices : sequence of two ints or None
            Axis values (upper and lower) to split component into parts. If None, the no need in split.

        Returns
        -------
        prototype : `~.FaultPrototype` instance
            The most closest prototype to the splitted.
        new_prototypes : list of `~.FaultPrototype` instances
            Prototypes created from disconnected objects.
        """
        new_prototypes = []

        # No splitting applied
        if (split_indices[0] is None) and (split_indices[1] is None):
            return self, new_prototypes

        axis_for_objects_separating = self.direction if axis in (-1, 2) else 2

        # Cut upper part and separate disconnected objects
        if (split_indices[0] is not None) and \
           (np.min(self.points[:, axis]) < split_indices[0] < np.max(self.points[:, axis])):

            points_outer = self.points[self.points[:, axis] < split_indices[0]]
            self.points = self.points[self.points[:, axis] >= split_indices[0]]

            if len(points_outer) > 0:
                new_prototypes.extend(self._separate_objects(points_outer, axis=axis_for_objects_separating))

        # Cut lower part and separate disconnected objects
        if (split_indices[1] is not None) and \
           (np.min(self.points[:, axis]) < split_indices[1] < np.max(self.points[:, axis])):

            points_outer = self.points[self.points[:, axis] > split_indices[1]]
            self.points = self.points[self.points[:, axis] <= split_indices[1]]

            if len(points_outer) > 0:
                new_prototypes.extend(self._separate_objects(points_outer, axis=axis_for_objects_separating))

        new_prototypes.extend(self._separate_objects(self.points, axis=axis_for_objects_separating))

        # Update self
        self.points = new_prototypes[-1].points
        self._bbox = new_prototypes[-1].bbox
        self._contour = None
        self._borders = {}
        self._last_component = None
        return self, new_prototypes[:-1]


    def get_border(self, border, projection_axis):
        """ Get contour border.

        Parameters
        ----------
        border : {'up', 'down', 'left', 'right'}
            Which object border to get.
        projection_axis : {0, 1}
            Which projection is used to get the 2d contour coordinates.

        Returns
        -------
        border : np.ndarray of (N, 3) shape
            Sorted coordinates of the requested border.
        """
        if border not in self._borders:
            # Delete extra border from contour
            # For border removing we apply groupby which works only for the last axis, so we swap axes coords
            if border in ('left', 'right'):
                border_coords = self.contour.copy()
                border_coords[:, [-1, 1-projection_axis]] = border_coords[:, [1-projection_axis, -1]]
                border_coords = border_coords[border_coords[:, 1-projection_axis].argsort()] # Groupby needs sorted data
            else:
                border_coords = self.contour

            # Delete border by applying groupby
            if border not in ('up', 'left'):
                border_coords = groupby_max(border_coords)
            else:
                border_coords = groupby_min(border_coords)

            # Restore 3d coordinates
            projection_axis = 1 - self.direction

            if border in ('left', 'right'):
                border_coords[:, [-1, 1-projection_axis]] = border_coords[:, [1-projection_axis, -1]]

            border_coords = restore_coords_from_projection(coords=self.coords, buffer=border_coords,
                                                           axis=projection_axis)
            self._borders[border] = border_coords

        return self._borders[border]
