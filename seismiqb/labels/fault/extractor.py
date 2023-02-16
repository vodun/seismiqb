""" Faults extractor from point cloud. """
from collections import deque
import numpy as np

from cc3d import connected_components
from cv2 import dilate
from scipy.ndimage import find_objects

from batchflow import Notifier

from .base import Fault
from .postprocessing import skeletonize
from .utils import (bboxes_adjacent, bboxes_embedded, bboxes_intersected, dilate_coords, find_contour,
                    thin_coords, min_max_depthwise_distances, restore_coords_from_projection)
from ...utils import groupby_min, groupby_max


class FaultExtractor:
    """ ..!!..

    Main logic implementation. There are much to do and much to optimize in this code.

    A few comments about the main idea:
    We extract connected components on each slide.
    We take unmerged component and find the closest component on next slide.
    Merge them into a prototype (3d points body).
    Repeat this for the last merged component while can.
    In this way we extract all prototypes.
    After that we find connected prototypes and concat them.

    Main naming rules:
    - Component is a 2d connected component on some slide.
    - Prototype is a 3d points body of merged components.

    Parameters
    ----------
    data : np.ndarray or :class:`~.Geometry` instance
        A 3d array with smoothed predictions with shape corresponds to the field shape.
    direction : {0, 1}
        Prediction direction, can be 0 (ilines) or 1 (xlines).
    component_len_threshold : int
        Threshold to filter out too small connected components on data slides.
        If 0, then no filter applied (recommended for higher accuracy).
        If more than 0, then extraction will be faster.
    """
    def __init__(self, data, direction, component_len_threshold=0):
        """ Init data container with components info for each slide. """
        self.shape = data.shape

        self.direction = direction
        self.orthogonal_direction = 1 - self.direction

        self.dilation = 3 # constant for internal operations
        self.component_len_threshold = component_len_threshold

        self.container = self._init_container(data=data)
        self._first_slide_with_mergeable = 0 # variable for internal operations speed up

        self.prototypes_queue = deque() # prototypes for extension
        self.prototypes = [] # extracted prototypes

    def _init_container(self, data):
        """ Init data container with info about extracted connected components. """
        dilation_structure = np.ones((1, self.dilation), np.uint8)
        container = {}

        # Process data slides: extract connected components and their info
        for slide_idx in Notifier('t')(range(self.shape[self.direction])):
            # Get smoothed data slide
            smoothed = data.take(slide_idx, axis=self.direction)

            # Get skeletonized slide
            mask = skeletonize(smoothed, width=3)
            mask = dilate(mask, (1, 3))

            # Extract connected components from the slide
            labeled = connected_components(mask > 0)
            objects = find_objects(labeled)

            # Get components info
            coords, bboxes, lengths = [], [], []

            for idx, object_bbox in enumerate(objects, start=1):
                # Refined coords: we refine skeletonize effects by applying it on limited area
                dilation_axis = self.orthogonal_direction
                dilation_ranges = (max(0, object_bbox[0].start - self.dilation // 2),
                                   min(object_bbox[0].stop + self.dilation // 2, self.shape[dilation_axis]))

                object_mask = labeled[dilation_ranges[0]:dilation_ranges[1], object_bbox[-1]] == idx

                # Filter out too little components
                length = np.count_nonzero(object_mask)

                if length <= self.component_len_threshold:
                    continue

                # Extract refined component coordinates
                # skeletonize on the full slide is not so accurate than for limited area
                object_mask = dilate(object_mask.astype(np.uint8), dilation_structure)

                dilated_coords_2d = np.nonzero(object_mask)

                dilated_coords = np.zeros((len(dilated_coords_2d[0]), 3), dtype=np.int32)

                dilated_coords[:, self.direction] = slide_idx
                dilated_coords[:, dilation_axis] = dilated_coords_2d[0] + dilation_ranges[0]
                dilated_coords[:, 2] = dilated_coords_2d[1] + object_bbox[1].start

                smoothed_values = smoothed[dilated_coords[:, dilation_axis], dilated_coords[:, -1]]

                refined_coords = thin_coords(coords=dilated_coords, values=smoothed_values)

                # Filter out too little components
                # Previous length and len(refined_coords) are different
                # because original coords have more than one coord per depth when refined are not
                length = len(refined_coords)

                if length <= self.component_len_threshold:
                    continue

                lengths.append(length)
                coords.append(refined_coords)

                # Bbox
                bbox = np.empty((3, 2), np.int32)

                bbox[self.direction, :] = slide_idx
                bbox[self.orthogonal_direction, :] = dilation_ranges
                bbox[-1, :] = object_bbox[-1].start, object_bbox[-1].stop - 1

                bboxes.append(bbox)

            container[slide_idx] = {
                'coords': coords,
                'bboxes': bboxes,
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
            start_slide_idx, idx = self._find_not_merged_component()

            if idx is None: # No components to concat
                return None

            component = self.container[start_slide_idx]['coords'][idx]
            component_bbox = self.container[start_slide_idx]['bboxes'][idx]

            self.container[start_slide_idx]['lengths'][idx] = -1 # Mark this component as unmergeable
            prototype = FaultPrototype(coords=component, direction=self.direction,
                                       last_slide_idx=start_slide_idx, last_component=component,
                                       last_component_bbox=component_bbox)
        else:
            prototype = self.prototypes_queue.popleft()

            start_slide_idx = prototype.last_slide_idx
            component = prototype.last_component
            component_bbox = prototype.last_component_bbox

        # Find closest components on next slides
        for slide_idx_ in range(start_slide_idx+1, self.shape[self.direction]):
            # Find the closest component on the slide_idx_ to the current
            component, component_bbox, split_indices = self._find_closest_component(component=component,
                                                                                    component_bbox=component_bbox,
                                                                                    slide_idx=slide_idx_)

            # Postprocess prototype - it need to be splitted if it is out of component ranges
            if component is not None:
                prototype, new_prototypes = prototype.split(split_indices=split_indices, axis=2)
                self.prototypes_queue.extend(new_prototypes)

                prototype.append(component, bbox=component_bbox, slide_idx=slide_idx_)
            else:
                break

        return prototype

    def _find_not_merged_component(self):
        """ Find the longest not merged item on the minimal slide. """
        for slide_idx in range(self._first_slide_with_mergeable, self.shape[self.direction]):
            slide_info = self.container[slide_idx]

            if len(slide_info['lengths']) > 0:
                argmax = np.argmax(slide_info['lengths'])

                if slide_info['lengths'][argmax] != -1:
                    self._first_slide_with_mergeable = slide_idx
                    return slide_idx, argmax

        return None, None

    def _find_closest_component(self, component, component_bbox, slide_idx, distances_threshold=None,
                                depth_iteration_step=10, depths_threshold=20):
        """ Find the closest component to the provided one, get splitting indices for prototype if needed.

        Parameters
        ----------
        component : np.ndarray of (N, 3) shape
            Component coordinates for which to find the closest from the slide.
        component_bbox : np.ndarray of (3, 2) shape
            Component bounding box.
        slide_idx : int
            Index of the slide on which to find the closest component.
        distances_threshold : int, optional
            Threshold for the max possible axis-wise distance between components.
        depth_iteration_step : int
            The depth iteration step to find distances between components.
            Value 1 is recommended for higher accuracy.
            Value more than 1 can speed up the finding process.
        depths_threshold : int
            Depth-length threshold to decide to split closest component or prototype.
            If one component is longer than another more than on depths_threshold,
            then we need to split the longest one into parts:
             - one part corresponds to the closest component;
             - another correspond to the different component, which is not allowed for merge to the current prototype.
        """
        # Dilate component bbox for detecting close components: component on next slide can be shifted
        dilated_bbox = component_bbox[self.orthogonal_direction, :] + (-self.dilation // 2, self.dilation // 2)
        component_bbox[self.orthogonal_direction, 0] = max(0, dilated_bbox[0])
        component_bbox[self.orthogonal_direction, 1] = min(dilated_bbox[1], self.shape[self.orthogonal_direction])

        min_distance = distances_threshold if distances_threshold is not None else 100

        # Init returned values
        closest_component = None
        closest_component_bbox = None
        prototype_split_indices = [None, None]

        component_split_indices = [None, None]

        # Iter over components and find the closest one
        for idx, other_bbox in enumerate(self.container[slide_idx]['bboxes']):
            if self.container[slide_idx]['lengths'][idx] != -1:
                # Check bboxes intersection
                if not bboxes_intersected(component_bbox, other_bbox, axes=(self.orthogonal_direction, -1)):
                    continue

                # Check closeness of some points (iter over intersection depths with some step)
                # Faster then component intersection, but not so accurate
                other_component = self.container[slide_idx]['coords'][idx]

                intersection_depths = (max(component_bbox[-1, 0], other_bbox[-1, 0]),
                                       min(component_bbox[-1, 1], other_bbox[-1, 1]))

                step = min(depth_iteration_step, (intersection_depths[1]-intersection_depths[0])//3)
                step = max(step, 1)

                components_distances = min_max_depthwise_distances(component, other_component,
                                                                   depths_ranges=intersection_depths, step=step,
                                                                   axis=self.orthogonal_direction,
                                                                   max_threshold=min_distance)

                if (components_distances[0] is None) or (components_distances[0] > 1):
                    # Components are not close
                    continue

                if components_distances[1] < min_distance:
                    # The most depthwise distant points in components are close enough -> we can combine components
                    merged_idx = idx
                    min_distance = components_distances[1]
                    intersection_borders = intersection_depths
                    closest_component = other_component
                    closest_component_bbox = other_bbox

                    if min_distance == 0:
                        # The closest component is founded
                        break

        if closest_component is not None:
            # Process (split if needed) founded component and get split indices for prototype
            self.container[slide_idx]['lengths'][merged_idx] = -1 # mark component as unmergeable

            # Get prototype split indices: check that the new component is smaller than the previous one (for each border)
            if intersection_borders[0] - component_bbox[-1, 0] > depths_threshold:
                prototype_split_indices[0] = intersection_borders[0]

            if component_bbox[-1, 1] - intersection_borders[1] > depths_threshold:
                prototype_split_indices[1] = intersection_borders[1]

            # Split new component: check that the new component is bigger than the previous one (for each border)
            # Create splitted items and save them as new elements for merge
            if intersection_borders[0] - closest_component_bbox[-1, 0] > depths_threshold:
                component_split_indices[0] = intersection_borders[0]

            if closest_component_bbox[-1, 1] - intersection_borders[1] > depths_threshold:
                component_split_indices[1] = intersection_borders[1]

            closest_component, closest_component_bbox = self._split_component(component=closest_component,
                                                                              bbox=closest_component_bbox,
                                                                              split_indices=component_split_indices,
                                                                              slide_idx=slide_idx)

        return closest_component, closest_component_bbox, prototype_split_indices

    def _split_component(self, component, bbox, split_indices, slide_idx):
        """ Depth-wise prototypes split by indices.

        Parameters
        ----------
        component : np.ndarray of (N, 3) shape
            Component to split.
        bbox : np.ndarray of (3, 2) shape
            Component bbox.
        split_indices : sequence of two ints
            Depth values to split component into parts.
        slide_idx : int
            Component slide index.
        """
        # Cut upper part of the component and save it as another item
        if split_indices[0] is not None:
            splitted_component = component[component[:, -1] < split_indices[0]]

            splitted_component_bbox = bbox.copy()
            splitted_component_bbox[-1, 1] = max(0, split_indices[0] - 1)

            self._add_new_component(slide_idx=slide_idx, coords=splitted_component, bbox=splitted_component_bbox)

            # Extract suitable part
            component = component[component[:, -1] >= split_indices[0]]
            bbox[-1, 0] = split_indices[0]

        # Cut lower part of the component and save it as another item
        if split_indices[1] is not None:
            splitted_component = component[component[:, -1] > split_indices[1]]

            splitted_component_bbox = bbox.copy()
            splitted_component_bbox[-1, 0] = min(split_indices[1] + 1, self.shape[-1])

            self._add_new_component(slide_idx=slide_idx, coords=splitted_component, bbox=splitted_component_bbox)

            # Extract suitable part
            component = component[component[:, -1] <= split_indices[1]]
            bbox[-1, 1] = split_indices[1]

        return component, bbox

    def _add_new_component(self, slide_idx, coords, bbox):
        """ Add new item into the container.

        New items are creating after components splitting.

        Parameters
        ----------
        slide_idx : int
            Number of the slide on which we need to add new component.
        coords : np.ndarray of (N, 3) shape
            Component coordinates.
        bbox: np.ndarray of (3, 2) shape
            Component bounding box.
        """
        if len(coords) > self.component_len_threshold:
            self.container[slide_idx]['bboxes'].append(bbox)
            self.container[slide_idx]['coords'].append(coords)
            self.container[slide_idx]['lengths'].append(len(coords))


    def concat_connected_prototypes(self, intersection_ratio_threshold=None, axis=2,
                                    border_threshold=50, width_split_threshold=100):
        """ Concat prototypes which are connected.

        Under the hood we compare prototypes with each other and find which are connected as puzzles.
        For this we get neighboring borders and compare them.
        If borders are similar enough then we merge prototypes into one.

        Parameters
        ----------
        intersection_ratio_threshold : float or None
            Prototypes contours intersection ratio to decide that prototypes are not close.
            Possible values are float numbers in the (0, 1] interval or None.
            If None, then default values are used: 0.5 for the depth axis and 0.9 for others.
        axis : {0, 1, 2}
            Axis along which to find prototypes connections.
            Recommended values are 2 (for depths) or self.direction.
        border_threshold : int
            Minimal amount of different border contour points to decide that prototypes are not close.
        width_split_threshold : int or None
            Contour widths (along self.direction axis) difference threshold to decide that prototypes needn't
            to be splitted by self.direction axis.
            If value is None, then no splitting applied. But there are the risk of interpenetration of
            triangulated surfaces in this case.
            With lower value more splitting applied and smaller prototypes are extracted.
            With higher value there are less prototypes with higher areas extracted.
            So, if you want more detailing, then provide smaller width_split_threshold (near to 10).
            If you want to extract bigger surfaces, then provide higher width_split_threshold (near to 100 or None).
        """
        margin = 1 # local constant for code prettifying

        if intersection_ratio_threshold is None:
            intersection_ratio_threshold = 0.5 if axis in (2, -1) else 0.9

        overlap_axis = self.direction if axis in (-1, 2) else 2
        split_axis = overlap_axis

        # Under the hood, we check borders connectivity (as puzzles)
        borders_to_check = ('up', 'down') if axis in (-1, 2) else ('left', 'right')

        # Presort objects by other valuable axis for early stopping
        sort_axis = overlap_axis
        prototypes_starts = np.array([prototype.bbox[sort_axis, 0] for prototype in self.prototypes])
        prototypes_order = np.argsort(prototypes_starts)

        new_prototypes = []

        for i, prototype_1_idx in enumerate(prototypes_order):
            prototype_1 = self.prototypes[prototype_1_idx]

            for prototype_2_idx in prototypes_order[i+1:]:
                prototype_2 = self.prototypes[prototype_2_idx]

                # Exit if we out of sort_axis ranges for prototype_1
                if (prototype_1.bbox[sort_axis, 1] < prototype_2.bbox[sort_axis, 0]):
                    break

                adjacent_borders = bboxes_adjacent(prototype_1.bbox, prototype_2.bbox)

                if adjacent_borders is None:
                    continue

                # Check that bboxes overlap is enough
                intersection_threshold = min(prototype_1.bbox[overlap_axis, 1]-prototype_1.bbox[overlap_axis, 0],
                                             prototype_2.bbox[overlap_axis, 1]-prototype_2.bbox[overlap_axis, 0])
                intersection_threshold *= intersection_ratio_threshold

                intersection_length = adjacent_borders[overlap_axis][1] - adjacent_borders[overlap_axis][0]

                if intersection_length < intersection_threshold:
                    continue

                # Find object contours on close borders
                is_first_upper = prototype_1.bbox[axis, 0] < prototype_2.bbox[axis, 0]

                contour_1 = prototype_1.get_border(border=borders_to_check[is_first_upper],
                                                    projection_axis=self.orthogonal_direction)
                contour_2 = prototype_2.get_border(border=borders_to_check[~is_first_upper],
                                                    projection_axis=self.orthogonal_direction)

                # Get border contours in the area of interest
                intersection_range = (min(adjacent_borders[axis]) - margin, max(adjacent_borders[axis]) + margin)

                contour_1 = contour_1[(contour_1[:, axis] >= intersection_range[0]) & \
                                      (contour_1[:, axis] <= intersection_range[1])]
                contour_2 = contour_2[(contour_2[:, axis] >= intersection_range[0]) & \
                                      (contour_2[:, axis] <= intersection_range[1])]

                # If one data contour is much longer than other, then we can't connect them as puzzle details
                if len(contour_1) == 0 or len(contour_2) == 0:
                    continue

                length_ratio = min(len(contour_1), len(contour_2)) / max(len(contour_1), len(contour_2))

                if length_ratio < intersection_ratio_threshold:
                    continue

                # Correct border_threshold for too short borders
                max_n_points_out_intersection = (1 - intersection_ratio_threshold) * min(len(contour_1), len(contour_2))
                corrected_border_threshold = min(border_threshold, max_n_points_out_intersection)

                # Shift one of the objects, making their contours intersected
                shift = 1 if is_first_upper else -1
                contour_1[:, axis] += shift

                # Check that one component contour is inside another (for both)
                if self._is_contour_inside(contour_1, contour_2, border_threshold=corrected_border_threshold) or \
                   self._is_contour_inside(contour_2, contour_1, border_threshold=corrected_border_threshold):
                    # Split by split_axis for avoiding wrong prototypes shapes (like C or T-likable, etc.)
                    if (axis in (-1, 2)) and (width_split_threshold is not None) and \
                       (np.abs(prototype_1.width - prototype_2.width) > width_split_threshold):
                        split_indices = (max(prototype_1.bbox[split_axis, 0], prototype_2.bbox[split_axis, 0]),
                                         min(prototype_1.bbox[split_axis, 1], prototype_2.bbox[split_axis, 1]))

                        prototype_1, new_prototypes_ = prototype_1.split(split_indices, axis=split_axis)
                        new_prototypes.extend(new_prototypes_)

                        if len(new_prototypes_) > 0:
                            prototype_1 = FaultPrototype(prototype_1.coords, direction=split_axis)

                        prototype_2, new_prototypes_ = prototype_2.split(split_indices, axis=split_axis)
                        new_prototypes.extend(new_prototypes_)

                        if len(new_prototypes_) > 0:
                            prototype_2 = FaultPrototype(prototype_2.coords, direction=split_axis)

                    prototype_2.concat(prototype_1)
                    self.prototypes[prototype_2_idx] = prototype_2
                    self.prototypes[prototype_1_idx] = None
                    break

        self.prototypes = [prototype for prototype in self.prototypes if prototype is not None]
        self.prototypes.extend(new_prototypes)
        return self.prototypes

    def _is_contour_inside(self, contour_1, contour_2, border_threshold):
        """ Check that `contour_1` is almost inside dilated `contour_2`.

        We apply dilation for contour_2 because the fault can be a little shifted on other depths.

        Parameters
        ----------
        contour_1, contour_2 : np.ndarrays of (N, 3) shape
            Contours for check.
        border_threshold : int
            Minimal amount of different points to decide that contours are not close.
        """
        contour_1_set = set(tuple(x) for x in contour_1)

        # Objects can be shifted on `self.orthogonal_direction`, so apply dilation for coords
        contour_2_dilated = dilate_coords(coords=contour_2, dilate=self.dilation,
                                          axis=self.orthogonal_direction,
                                          max_value=self.shape[self.orthogonal_direction]-1)

        contour_2_dilated = set(tuple(x) for x in contour_2_dilated)

        return len(contour_1_set - contour_2_dilated) < border_threshold

    def concat_embedded_prototypes(self, border_threshold=100):
        """ Concat embedded prototypes with 2 or more closed borders.

        Under the hood, we compare different prototypes to find pairs in which one prototype is inside
        another prototype area.
        If more than two borders of internal prototype is connected with other prototype,
        then we merge these prototypes into one.

        Internal logic looks similar to `.concat_connected_prototypes`,
        but now we find embedded bboxes and need two borders coincidence.

        Embedded prototypes examples:

        ||||||  or  |||||||  or  ||||||  etc.
        ...|||      |...|||      |||...
           |||      ||||||
        ||||||

         - where | means one prototype points, and . - other prototype points.

        Parameters
        ----------
        border_threshold : int
            Minimal amount of different points to decide that contours are not close.
        """
        # Presort objects by other valuable axis for early stopping
        sort_axis = self.direction
        prototypes_starts = np.array([prototype.bbox[sort_axis, 0] for prototype in self.prototypes])
        prototypes_order = np.argsort(prototypes_starts)

        margin = 3 # local constant

        for i, prototype_1_idx in enumerate(prototypes_order):
            prototype_1 = self.prototypes[prototype_1_idx]

            for prototype_2_idx in prototypes_order[i+1:]:
                prototype_2 = self.prototypes[prototype_2_idx]
                # Check that prototypes are embedded
                if (prototype_1.bbox[sort_axis, 1] < prototype_2.bbox[sort_axis, 0]):
                    break

                is_embedded, is_second_inside_first = bboxes_embedded(prototype_1.bbox, prototype_2.bbox, margin=margin)

                if not is_embedded:
                    continue

                # Find the main object and internal one (prototype_to_check)
                main_object_coords = prototype_1.coords if is_second_inside_first else prototype_2.coords
                prototype_to_check = prototype_2 if is_second_inside_first else prototype_1

                # Check borders connections
                close_borders_counter = 0

                for border in ('up', 'down', 'left', 'right'): # TODO: get more optimal order depend on bboxes
                    # Find internal object border contour
                    contour = prototype_to_check.get_border(border=border, projection_axis=self.orthogonal_direction)

                    # Shift contour to make it intersected with another object
                    shift = -1 if border in ('up', 'left') else 1
                    shift_axis = self.direction if border in ('left', 'right') else 2
                    contour[:, shift_axis] += shift

                    # Get main object coords in the area of the interest for speeding up evaluations
                    slices = prototype_to_check.bbox.copy()
                    slices[:, 0] -= margin
                    slices[:, 1] += margin

                    main_object_coords_sliced = main_object_coords[(main_object_coords[:, 0] >= slices[0, 0]) & \
                                                                   (main_object_coords[:, 0] <= slices[0, 1]) & \
                                                                   (main_object_coords[:, 1] >= slices[1, 0]) & \
                                                                   (main_object_coords[:, 1] <= slices[1, 1]) & \
                                                                   (main_object_coords[:, 2] >= slices[2, 0]) & \
                                                                   (main_object_coords[:, 2] <= slices[2, 1])]

                    # Check that the shifted border is inside the main_object area
                    corrected_border_threshold = min(border_threshold, len(contour)//2)

                    if self._is_contour_inside(contour, main_object_coords_sliced,
                                               border_threshold=corrected_border_threshold):
                        close_borders_counter += 1

                    if close_borders_counter >= 2:
                        break

                # If objects have more than 2 closed borders then they are parts of the same prototype -> merge them
                if close_borders_counter >= 2:
                    prototype_2.concat(prototype_1)
                    self.prototypes[prototype_2_idx] = prototype_2
                    self.prototypes[prototype_1_idx] = None
                    break

        self.prototypes = [prototype for prototype in self.prototypes if prototype is not None]
        return self.prototypes

    # Addons
    def run(self, concat_iters=20, intersection_ratio_threshold=None,
            prolongate_in_depth=False, additional_filters=False, **filtering_kwargs):
        """ Recommended full extracting procedure.

        The procedure scheme is:
            - extract prototypes from point cloud;
            - filter too small prototypes (for speed up);
            - concat connected prototypes (concat by depth axis, concat by self.direction axis) concat_iters times
            with changed `intersection_ratio_threshold`;
            - filter too small prototypes (for speed up);
            - concat embedded prototypes;
            - filter all unsuitable prototypes

        Parameters
        ----------
        concat_iters : int
            Maximal amount of connected component concatenation operations which are include concat along
            the depth and `self.direction` axes.
        intersection_ratio_threshold : dict or None
            Prototype borders intersection ratio to decide that prototypes can be connected.
            Note, it is decrementally changed. Keys are axes and values are in the (start, stop, step) format.
        prolongate_in_depth : bool
            Whether to maximally prolongate faults in depth or not.
            If True, then surfaces will be tall and thin.
            If False, then surfaces will be more longer for `self.direction` than for depth axis.
        additional_filters : bool
            Whether to apply additional filtering for speed up.
        min_intersection_ratio_threshold : float
            Minimal preferred value of `intersection_ratio_threshold`.
        filtering_kwargs
            kwargs for the `.filter_prototypes` method.
            This kwargs are applied for filtration after full extraction procedure.

        Returns
        -------
        prototypes: list of the FaultPrototype instances
            Resulting prototypes.
        stats : dict
            Amount of prototypes after each proceeding. Helpful for debug.
        """
        stats = {}

        if intersection_ratio_threshold is None:
            intersection_ratio_threshold = {
                self.direction: (0.9, 0.7, 0.05), # (start, stop, step)
                2: (0.9, 0.4, 0.05)
            }

        depth_intersection_threshold = intersection_ratio_threshold[2][0]
        direction_intersection_threshold = intersection_ratio_threshold[self.direction][0]

        # Extract prototypes from data
        _ = self.extract_prototypes()
        stats['extracted'] = len(self.prototypes)

        # Filter for speed up
        if additional_filters:
            self.prototypes = self.filter_prototypes(min_height=3, min_width=3, min_n_points=10)
            stats['filtered_extracted'] = len(self.prototypes)

        # Concat connected (as puzzles) prototypes
        previous_iter_prototypes_amount = stats['extracted'] + 100 # to avoid stopping after first concat

        stats['after_connected_concat'] = {}

        for i in Notifier('t')(concat_iters):
            stats['after_connected_concat'][i] = []
            # Concat by depth axis
            _ = self.concat_connected_prototypes(intersection_ratio_threshold=depth_intersection_threshold,
                                                 axis=-1)
            stats['after_connected_concat'][i].append(len(self.prototypes))

            # Concat by direction axis
            if prolongate_in_depth and (depth_intersection_threshold <= intersection_ratio_threshold[2][1]):
                _ = self.concat_connected_prototypes(intersection_ratio_threshold=direction_intersection_threshold,
                                                    axis=self.direction)

                stats['after_connected_concat'][i].append(len(self.prototypes))

            # Early stopping
            if (depth_intersection_threshold <= intersection_ratio_threshold[2][1]) and \
               (direction_intersection_threshold <= intersection_ratio_threshold[self.direction][1]) and \
               (stats['after_connected_concat'][i][-1] == previous_iter_prototypes_amount):
                break

            previous_iter_prototypes_amount = stats['after_connected_concat'][i][-1]

            depth_intersection_threshold = round(depth_intersection_threshold - intersection_ratio_threshold[2][-1], 2)
            depth_intersection_threshold = max(depth_intersection_threshold, intersection_ratio_threshold[2][1])

            if prolongate_in_depth and (depth_intersection_threshold <= intersection_ratio_threshold[2][1]):
                direction_intersection_threshold = round(direction_intersection_threshold - \
                                                            intersection_ratio_threshold[self.direction][-1], 2)
                direction_intersection_threshold = max(direction_intersection_threshold,
                                                        intersection_ratio_threshold[self.direction][1])

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


class FaultPrototype:
    """ Class for faults prototypes. Provides a necessary API for convenient prototype extraction process. """
    def __init__(self, coords, direction, last_slide_idx=None, last_component=None, last_component_bbox=None):
        """ Fault prototype initialization.

        Note, last_* parameters are preferred for prototype extraction and are optional:
        they are used for finding closest components on next slides.

        Parameters
        ----------
        coords : np.ndarray of (N, 3) shape
            Prototype coordinates.
        direction : {0, 1}
            Direction along which the prototype is extracted (the same as prediction direction).
        last_slide_idx : int, optional
            Index of the slide from which the last added component was extracted.
        last_component : np.ndarray of (N, 3) shape
            Coordinates of the last added component.
        last_component_bbox : np.ndarray of (3, 2) shape
            Bounding box of the `last_component`.
        """
        self.coords = coords
        self.direction = direction

        self._bbox = None

        self._last_slide_idx = last_slide_idx
        self._last_component = last_component
        self._last_component_bbox = last_component_bbox

        self._contour = None
        self._borders = {}

    @property
    def bbox(self):
        """ Bounding box. """
        if self._bbox is None:
            self._bbox = np.column_stack([np.min(self.coords, axis=0), np.max(self.coords, axis=0)])
        return self._bbox

    # Stats for filtering
    @property
    def height(self):
        """ Height (length along the depth axis). """
        return self.bbox[-1, 1] - self.bbox[-1, 0]

    @property
    def width(self):
        """ Width (length along the `self.direction` axis). """
        return self.bbox[self.direction, 1] - self.bbox[self.direction, 0]

    @property
    def n_points(self):
        """ Amount of the surface points. """
        return len(self.coords)

    # For internal needs
    @property
    def last_slide_idx(self):
        """ Index of the slide from which the last added component was extracted. """
        if self._last_slide_idx is None:
            self._last_slide_idx = self.coords[:, self.direction].max()
        return self._last_slide_idx

    @property
    def last_component(self):
        """ Coordinates (N, 3) of the last added component. """
        if self._last_component is None:
            self._last_component = self.coords[self.coords[:, self.direction] == self.last_slide_idx]
        return self._last_component

    @property
    def last_component_bbox(self):
        """ Bounding box of the `self.last_component`. """
        if self._last_component_bbox is None:
            self._last_component_bbox = np.column_stack([np.min(self.last_component, axis=0),
                                                         np.max(self.last_component, axis=0)])
        return self._last_component_bbox

    @property
    def contour(self):
        """ Contour of 2d projection on axis, orthogonal to self.direction.

        Note, output coordinates, which corresponding to the projection axis are zeros.
        """
        if self._contour is None:
            projection_axis = 1 - self.direction
            self._contour = find_contour(coords=self.coords, projection_axis=projection_axis)
        return self._contour

    def append(self, coords, bbox, slide_idx=None):
        """ Append new component into prototype.

        Parameters
        ----------
        coords : np.ndarray of (N, 3) shape
            Coordinates of the component to add into the prototype.
        bbox : np.ndarray of (3, 2) shape
            Bounding box of the added component.
        slide_idx : int or None
            The added component slide index.
        """
        self.coords = np.vstack([self.coords, coords])

        self._contour = None
        self._borders = {}

        self._last_slide_idx = slide_idx
        self._last_component = coords
        self._last_component_bbox = bbox

        self._bbox = self._concat_bbox(self._last_component_bbox)

    def concat(self, other):
        """ Concatenate two prototypes. """
        self.coords = np.vstack([self.coords, other.coords])

        self._bbox = self._concat_bbox(other.bbox)

        self._contour = None
        self._borders = {}

        self._last_slide_idx = None
        self._last_component = None
        self._last_component_bbox = None

    def _concat_bbox(self, other_bbox):
        """ Concat bboxes of two objects into one. """
        bbox = np.empty((3, 2), np.int32)
        bbox[:, 0] = np.min((self.bbox[:, 0], other_bbox[:, 0]), axis=0)
        bbox[:, 1] = np.max((self.bbox[:, 1], other_bbox[:, 1]), axis=0)
        return bbox

    def _separate_objects(self, coords, axis):
        """ Separate coords into different object coords depend on their connectedness by axis.

        After split we can have the situation when splitted part has more than one connected component.
        This method separate disconnected parts into different prototypes.
        """
        # Get coordinates along the axis
        unique_direction_coords = np.unique(coords[:, axis])

        # Slides distance more than 1 -> different objects
        split_indices = np.nonzero(unique_direction_coords[1:] - unique_direction_coords[:-1] > 1)[0]

        if len(split_indices) == 0:
            return [FaultPrototype(coords=coords, direction=self.direction)]

        # Separate disconnected objects and create new prototypes instances
        start_indices = unique_direction_coords[split_indices + 1]
        start_indices = np.insert(start_indices, 0, 0)

        end_indices = unique_direction_coords[split_indices]
        end_indices = np.append(end_indices, unique_direction_coords[-1])

        prototypes = []

        for start_idx, end_idx in zip(start_indices, end_indices):
            coords_ = coords[(start_idx <= coords[:, axis]) & (coords[:, axis] <= end_idx)]
            prototype = FaultPrototype(coords=coords_, direction=self.direction, last_slide_idx=end_idx)
            prototypes.append(prototype)

        return prototypes

    def split(self, split_indices, axis=-1):
        """ Axis-wise prototypes split by indices.

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
           (np.min(self.coords[:, axis]) < split_indices[0] < np.max(self.coords[:, axis])):

            coords_outer = self.coords[self.coords[:, axis] < split_indices[0]]
            self.coords = self.coords[self.coords[:, axis] >= split_indices[0]]

            if len(coords_outer) > 0:
                new_prototypes.extend(self._separate_objects(coords_outer, axis=axis_for_objects_separating))

        # Cut lower part and separate disconnected objects
        if (split_indices[1] is not None) and \
           (np.min(self.coords[:, axis]) < split_indices[1] < np.max(self.coords[:, axis])):

            coords_outer = self.coords[self.coords[:, axis] > split_indices[1]]
            self.coords = self.coords[self.coords[:, axis] <= split_indices[1]]

            if len(coords_outer) > 0:
                new_prototypes.extend(self._separate_objects(coords_outer, axis=axis_for_objects_separating))

        new_prototypes.extend(self._separate_objects(self.coords, axis=axis_for_objects_separating))
        return new_prototypes[-1], new_prototypes[:-1]

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
            Coordinates of the requested border.
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
