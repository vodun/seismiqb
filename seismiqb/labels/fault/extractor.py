""" [Draft] Extractor of fault surfaces from cloud of points. """
from collections import defaultdict, deque
import numpy as np

from cc3d import connected_components
from cv2 import dilate
from scipy.ndimage import find_objects
from sklearn.neighbors import KDTree

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
    """
    def __init__(self, smoothed_array, skeletonized_array=None, direction=0, component_len_threshold=0):
        """ Init data container with components info for each slide.

        ..!!..

        Parameters
        ----------
        direction : {0, 1}
            Prediction direction corresponding ilines (0) or xlines (1).
        component_len_threshold : int
            Threshold to filter out too small connected components on data slides.
        """
        self.shape = smoothed_array.shape

        self.direction = direction
        self.orthogonal_direction = 1 - self.direction

        self.component_len_threshold = component_len_threshold # TODO: temporally unused, change value
        # self.height_threshold = None # TODO: temporally unused, add later

        self.dilation = 3 # constant for internal operations
        dilation_structure = np.ones((1, self.dilation), np.uint8)

        self.container = {}

        self.prototypes_queue = deque() # prototypes for extension
        self.prototypes = [] # extracted prototypes

        for slide_idx in Notifier('t')(range(self.shape[self.direction])):
            # Process data
            smoothed = np.take(smoothed_array, slide_idx, axis=self.direction)

            if skeletonized_array is not None:
                mask = np.take(skeletonized_array, slide_idx, axis=self.direction)
            else:
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

                if length <= component_len_threshold:
                    continue

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

                if length <= component_len_threshold:
                    continue

                lengths.append(length)
                coords.append(refined_coords)

                # Bbox
                bbox = np.empty((3, 2), np.int32)

                bbox[self.direction, :] = slide_idx
                bbox[self.orthogonal_direction, :] = dilation_ranges
                bbox[-1, :] = object_bbox[-1].start, object_bbox[-1].stop - 1

                bboxes.append(bbox)

            self.container[slide_idx] = {
                'coords': coords,
                'bboxes': bboxes,
                'lengths': lengths
            }

        self._first_slide_with_mergeable = 0

    # Prototypes extraction
    def extract_prototypes(self):
        """ Extract all fault prototypes from the point cloud. """
        prototype = self.extract_prototype()

        while prototype is not None:
            self.prototypes.append(prototype)
            prototype = self.extract_prototype()

    def extract_prototype(self):
        """ Extract one fault prototype from the point cloud. """
        idx = -1

        if len(self.prototypes_queue) == 0:
            start_slide_idx, idx = self._find_not_merged_component()

            if idx is None: # No components to concat
                return None

            component = self.container[start_slide_idx]['coords'][idx]
            component_bbox = self.container[start_slide_idx]['bboxes'][idx]

            self.container[start_slide_idx]['lengths'][idx] = -1 # Mark this component as unmergeable
            prototype = FaultPrototype(coords=component, direction=self.direction,
                                       last_slide_idx=start_slide_idx, last_component=component)
        else:
            prototype = self.prototypes_queue.popleft()

            start_slide_idx = prototype.last_slide_idx
            component = prototype.last_component
            component_bbox = None

        # Find closest components on next slides and split them if needed
        for slide_idx_ in range(start_slide_idx+1, self.shape[self.direction]):
            # Find the closest component on the slide_idx_ to the current
            component, component_bbox, split_indices = self._find_closest_component(component=component,
                                                                                    component_bbox=component_bbox,
                                                                                    slide_idx=slide_idx_)

            # Postprocess prototype
            if component is not None:
                # Split current prototype and add new to queue
                prototype, new_prototypes = prototype.split(split_indices=split_indices)
                self.prototypes_queue.extend(new_prototypes)

                prototype.append(component, slide_idx=slide_idx_)
            else:
                break

        return prototype

    def _find_not_merged_component(self):
        """ Find the longest not merged item on the minimal slide. """
        for slide_idx in range(self._first_slide_with_mergeable, self.shape[self.direction]):
            slide_info = self.container[slide_idx]
            argmax = np.argmax(slide_info['lengths'])

            if slide_info['lengths'][argmax] != -1:
                self._first_slide_with_mergeable = slide_idx
                return slide_idx, argmax

        return None, None

    def _find_closest_component(self, component, component_bbox, slide_idx, distances_threshold=5,
                                depth_iteration_step=10, depths_threshold=5):
        """ Find the closest component to component on the slide, get splitting depths for them if needed.

        ..!!..
        """
        # Process inputs
        # Dilate component bbox for detecting close components: component on next slide can be shifted
        if component_bbox is None:
            component_bbox = np.column_stack([np.min(component, axis=0), np.max(component, axis=0)])

        component_bbox[self.orthogonal_direction] += (-self.dilation // 2, self.dilation // 2)

        min_distance = distances_threshold

        # Init returned values
        closest_component = None
        closest_component_bbox = None
        prototype_split_indices = [None, None]

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

                step = np.clip((intersection_depths[1]-intersection_depths[0])//3, 1, depth_iteration_step)
                # step = 1 TODO: check accuracy

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
            # Merge founded component and split prototype and new component extra parts
            self.container[slide_idx]['lengths'][merged_idx] = -1

            # Split prototype: check that the new component is smaller than the previous one (for each border)
            if intersection_borders[0] - component_bbox[-1, 0] > depths_threshold:
                prototype_split_indices[0] = intersection_borders[0]

            if component_bbox[-1, 1] - intersection_borders[1] > depths_threshold:
                prototype_split_indices[1] = intersection_borders[1]

            # Split new component: check that the new component is bigger than the previous one (for each border)
            # Create splitted items and save them as new elements for merge
            if intersection_borders[0] - closest_component_bbox[-1, 0] > depths_threshold:
                item_split_idx = intersection_borders[0]

                # Cut upper part of component on next slide and save extra data as another item
                new_component = closest_component[closest_component[:, -1] < item_split_idx]

                new_component_bbox = closest_component_bbox.copy()
                new_component_bbox[-1, 1] = item_split_idx - 1

                self._add_new_component(slide_idx=slide_idx, coords=new_component, bbox=new_component_bbox)

                # Extract suitable part
                closest_component = closest_component[closest_component[:, -1] >= item_split_idx]
                closest_component_bbox[-1, 0] = item_split_idx

            if closest_component_bbox[-1, 1] - intersection_borders[1] > depths_threshold:
                item_split_idx = intersection_borders[1]

                # Cut lower part of component on next slide and save extra data as another item
                new_component = closest_component[closest_component[:, -1] > item_split_idx]

                new_component_bbox = closest_component_bbox.copy()
                new_component_bbox[-1, 0] = item_split_idx + 1

                self._add_new_component(slide_idx=slide_idx, coords=new_component, bbox=new_component_bbox)

                # Extract suitable part
                closest_component = closest_component[closest_component[:, -1] <= item_split_idx]
                closest_component_bbox[-1, 1] = item_split_idx

        return closest_component, closest_component_bbox, prototype_split_indices

    def _add_new_component(self, slide_idx, coords, bbox):
        """ Add new items into the container. """
        if len(coords) > self.component_len_threshold:
            self.container[slide_idx]['bboxes'].append(bbox)
            self.container[slide_idx]['coords'].append(coords)
            self.container[slide_idx]['lengths'].append(len(coords))

    # Prototypes concatenation
    def concat_prototypes(self, type='connected', **kwargs):
        # TODO: make to_concat argument, and call find_*_prototypes methods outside (later)
        """ Find and concat prototypes connected on the `axis`.

        ..!!..
        """
        # Concat coords and remove concated parts
        if type == 'connected':
            to_concat = self.find_connected_prototypes(**kwargs)
        else:
            # TODO: improve timings for `self.find_embedded_prototypes``
            to_concat = self.find_embedded_prototypes()

        remove_elements = []

        for where_to_concat_idx, what_to_concat_indices in to_concat.items():
            main_prototype = self.prototypes[where_to_concat_idx]

            for idx in what_to_concat_indices:
                main_prototype.concat(self.prototypes[idx])

            remove_elements.extend(what_to_concat_indices)

        remove_elements.sort(reverse=True)

        for idx in remove_elements:
            _ = self.prototypes.pop(idx)

    def find_connected_prototypes(self, intersection_ratio_threshold=None, contour_threshold=10, axis=2):
        """ Find prototypes which are connected as puzzles.

        ..!!..

        Parameters
        ----------
        intersection_ratio_threshold : float
            Prototypes contours intersection ratio to decide that prototypes are not close.
        contour_threshold : int
            Amount of different contour points to decide that prototypes are not close.
        """
        margin = 1 # local constant for code prettifying

        if intersection_ratio_threshold is None:
            intersection_ratio_threshold = 0.5 if axis in (2, -1) else 0.9

        to_concat = defaultdict(list) # owner -> items
        concated_with = {} # item -> owner

        overlapping_axis = self.direction if axis in (-1, 2) else 2

        # Under the hood, we check borders connectivity (as puzzles)
        borders_to_check = ('up', 'down') if axis in (-1, 2) else ('left', 'right')

        # Presort objects by other valuable axis for early stopping
        sort_axis = 2 if axis == self.direction else self.direction
        prototypes_starts = np.array([prototype.bbox[sort_axis, 0] for prototype in self.prototypes])
        prototypes_order = np.argsort(prototypes_starts)

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
                intersection_threshold = min(prototype_1.bbox[overlapping_axis, 1]-prototype_1.bbox[overlapping_axis, 0],
                                             prototype_2.bbox[overlapping_axis, 1]-prototype_2.bbox[overlapping_axis, 0])
                intersection_threshold *= intersection_ratio_threshold

                intersection_length = adjacent_borders[overlapping_axis][1] - adjacent_borders[overlapping_axis][0]

                if intersection_length < intersection_threshold:
                    continue

                # Find object contours on close borders
                is_first_upper = prototype_1.bbox[axis, 0] < prototype_2.bbox[axis, 0]

                contour_1 = prototype_1.get_borders(border=borders_to_check[is_first_upper],
                                                    projection_axis=self.orthogonal_direction)
                contour_2 = prototype_2.get_borders(border=borders_to_check[~is_first_upper],
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

                # Process objects with too small border contours
                if (len(contour_1) < 4*contour_threshold) or (len(contour_2) < 4*contour_threshold):
                    corrected_contour_threshold = 1
                else:
                    corrected_contour_threshold = contour_threshold

                # Shift one of the objects, making their contours intersected
                shift = 1 if is_first_upper else -1
                contour_1[:, -1] += shift

                # Check that one component contour is inside another (for both)
                if self._is_contour_inside(contour_1, contour_2, contour_threshold=corrected_contour_threshold):
                    to_concat, concated_with = _add_link(item_i=prototype_1_idx, item_j=prototype_2_idx,
                                                         to_concat=to_concat, concated_with=concated_with)
                    continue

                if self._is_contour_inside(contour_2, contour_1, contour_threshold=corrected_contour_threshold):
                    to_concat, concated_with = _add_link(item_i=prototype_1_idx, item_j=prototype_2_idx,
                                                         to_concat=to_concat, concated_with=concated_with)

        return to_concat

    def _is_contour_inside(self, contour_1, contour_2, contour_threshold):
        """ Check that `contour_1` is almost inside dilated `contour_2`. """
        contour_1_set = set(tuple(x) for x in contour_1)

        # Objects can be shifted on `self.orthogonal_direction`, so apply dilation for coords
        contour_2_dilated = dilate_coords(coords=contour_2, dilate=self.dilation,
                                          axis=self.orthogonal_direction,
                                          max_value=self.shape[self.orthogonal_direction]-1)

        contour_2_dilated = set(tuple(x) for x in contour_2_dilated)

        return len(contour_1_set - contour_2_dilated) < contour_threshold

    def find_embedded_prototypes(self, distances_threshold=2):
        """ Find embedded prototypes (with 2 or more closed borders.

        Examples
        --------

        ||||||
        ...|||
        ||||||

        ||||||
        ...|||

        ||||||
        ...|||
           |||
        ||||||

        where | means one prototype points, and . - other prototype points
        """
        to_concat = defaultdict(list) # owner -> items
        concated_with = {} # item -> owner

        # Presort objects by other valuable axis for early stopping
        sort_axis = self.direction
        prototypes_starts = np.array([prototype.bbox[sort_axis, 0] for prototype in self.prototypes])
        prototypes_order = np.argsort(prototypes_starts)

        for i, prototype_1_idx in enumerate(prototypes_order):
            prototype_1 = self.prototypes[prototype_1_idx]

            for prototype_2_idx in prototypes_order[i+1:]:
                prototype_2 = self.prototypes[prototype_2_idx]

                if (prototype_1.bbox[sort_axis, 1] < prototype_2.bbox[sort_axis, 0]):
                    break

                is_embedded, is_second_inside_first = bboxes_embedded(prototype_1.bbox, prototype_2.bbox)

                if not is_embedded:
                    continue

                tree = KDTree(prototype_1.coords) if is_second_inside_first else KDTree(prototype_2.coords)
                prototype_to_check = prototype_2 if is_second_inside_first else prototype_1

                close_borders_counter = 0

                for border in ('up', 'down', 'left', 'right'):
                    contour_ = prototype_to_check.get_borders(border=border, projection_axis=self.orthogonal_direction)
                    distances, _ = tree.query(contour_)

                    if np.percentile(distances, 90) < distances_threshold:
                        close_borders_counter += 1

                    if close_borders_counter >= 2:
                        break

                if close_borders_counter >= 2:
                    to_concat, concated_with = _add_link(item_i=prototype_1_idx, item_j=prototype_2_idx,
                                                         to_concat=to_concat, concated_with=concated_with)
        return to_concat

    # Addons
    def run_prototypes_concat(self, iters=5):
        """ Only for tests. Will be removed. """
        previous_prototypes_amount = len(self.prototypes) + 100 # to avoid stopping after first concat
        print("Start amount: ", len(self.prototypes))

        for _ in range(iters):
            self.concat_prototypes(type='connected', axis=-1)

            print("After depths concat: ", len(self.prototypes))

            if len(self.prototypes) < previous_prototypes_amount:
                previous_prototypes_amount = len(self.prototypes)
            else:
                break

            self.concat_prototypes(type='connected', axis=self.direction)

            print("After ilines concat: ", len(self.prototypes))

            if len(self.prototypes) < previous_prototypes_amount:
                previous_prototypes_amount = len(self.prototypes)
            else:
                break

    def prototypes_to_faults(self, field):
        """ Convert all prototypes to faults. """
        faults = [Fault(prototype.coords, field=field) for prototype in self.prototypes]
        return faults


class FaultPrototype:
    """ ..!!.. """
    def __init__(self, coords, direction, last_slide_idx=None, last_component=None):
        """..!!.."""
        self.coords = coords
        self.direction = direction

        self._bbox = None
        self._last_slide_idx = last_slide_idx
        self._last_component = last_component

        self._contour = None
        self._borders = {}

    @property
    def bbox(self):
        """..!!.."""
        if self._bbox is None:
            self._bbox = np.column_stack([np.min(self.coords, axis=0), np.max(self.coords, axis=0)])
        return self._bbox

    @property
    def last_slide_idx(self):
        if self._last_slide_idx is None:
            self._last_slide_idx = self.coords[:, self.direction].max()
        return self._last_slide_idx

    @property
    def last_component(self):
        if self._last_component is None:
            self._last_component = self.coords[self.coords[:, self.direction] == self.last_slide_idx]
        return self._last_component

    @property
    def contour(self):
        """ Contour of 2d projection on axis, orthogonal to self.direction."""
        if self._contour is None:
            projection_axis = 1 - self.direction
            self._contour = find_contour(coords=self.coords, projection_axis=projection_axis)
        return self._contour

    def append(self, coords, slide_idx=None):
        """ Append new component (coords) into prototype. """
        self.coords = np.vstack([self.coords, coords])

        self._bbox = None
        self._contour = None
        self._borders = {}
        self._last_slide_idx = slide_idx
        self._last_component = coords

    def concat(self, other):
        """ Concatenate two prototypes. """
        self.coords = np.vstack([self.coords, other.coords])

        new_bbox = np.empty((3, 2), np.int32)
        new_bbox[:, 0] = np.min((self.bbox[:, 0], other.bbox[:, 0]), axis=0)
        new_bbox[:, 1] = np.max((self.bbox[:, 1], other.bbox[:, 1]), axis=0)

        self._bbox = new_bbox
        self._contour = None
        self._borders = {}
        self._last_slide_idx = None
        self._last_component = None

    def _split_by_direction(self, coords):
        """ Direction-wise prototypes split.

        After depth-wise split we can have the situation when splitted part has more than one connected component.
        This method split disconnected parts into different prototypes.
        """
        unique_direction_coords = np.unique(coords[:, self.direction])
        # Slides distance more than 1 -> different objects
        split_indices = np.nonzero(unique_direction_coords[1:] - unique_direction_coords[:-1] > 1)[0]

        if len(split_indices) == 0:
            return [FaultPrototype(coords=coords, direction=self.direction)]

        start_indices = unique_direction_coords[split_indices + 1]
        start_indices = np.insert(start_indices, 0, 0)

        end_indices = unique_direction_coords[split_indices]
        end_indices = np.append(end_indices, unique_direction_coords[-1])

        prototypes = []

        for start_idx, end_idx in zip(start_indices, end_indices):
            coords_ = coords[(start_idx <= coords[:, self.direction]) & (coords[:, self.direction] <= end_idx)]
            prototype = FaultPrototype(coords=coords_, direction=self.direction, last_slide_idx=end_idx)
            prototypes.append(prototype)

        return prototypes

    def split(self, split_indices):
        """ Depth-wise prototypes split. """
        new_prototypes = []

        if (split_indices[0] is None) and (split_indices[1] is None):
            return self, new_prototypes

        # Cut upper part
        if split_indices[0] is not None:
            coords_outer = self.coords[self.coords[:, -1] < split_indices[0]]
            self.coords = self.coords[self.coords[:, -1] >= split_indices[0]]

            new_prototypes.extend(self._split_by_direction(coords_outer))

        # Cut lower part
        if split_indices[1] is not None:
            coords_outer = self.coords[self.coords[:, -1] > split_indices[1]]
            self.coords = self.coords[self.coords[:, -1] <= split_indices[1]]

            new_prototypes.extend(self._split_by_direction(coords_outer))

        new_prototypes.extend(self._split_by_direction(self.coords))
        return new_prototypes[-1], new_prototypes[:-1]

    def get_borders(self, border, projection_axis):
        """ Get contour border.

        Parameters
        ----------
        border : {'up', 'down', 'left', 'right'}
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

            border_coords = restore_coords_from_projection(coords=self.coords, buffer=border_coords, axis=projection_axis)
            self._borders[border] = border_coords

        return self._borders[border]


# Helpers
# Component dependencies
def _add_link(item_i, item_j, to_concat, concated_with):
    """ Add item_i and item_j to dependencies graph of elements to concat.

    `to_concat` is the dict in the format {'owner_idx': [items_indices]} and contains
    which components to concat into owner-component.
    `concated_with` is the dict in the format {'item_idx': owner_idx} and contains
    to which component (owner) merge item.

    ..!!..
    """
    if item_i not in concated_with:
        if item_j not in concated_with:
            # Add both
            owner, item = item_i, item_j

            to_concat[owner].append(item)

            concated_with[owner] = owner
            concated_with[item] = owner
        else:
            # Add item_i to item_j owner
            owner, item = concated_with[item_j], item_i

            to_concat[owner].append(item)
            concated_with[item] = owner
    else:
        owner = concated_with[item_i]

        if item_j not in concated_with:
            # Add item_j to item_i owner
            item = item_j

            to_concat[owner].append(item)
            concated_with[item] = owner
        else:
            # Merge items from item_j owner with item_i owner items
            other_owner = concated_with[item_j]

            # Merge item lists
            new_components = to_concat[other_owner] + [other_owner]
            to_concat[owner] += new_components
            del to_concat[other_owner]

            # Update owner links
            for item in new_components:
                concated_with[item] = owner
    return to_concat, concated_with
