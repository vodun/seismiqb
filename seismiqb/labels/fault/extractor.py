""" [Draft] Extractor of fault surfaces from cloud of points. """
from collections import defaultdict, deque
import numpy as np

from cc3d import connected_components
from scipy.ndimage import find_objects
from scipy.ndimage.morphology import binary_dilation

from batchflow import Notifier

from .base import Fault
from .utils import (bboxes_adjacent, bboxes_intersected, dilate_coords, find_contour,
                    thin_coords, max_depthwise_distance, restore_coords_from_projection)
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
    def __init__(self, skeletonized_array, smoothed_array, direction=0, component_len_threshold=0):
        """ Init data container with components info for each slide.
        # TODO: add skeletonized_array=None option

        ..!!..

        Parameters
        ----------
        direction : {0, 1}
            Prediction direction corresponding ilines (0) or xlines (1).
        component_len_threshold : int
            Threshold to filter out too small connected components on data slides.
        """
        self.shape = skeletonized_array.shape

        self.direction = direction
        self.orthogonal_direction = 1 - self.direction

        self.component_len_threshold = component_len_threshold # TODO: temporally unused, change value
        # self.height_threshold = None # TODO: temporally unused, add later

        self.dilation = 3 # constant for internal operations

        self.container = {}

        self.prototypes_queue = deque() # prototypes for extension
        self.prototypes = [] # extracted prototypes

        for slide_idx in Notifier('t')(range(self.shape[self.direction])):
            mask = np.take(skeletonized_array, slide_idx, axis=self.direction)
            smoothed = np.take(smoothed_array, slide_idx, axis=self.direction)

            # Extract connected components from the slide
            labeled = connected_components(mask > 0)
            objects = find_objects(labeled)

            # Get components info
            coords, bboxes, lengths = [], [], []

            for idx, object_bbox in enumerate(objects, start=1):
                # Refined coords: we refine skeletonize effects by applying it on limited area
                dilation_axis = self.orthogonal_direction
                dilation_ranges = (max(object_bbox[0].start - self.dilation // 2, 0),
                                   min(object_bbox[0].stop + self.dilation // 2, self.shape[dilation_axis]))

                object_mask = labeled[dilation_ranges[0]:dilation_ranges[1], object_bbox[-1]] == idx
                object_mask = binary_dilation(object_mask, structure=np.ones((1, self.dilation), bool))

                dilated_coords_2d = np.nonzero(object_mask)

                dilated_coords = np.zeros((len(dilated_coords_2d[0]), 3), dtype=np.int16)

                dilated_coords[:, self.direction] = slide_idx
                dilated_coords[:, dilation_axis] = dilated_coords_2d[0] + dilation_ranges[0]
                dilated_coords[:, 2] = dilated_coords_2d[1] + object_bbox[1].start

                smoothed_values = smoothed[dilated_coords[:, dilation_axis], dilated_coords[:, -1]]

                refined_coords = thin_coords(coords=dilated_coords, values=smoothed_values)

                coords.append(refined_coords)

                # Bbox
                bbox = np.empty((3, 2), int)

                bbox[self.direction, :] = slide_idx
                bbox[self.orthogonal_direction, :] = (np.min(refined_coords[:, self.orthogonal_direction]),
                                                      np.max(refined_coords[:, self.orthogonal_direction]))
                bbox[-1, :] = (np.min(refined_coords[:, -1]), np.max(refined_coords[:, -1]))

                bboxes.append(bbox)

                # Length
                lengths.append(len(refined_coords))

            # Filter components by length
            lengths = np.array(lengths)
            is_too_small = lengths <= component_len_threshold
            lengths[is_too_small] = -1 # -1 is a flag for unmergeable components

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

            self.container[start_slide_idx]['lengths'][idx] = -1 # Mark this component as unmergeable
            prototype = FaultPrototype(coords=component, direction=self.direction,
                                       last_slide_idx=start_slide_idx, last_component=component)
        else:
            prototype = self.prototypes_queue.popleft()

            start_slide_idx = prototype.last_slide_idx
            component = prototype.last_component

        # Find closest components on next slides and split them if needed
        for slide_idx_ in range(start_slide_idx+1, self.shape[self.direction]):
            # Find the closest component on the slide_idx_ to the current
            component, prototype_split_indices = self._find_closest_component(component=component,
                                                                              slide_idx=slide_idx_)
            # Postprocess prototype
            if component is not None:
                # Split current prototype and add new to queue
                if prototype_split_indices[0] is not None:
                    new_prototype = prototype.split(split_depth=prototype_split_indices[0], cut_upper_part=True)
                    self.prototypes_queue.append(new_prototype)

                if prototype_split_indices[1] is not None:
                    new_prototype = prototype.split(split_depth=prototype_split_indices[1], cut_upper_part=False)
                    self.prototypes_queue.append(new_prototype)

                prototype.append(component, slide_idx=slide_idx_)
            else:
                break

        return prototype

    def _find_not_merged_component(self):
        """ Find the longest not merged item on the minimal slide. """
        idx = None

        for slide_idx in range(self._first_slide_with_mergeable, self.shape[self.direction]):
            slide_info = self.container[slide_idx]

            if (slide_info['lengths'] != -1).any():
                idx = np.argmax(slide_info['lengths'])
                self._first_slide_with_mergeable = slide_idx
                break

        return slide_idx, idx

    def _find_closest_component(self, component, slide_idx, distances_threshold=5,
                                depth_iteration_step=10, depths_threshold=5):
        """ Find the closest component to component on the slide, get splitting depths for them if needed.

        ..!!..
        """
        # Process inputs
        # Dilate component bbox for detecting close components: component on next slide can be shifted
        component_bbox = np.column_stack([np.min(component, axis=0), np.max(component, axis=0)])
        component_bbox[self.orthogonal_direction, 0] -= self.dilation // 2 # dilate bbox
        component_bbox[self.orthogonal_direction, 1] += self.dilation // 2

        min_distance = distances_threshold

        # Init returned values
        closest_component = None
        prototype_split_indices = [None, None]

        # Iter over components and find the closest one
        for idx, other_bbox in enumerate(self.container[slide_idx]['bboxes']):
            if self.container[slide_idx]['lengths'][idx] != -1:
                # Check bboxes intersection
                if not bboxes_intersected(component_bbox, other_bbox, axes=(self.orthogonal_direction, -1)):
                    continue

                # Check closeness of some points (iter over intersection depths with some step)
                # Faster then component intersection, but not so accurate
                # TODO: check intersection again; compare results
                other_component = self.container[slide_idx]['coords'][idx]

                intersection_depths = (max(component_bbox[-1, 0], other_bbox[-1, 0]),
                                       min(component_bbox[-1, 1], other_bbox[-1, 1]))

                step = np.clip((intersection_depths[1]-intersection_depths[0])//3, 1, depth_iteration_step)

                distance = max_depthwise_distance(component, other_component,
                                                  depths_ranges=intersection_depths, step=step,
                                                  axis=self.orthogonal_direction, max_threshold=min_distance)

                if distance < min_distance:
                    merged_idx = idx
                    min_distance = distance
                    intersection_borders = intersection_depths
                    closest_component = other_component
                    closest_component_bbox = other_bbox

                    if min_distance == 0:
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
                self._add_new_component(slide_idx=slide_idx, coords=new_component)

                # Extract suitable part
                closest_component = closest_component[closest_component[:, -1] >= item_split_idx]

            if closest_component_bbox[-1, 1] - intersection_borders[1] > depths_threshold:
                item_split_idx = intersection_borders[1]

                # Cut lower part of component on next slide and save extra data as another item
                new_component = closest_component[closest_component[:, -1] > item_split_idx]
                self._add_new_component(slide_idx=slide_idx, coords=new_component)

                # Extract suitable part
                closest_component = closest_component[closest_component[:, -1] <= item_split_idx]

        return closest_component, prototype_split_indices

    def _add_new_component(self, slide_idx, coords):
        """ Add new items into the container. """
        # Object bbox
        bbox = np.column_stack([np.min(coords, axis=0), np.max(coords, axis=0)])
        self.container[slide_idx]['bboxes'].append(bbox)

        # Object coords
        self.container[slide_idx]['coords'].append(coords)

        # Length
        length = len(coords) if len(coords) > self.component_len_threshold else -1
        self.container[slide_idx]['lengths'] = np.append(self.container[slide_idx]['lengths'], length)

    # Prototypes concatenation
    def concat_connected_prototypes(self, axis=None):
        """ Find and concat prototypes connected on the `axis`.

        ..!!..
        """
        # Concat coords and remove concated parts
        to_concat = self.find_connected_prototypes(axis=axis)

        remove_elements = []

        for where_to_concat_idx, what_to_concat_indices in to_concat.items():
            main_prototype = self.prototypes[where_to_concat_idx]

            for idx in what_to_concat_indices:
                main_prototype.concat(self.prototypes[idx])

            remove_elements.extend(what_to_concat_indices)

        remove_elements.sort()
        remove_elements = remove_elements[::-1]

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
            intersection_ratio_threshold = 0.5 if axis in (2, -1) else 0.95

        to_concat = defaultdict(list) # owner -> items
        concated_with = {} # item -> owner

        overlapping_axis = self.direction if axis in (-1, 2) else 2

        if axis in (-1, 2): # not necessary object borders for contour finding
            removed_borders = ('up', 'down')
        else:
            removed_borders = ('left', 'right')

        for j, prototype_1 in enumerate(self.prototypes):
            for k, prototype_2 in enumerate(self.prototypes[j+1:]):
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

                contour_1 = prototype_1.get_borders(removed_border=removed_borders[~is_first_upper], axis=axis)
                contour_2 = prototype_2.get_borders(removed_border=removed_borders[is_first_upper], axis=axis)

                # Get border contours in the area of interest
                intersection_range = (min(adjacent_borders[axis]) - margin, max(adjacent_borders[axis]) + margin)

                contour_1 = contour_1[(contour_1[:, axis] >= intersection_range[0]) & \
                                      (contour_1[:, axis] <= intersection_range[1])]
                contour_2 = contour_2[(contour_2[:, axis] >= intersection_range[0]) & \
                                      (contour_2[:, axis] <= intersection_range[1])]
                

                # If one data contour is much longer than other, then we can't connect them as puzzle details
                length_ratio = min(len(contour_1), len(contour_2)) / max(len(contour_1), len(contour_2))

                if length_ratio < intersection_ratio_threshold:
                    continue

                # Shift one of the objects, making their contours intersected
                shift = 1 if is_first_upper else -1
                contour_1[:, -1] += shift

                # Process objects with too small border contours
                if (len(contour_1) < 4*contour_threshold) or (len(contour_2) < 4*contour_threshold):
                    corrected_contour_threshold = 1
                else:
                    corrected_contour_threshold = contour_threshold

                # Check that one component contour is inside another (for both)
                if self._is_contour_inside(contour_1, contour_2, contour_threshold=corrected_contour_threshold):
                    to_concat, concated_with = _add_link(item_i=j, item_j=k+j+1,
                                                         to_concat=to_concat, concated_with=concated_with)
                    continue

                if self._is_contour_inside(contour_2, contour_1, contour_threshold=corrected_contour_threshold):
                    to_concat, concated_with = _add_link(item_i=j, item_j=k+j+1,
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

    # Addons
    def run_prototypes_concat(self, iters=5):
        """ Only for tests. Will be removed. """
        previous_prototypes_amount = len(self.prototypes) + 100 # to avoid stopping after first concat
        print("Start amount: ", len(self.prototypes))

        for _ in range(iters):
            self.concat_connected_prototypes(axis=-1)

            print("After depths concat: ", len(self.prototypes))

            if len(self.prototypes) < previous_prototypes_amount:
                previous_prototypes_amount = len(self.prototypes)
            else:
                break

            self.concat_connected_prototypes(axis=self.direction)

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
            self._last_slide_idx = self.coordinates[:, self.direction].max()
        return self._last_slide_idx

    @property
    def last_component(self):
        if self._last_component is None:
            self._last_component = self.coordinates[self.coordinates[:, self.direction] == self.last_slide_idx]
        return self._last_component

    @property
    def contour(self):
        """ Contour of 2d projection on axis, orthogonal to self.direction."""
        if self._contour is None:
            projection_axis = 1 - self.direction
            self._contour = find_contour(coords=self.coords, projection_axis=projection_axis)
        return self._contour

    def append(self, coords, slide_idx=None):
        """ Append new coords into prototype. """
        self.coords = np.vstack([self.coords, coords])

        self.reset_borders()
        self._last_slide_idx = slide_idx
        self._last_component = coords

    def concat(self, other):
        """ Concatenate two prototypes. """
        self.coords = np.vstack([self.coords, other.coords])

        self.reset_borders()
        self._last_slide_idx = None
        self._last_component = None

    def split(self, split_depth, cut_upper_part):
        """ Split prototype into two parts by `split_depth`. """
        if cut_upper_part:
            new_coords = self.coords[self.coords[:, -1] < split_depth]

            self.coords = self.coords[self.coords[:, -1] >= split_depth]
            self._last_component = self.last_component[self.last_component[:, -1] >= split_depth]
        else:
            new_coords = self.coords[self.coords[:, -1] > split_depth]

            self.coords = self.coords[self.coords[:, -1] <= split_depth]
            self._last_component = self.last_component[self.last_component[:, -1] <= split_depth]

        self.reset_borders()

        new_last_slide = np.max(new_coords[:, self.direction])
        new_last_component = new_coords[new_coords[:, self.direction] == new_last_slide]

        new_prototype = FaultPrototype(coords=new_coords, direction=self.direction,
                                       last_slide_idx=new_last_slide, last_component=new_last_component)
        return new_prototype

    def get_borders(self, removed_border, axis):
        """ Get contour borders except the one.

        Parameters
        ----------
        removed_border : {'up', 'down', 'left', 'right'}
        """
        if removed_border not in self._borders.keys():
            # Delete extra border from contour
            # For border removing we apply groupby which works only for the last axis, so we swap axes coords
            if removed_border in ('left', 'right'):
                border_coords = self.contour.copy()
                border_coords[:, [-1, axis]] = border_coords[:, [axis, -1]]
                border_coords = border_coords[border_coords[:, axis].argsort()] # Groupby needs sorted data
            else:
                border_coords = self.contour

            # Delete border by applying groupby
            if removed_border in ('up', 'left'):
                border_coords = groupby_max(border_coords)
            else:
                border_coords = groupby_min(border_coords)

            # Restore 3d coordinates
            projection_axis = 1 - self.direction

            if removed_border in ('left', 'right'):
                border_coords[:, [-1, axis]] = border_coords[:, [axis, -1]]

            border_coords = restore_coords_from_projection(coords=self.coords, buffer=border_coords, axis=projection_axis)
            self._borders[removed_border] = border_coords

        return self._borders[removed_border]

    def reset_borders(self):
        """..!!.."""
        self._bbox = None
        self._contour = None
        self._borders = {}


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
    if item_i not in concated_with.keys():
        if item_j not in concated_with.keys():
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

        if item_j not in concated_with.keys():
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
