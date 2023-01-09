""" [Draft] Extractor of fault surfaces from cloud of points. """
from collections import defaultdict, deque
import numpy as np

from cc3d import connected_components
from scipy.ndimage import find_objects
from scipy.ndimage.morphology import binary_dilation

from batchflow import Notifier

from .base import Fault
from .utils import dilate_coords, thin_coords, bboxes_intersected, bboxes_adjoin, max_depthwise_distance, find_border

# TODO: add class FaultPrototype with coords and bbox, and concat operation
# TODO: add info about connected prototypes (maybe into FaultPrototype)
# TODO: add prototype contour into instance

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
    def __init__(self, skeletonized_array, smoothed_array, orientation=0, component_len_threshold=0):
        """ Init data container with components info for each slide.
        # TODO: add skeletonized_array=None option

        ..!!..

        Parameters
        ----------
        orientation : {0, 1}
            Prediction orientation corresponding ilines (0) or xlines (1).
        component_len_threshold : int
            Threshold to filter out too small connected components on data slides.
        """
        self.shape = skeletonized_array.shape

        self.orientation = orientation
        self.orthogonal_orientation = 1 if orientation == 0 else 0

        self.component_len_threshold = component_len_threshold # TODO: temporally unused, change value
        # self.height_threshold = None # TODO: temporally unused, add later

        self.dilation = 3 # constant for internal operations

        self.container = {}

        self.prototypes_queue = deque() # prototypes for extension
        self.prototypes = []

        for slide_idx in Notifier('t')(range(self.shape[self.orientation])):
            mask = skeletonized_array[slide_idx, :, :]
            smoothed = smoothed_array[slide_idx, :, :]

            # Extract connected components from the slide
            labeled = connected_components(mask > 0)
            objects = find_objects(labeled)

            # Get components coords and eval lengths
            lengths = []
            objects_coords = []
            objects_bboxes = []

            for idx, object_bbox in enumerate(objects):
                # Refined coords: we refine skeletonize effects by applying it on limited area
                # TODO: improve skeletonize and reduce these steps
                dilation_axis = self.orthogonal_orientation
                dilation_ranges = (np.clip(object_bbox[0].start - self.dilation // 2, 0, None),
                                   np.clip(object_bbox[0].stop + self.dilation // 2, 0, self.shape[dilation_axis]))

                object_mask = labeled[dilation_ranges[0]:dilation_ranges[1], object_bbox[-1]] == idx + 1
                object_mask = binary_dilation(object_mask, structure=np.ones((1, self.dilation), bool))

                dilated_coords_2d = np.nonzero(object_mask)

                dilated_coords = np.zeros((len(dilated_coords_2d[0]), 3), dtype=int)

                dilated_coords[:, self.orientation] = slide_idx
                dilated_coords[:, dilation_axis] = dilated_coords_2d[0] + dilation_ranges[0]
                dilated_coords[:, 2] = dilated_coords_2d[1] + object_bbox[1].start

                smoothed_values = smoothed[dilated_coords[:, dilation_axis], dilated_coords[:, -1]]

                refined_coords = thin_coords(coords=dilated_coords, values=smoothed_values)

                objects_coords.append(refined_coords)

                # Bbox
                bbox = np.empty((3, 2), int)

                bbox[self.orientation, :] = slide_idx
                bbox[self.orthogonal_orientation, :] = (np.min(refined_coords[:, self.orthogonal_orientation]),
                                                        np.max(refined_coords[:, self.orthogonal_orientation]))
                                                        # TODO: fix bboxes everywhere: must be range, not max coord
                bbox[-1, :] = (np.min(refined_coords[:, -1]), np.max(refined_coords[:, -1]))

                objects_bboxes.append(bbox)

                # Length
                lengths.append(len(refined_coords))

            # Init merging state
            mergeable = np.array([True] * len(lengths))

            # Filter components by length
            lengths = np.array(lengths)
            is_too_small = lengths <= component_len_threshold
            mergeable[is_too_small] = False

            self.container[slide_idx] = {
                'objects_coords': objects_coords,
                'objects_bboxes': objects_bboxes,

                'lengths': lengths,

                'mergeable': mergeable
            }

        self._first_slide_with_mergeable = 0

    def extract_prototypes(self):
        """ Extract all fault prototypes from the point cloud. """
        prototype = True # init value for starting cycle

        while prototype is not None:
            prototype = self.extract_prototype()

            if prototype is not None:
                self.prototypes.append(prototype)

    def _add_new_component(self, slide_idx, coords):
        """ Add new items into the container. """
        # Object bbox
        mins_ = np.min(coords, axis=0)
        maxs_ = np.max(coords, axis=0)
        bbox = np.column_stack([mins_, maxs_])

        self.container[slide_idx]['objects_bboxes'].append(bbox)

        # Object coords
        self.container[slide_idx]['objects_coords'].append(coords)

        # Length
        length = len(coords)
        self.container[slide_idx]['lengths'] = np.append(self.container[slide_idx]['lengths'], length)

        # Merge state
        mergeable = length > self.component_len_threshold
        self.container[slide_idx]['mergeable'] = np.append(self.container[slide_idx]['mergeable'], mergeable)

    def _find_not_merged_component(self):
        """ Find the longest not merged item on the minimal slide. """
        idx = None

        for slide_idx in range(self._first_slide_with_mergeable, self.shape[self.orientation]):
            slide_info = self.container[slide_idx]

            if slide_info['mergeable'].any():
                max_len = np.max(slide_info['lengths'][slide_info['mergeable']])
                idx = np.argwhere((slide_info['lengths'] == max_len) & (slide_info['mergeable']))[0][0]
                self._first_slide_with_mergeable = slide_idx
                break

        return slide_idx, idx

    def extract_prototype(self):
        """ Extract one fault prototype from the point cloud. """
        idx = -1

        if len(self.prototypes_queue) == 0:
            start_slide_idx, idx = self._find_not_merged_component()

            if idx is None: # No components to concat
                return None

            component = self.container[start_slide_idx]['objects_coords'][idx]

            self.container[start_slide_idx]['mergeable'][idx] = False
            prototype = component
        else:
            prototype = self.prototypes_queue.popleft()

            start_slide_idx = np.max(prototype[:, self.orientation])
            component = prototype[prototype[:, self.orientation] == start_slide_idx]

        # Find closest components on next slides and split them if needed
        for slide_idx_ in range(start_slide_idx+1, self.shape[self.orientation]):
            component = dilate_coords(component, axis=self.orthogonal_orientation,
                                      max_value=self.shape[self.orthogonal_orientation]-1)

            # Find the closest component on the slide_idx_ to the current
            component, prototype_split_indices = self._find_closest_component(component=component,
                                                                              slide_idx=slide_idx_)
            # Postprocess prototype
            if component is not None:
                # Split current prototype and add new to queue
                if prototype_split_indices[0] is not None:
                    # Cut upper part of the prototype
                    new_prototype = prototype[prototype[:, -1] < prototype_split_indices[0]]

                    self.prototypes_queue.append(new_prototype)
                    prototype = prototype[prototype[:, -1] >= prototype_split_indices[0]]

                if prototype_split_indices[1] is not None:
                    # Cut lower part of the prototype
                    new_prototype = prototype[prototype[:, -1] > prototype_split_indices[1]]

                    self.prototypes_queue.append(new_prototype)
                    prototype = prototype[prototype[:, -1] <= prototype_split_indices[1]]

                prototype = np.vstack([prototype, component])
            else:
                break

        return prototype

    def _find_closest_component(self, component, slide_idx, distances_threshold=5,
                                depth_iteration_step=10, depths_threshold=5):
        """ Find the closest component to component on the slide, get splitting depths for them if needed.

        ..!!..
        """
        min_distance = distances_threshold

        closest_component = None
        prototype_split_indices = [None, None]

        mins_ = np.min(component, axis=0)
        maxs_ = np.max(component, axis=0)
        component_bbox = np.column_stack([mins_, maxs_])

        for idx, current_bbox in enumerate(self.container[slide_idx]['objects_bboxes']):
            if self.container[slide_idx]['mergeable'][idx]:
                # Check bbox intersection
                current_bbox[self.orthogonal_orientation, 0] -= self.dilation // 2 # dilate current_bbox
                current_bbox[self.orthogonal_orientation, 1] += self.dilation // 2

                if not bboxes_intersected(component_bbox, current_bbox, axes=(self.orthogonal_orientation, -1)):
                    continue

                # Check closeness of some points (iter over intersection depths with some step)
                current_component = self.container[slide_idx]['objects_coords'][idx]

                intersection_depths = (max(component_bbox[-1, 0], current_bbox[-1, 0]),
                                       min(component_bbox[-1, 1], current_bbox[-1, 1]))

                step = np.clip((intersection_depths[1]-intersection_depths[0])//3, 1, depth_iteration_step)

                distance = max_depthwise_distance(component, current_component,
                                                  depths_ranges=intersection_depths, step=step,
                                                  axis=self.orthogonal_orientation, max_threshold=min_distance)

                if distance < min_distance:
                    merged_idx = idx
                    min_distance = distance
                    intersection_borders = intersection_depths
                    closest_component = current_component
                    closest_component_bbox = current_bbox

                    if min_distance == 0:
                        break

        if closest_component is not None:
            # Merge founded component and split its extra parts
            self.container[slide_idx]['mergeable'][merged_idx] = False

            # Find prototype and new component splitting depths
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

    def find_connected_prototypes(self, intersection_ratio_threshold=None, contour_threshold=10, axis=2):
        """ Find prototypes which can be connected as puzzles.

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

        bboxes = [np.column_stack([np.min(coords, axis=0), np.max(coords, axis=0)]) for coords in self.prototypes]

        for j, prototype_1 in enumerate(self.prototypes):
            bbox_1 = bboxes[j]
            # TODO: Find contour and get it only in the intersection area?

            for k, prototype_2 in enumerate(self.prototypes[j+1:]):
                bbox_2 = bboxes[k + j + 1]

                intersection_borders = bboxes_adjoin(bbox_1, bbox_2, axis=axis)

                if intersection_borders[0] is None:
                    continue

                # Get area of interest
                intersection_coords_1 = prototype_1[(prototype_1[:, axis] >= intersection_borders[0] - margin) & \
                                                    (prototype_1[:, axis] <= intersection_borders[1] + margin)]

                intersection_coords_2 = prototype_2[(prototype_2[:, axis] >= intersection_borders[0] - margin) & \
                                                    (prototype_2[:, axis] <= intersection_borders[1] + margin)]

                # Prepare data for contouring
                # TODO: reduce this clause, sorting can be worthy
                if axis not in (-1, 2):
                    # For contour finding we apply groupby which works only for the last axis, so we swap axes coords
                    intersection_coords_1[:, [-1, axis]] = intersection_coords_1[:, [axis, -1]]
                    intersection_coords_2[:, [-1, axis]] = intersection_coords_2[:, [axis, -1]]

                    # Groupby implementation needs sorted data
                    intersection_coords_1 = intersection_coords_1[intersection_coords_1[:, 0].argsort()]
                    intersection_coords_2 = intersection_coords_2[intersection_coords_2[:, 0].argsort()]

                is_first_upper = (bbox_1[axis, 0] < bbox_2[axis, 0]) or (bbox_1[axis, 1] < bbox_2[axis, 1])

                # Find object contours in the area of interest
                contour_1 = find_border(coords=intersection_coords_1, find_lower_border=is_first_upper,
                                        projection_axis=self.orthogonal_orientation)
                contour_2 = find_border(coords=intersection_coords_2, find_lower_border=~is_first_upper,
                                        projection_axis=self.orthogonal_orientation)
                # TODO: rethink borders intersection, maybe images operations will be helpful

                # Simple check: if one data contour is much longer than other,
                # then we can't connect them as puzzle details
                length_ratio = min(len(contour_1), len(contour_2)) / max(len(contour_1), len(contour_2))

                if length_ratio < intersection_ratio_threshold:
                    continue

                # Shift one of the objects, making their contours intersected
                shift = 1 if is_first_upper else - 1
                contour_1[:, -1] += shift

                # Evaluate objects heights for threshold
                # TODO: rethink this part, it seems extra
                height_1 = contour_1[:, -1].max() - contour_1[:, -1].min() + 1
                height_2 = contour_2[:, -1].max() - contour_2[:, -1].min() + 1

                corrected_contour_threshold = contour_threshold

                # TODO: reduce next condition, maybe make more strict threshold

                # Flatten line-likable borders and objects with small borders and tighten threshold restrictions
                if (height_1 <= 1 + margin) or (height_2 <= 1 + margin): # one of them is a flatten line with borders on margin
                    # Get the most sufficient depth for objects
                    depths, frequencies = np.unique(np.hstack([contour_1[:, -1],
                                                               contour_2[:, -1]]), return_counts=True)

                    sufficient_depth = depths[np.argmax(frequencies)]

                    contour_1 = contour_1[contour_1[:, -1] == sufficient_depth]
                    contour_2 = contour_2[contour_2[:, -1] == sufficient_depth]

                    if (len(contour_1) == 0) or (len(contour_2) == 0):
                        # Contours are intersected on another depth, with less amount of points
                        continue

                    length_ratio = min(len(contour_2), len(contour_1)) / max(len(contour_2), len(contour_1))

                    if length_ratio < intersection_ratio_threshold:
                        continue

                    corrected_contour_threshold = 1

                # Check that one component contour is inside another (for both)
                # Objects can be shifted on orthogonal_orientation axis, so apply dilation for coords
                # TODO: try to reduce amount of checks with constraints
                # TODO: check reordering efficiency on huge data array
                # TODO: check images intersection efficiency
                if len(contour_1) > len(contour_2):
                    comparison_order = (contour_1, contour_2)
                else:
                    comparison_order = (contour_2, contour_1)

                for i in range(2):
                    contour_1_i = comparison_order[i]
                    contour_2_i = comparison_order[i != 1]

                    if self._is_contour_inside(contour_1_i, contour_2_i, contour_threshold=corrected_contour_threshold):
                        to_concat, concated_with = _add_link(item_i=j, item_j=k+j+1,
                                                            to_concat=to_concat, concated_with=concated_with)
                        break
        return to_concat

    def _is_contour_inside(self, contour_1, contour_2, contour_threshold):
        """ Check that `contour_1` is almost inside dilated `contour_2`. """
        contour_1_set = set(tuple(x) for x in contour_1)

        # Objects can be shifted on `self.orthogonal_orientation`
        contour_2_dilated = dilate_coords(coords=contour_2, dilate=self.dilation,
                                          axis=self.orthogonal_orientation,
                                          max_value=self.shape[self.orthogonal_orientation]-1)

        contour_2_dilated = set(tuple(x) for x in contour_2_dilated)

        return len(contour_1_set - contour_2_dilated) < contour_threshold

    def concat_connected_prototypes(self, axis=None):
        """ Find and concat prototypes connected by the `axis`.

        ..!!..
        """
        # Concat coords and remove concated parts
        to_concat = self.find_connected_prototypes(axis=axis)

        remove_elements = []

        for where_to_concat_idx, what_to_concat_indices in to_concat.items():
            coords = [self.prototypes[where_to_concat_idx]]

            for idx in what_to_concat_indices:
                coords.append(self.prototypes[idx])

            self.prototypes[where_to_concat_idx] = np.vstack(coords)
            remove_elements.extend(what_to_concat_indices)

        remove_elements.sort()
        remove_elements = remove_elements[::-1]

        for idx in remove_elements:
            _ = self.prototypes.pop(idx)

    def run_concat(self, iters=5):
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

            self.concat_connected_prototypes(axis=self.orientation)

            print("After ilines concat: ", len(self.prototypes))

            if len(self.prototypes) < previous_prototypes_amount:
                previous_prototypes_amount = len(self.prototypes)
            else:
                break

    def prototypes_to_faults(self, field):
        """ Convert all prototypes to faults. """
        faults = [Fault(prototype, field=field) for prototype in self.prototypes]
        return faults

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
