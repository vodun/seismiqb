""" [Draft] Extractor of fault surfaces from cloud of points. """
from collections import defaultdict
import numpy as np

from cc3d import connected_components
from scipy.ndimage import find_objects

from batchflow import Notifier

# from .base import Fault TODO: add prototypes_to_faults method
from .utils import dilate_coords, thin_coords, bboxes_intersected, bboxes_adjoin, depthwise_distances, find_border

class FaultExtractor:
    """ ..!!..

    Main logic implementation. There are much to do and much to optimize in this code."""
    def __init__(self, skeletonized_array, smoothed_array, orientation=0, component_length_threshold=0):
        """ Init data container with components info for each slide.

        ..!!..
        """
        self.shape = skeletonized_array.shape

        self.orientation = orientation
        self.orthogonal_orientation = 1 if orientation == 0 else 0

        self.component_length_threshold = component_length_threshold # TODO: temporally unused, change value
        # self.height_threshold = None # TODO: temporally unused, add later

        self.dilation_width = 3 # for internal operations, TODO: remove after checks

        self.container = {}
        self.components_queue = []
        self.n_not_merged = 0 # TODO: add proper update on extension stage

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
                # Bbox
                bbox = np.empty((3, 2), int)
                bbox_2d = np.array([[slc.start, slc.stop-1] for slc in object_bbox])

                bbox[self.orientation, :] = np.array([slide_idx, slide_idx], int)
                bbox[self.orthogonal_orientation, :] = bbox_2d[0, :]
                bbox[-1, :] = bbox_2d[-1, :]

                objects_bboxes.append(bbox)

                # Coords
                coords_2d = np.nonzero(labeled[object_bbox]==idx+1)

                object_coords = np.zeros((len(coords_2d[0]), 3), dtype=int)

                object_coords[:, self.orientation] = slide_idx
                object_coords[:, self.orthogonal_orientation] = coords_2d[0] + bbox_2d[0, 0]
                object_coords[:, 2] = coords_2d[1] + bbox_2d[1, 0]

                objects_coords.append(object_coords)

                # Length
                lengths.append(len(object_coords))

            # Init merging state
            mergeable = np.array([True for i in range(len(lengths))])

            # Filter components by length
            lengths = np.array(lengths)
            is_too_small = lengths <= component_length_threshold
            mergeable[is_too_small] = False

            self.container[slide_idx] = {
                'smoothed': smoothed,

                'objects_coords': objects_coords,
                'objects_bboxes': objects_bboxes,

                'lengths': lengths,

                'mergeable': mergeable
            }

            self.n_not_merged += np.sum(mergeable)

    def extract_prototypes(self):
        """ ..!!.. """
        prototypes = []

        while self.n_not_merged > 0:
            component_coords, merged, new_items, to_break = self.component_extraction()

            if to_break: # TODO: reduce this clause, it must be extra (fix n_not_merged evaluation)
                break

            # Update data
            self.n_not_merged -= len(merged)

            for slide_idx, object_idx in merged:
                self.container[slide_idx]['mergeable'][object_idx] = False

            # Add new items (from splitted parts)
            for slide_idx, new_items_list in new_items.items():
                self.n_not_merged += len(new_items_list)

                for new_item_coords in new_items_list:
                    # Object bbox
                    mins_ = np.min(new_item_coords, axis=0)
                    maxs_ = np.max(new_item_coords, axis=0)
                    bbox = np.column_stack([mins_, maxs_])

                    self.container[slide_idx]['objects_bboxes'].append(bbox)

                    # Object coords
                    self.container[slide_idx]['objects_coords'].append(new_item_coords)

                    # Length
                    length = len(new_item_coords)
                    self.container[slide_idx]['lengths'] = np.append(self.container[slide_idx]['lengths'], length)

                    # Merge state
                    mergeable = False if length <= self.component_length_threshold else True
                    self.container[slide_idx]['mergeable'] = np.append(self.container[slide_idx]['mergeable'],
                                                                       mergeable)

            prototypes.append(component_coords)

        return prototypes

    def find_not_merged_component(self):
        """ Find the longest not merged item on the minimal slide. """
        idx = None

        for slide_idx in range(self.shape[self.orientation]):
            if self.container[slide_idx]['mergeable'].any():
                max_len = np.max(self.container[slide_idx]['lengths'][self.container[slide_idx]['mergeable']])
                idx = np.argwhere((self.container[slide_idx]['lengths'] == max_len) & \
                                  (self.container[slide_idx]['mergeable']))[0][0]
                break

        return slide_idx, idx

    def component_extraction(self):
        """ Extract one component from cloud of points. """
        idx = -1

        if len(self.components_queue) == 0:
            start_slide_idx, idx = self.find_not_merged_component()

            if idx is None: # TODO: reduce this extra condition (fix n_not_merged update)
                return None, None, None, True

            merged = [(start_slide_idx, idx)]
            component_coords = self.container[start_slide_idx]['objects_coords'][idx]

            # Extract more close object skeleton
            dilated_component_coords = dilate_coords(component_coords, axis=self.orthogonal_orientation,
                                                     max_value=self.shape[self.orthogonal_orientation]-1)
            component_coords = self.extract_component_from_proba(slide_idx=start_slide_idx,
                                                                 coords=dilated_component_coords)

            current_slide_component_coords = component_coords
        else:
            merged = []
            component_coords = self.components_queue.pop()

            start_slide_idx = np.max(component_coords[:, self.orientation])
            current_slide_component_coords = component_coords[component_coords[:, self.orientation] == start_slide_idx]

        new_items = defaultdict(list)

        # Find closest components on next slides and split them if needed
        for slide_idx_ in range(start_slide_idx+1, self.shape[self.orientation]):
            current_slide_component_coords = dilate_coords(current_slide_component_coords, axis=self.orthogonal_orientation,
                                                           max_value=self.shape[self.orthogonal_orientation]-1)

            # Find the closest component on the slide_idx_ to the current
            current_slide_component_coords, merged_idx, item_split_indices, component_split_indices = self.find_closest_component(component_coords=current_slide_component_coords,
                                                                                                                                  slide_idx=slide_idx_)

            # Process founded component part
            if current_slide_component_coords is not None:
                merged.append((slide_idx_, merged_idx))

                # Extract component from the closest mask
                current_slide_component_coords = self.extract_component_from_proba(slide_idx=slide_idx_,
                                                                                   coords=current_slide_component_coords)

                # Split current component and add new to queue
                if component_split_indices[0] is not None:
                    # Cut upper part of the component
                    new_component_coords = component_coords[component_coords[:, -1] < component_split_indices[0]]

                    if len(new_component_coords) > 0: # TODO: check that it is an extra condition
                        self.components_queue.append(new_component_coords)
                        component_coords = component_coords[component_coords[:, -1] >= component_split_indices[0]]

                if component_split_indices[1] is not None:
                    # Cut lower part of the component
                    new_component_coords = component_coords[component_coords[:, -1] > component_split_indices[1]]

                    if len(new_component_coords) > 0: # TODO: check that it is an extra condition
                        self.components_queue.append(new_component_coords)
                        component_coords = component_coords[component_coords[:, -1] <= component_split_indices[1]]

                # Create splitted items and save them such new elements for merge
                if item_split_indices[0] is not None:
                    # Cut upper part of component on next slide
                    # Save extra data as another item
                    new_item_coords = current_slide_component_coords[current_slide_component_coords[:, -1] < item_split_indices[0]]
                    new_items[slide_idx_].append(new_item_coords)

                    # Extract suitable part
                    current_slide_component_coords = current_slide_component_coords[current_slide_component_coords[:, -1] >= item_split_indices[0]]

                if item_split_indices[1] is not None:
                    # Cut lower part of component on next slide
                    # Save extra data as another item
                    new_item_coords = current_slide_component_coords[current_slide_component_coords[:, -1] > item_split_indices[1]]
                    new_items[slide_idx_].append(new_item_coords)

                    # Extract suitable part
                    current_slide_component_coords = current_slide_component_coords[current_slide_component_coords[:, -1] <= item_split_indices[1]]

                component_coords = np.vstack([component_coords, current_slide_component_coords])
            else:
                break

        return component_coords, merged, new_items, False

    def extract_component_from_proba(self, slide_idx, coords):
        """ Extract component coordinates from the area defined by `coords` argument. """
        # Get probabilities in the dilated area and extract thin component
        smoothed_proba = self.container[slide_idx]['smoothed'][coords[:, self.orthogonal_orientation], coords[:, -1]]

        component_coords = thin_coords(coords=coords, values=smoothed_proba)
        component_coords[:, self.orientation] = slide_idx
        return component_coords

    def find_closest_component(self, component_coords, slide_idx, distances_threshold=10,
                               depth_iteration_step=10, depths_threshold=10):
        """ Find the closest component to component on the slide, get splitting depths for them if needed.

        ..!!..
        """
        closest_component_coords = None
        merged_idx = None
        item_split_indices = [None, None]
        component_split_indices = [None, None]

        mins_ = np.min(component_coords, axis=0)
        maxs_ = np.max(component_coords, axis=0)
        component_bbox = np.column_stack([mins_, maxs_])

        component_depth_ranges = (component_coords[:, -1].min(), component_coords[:, -1].max())

        min_distance = distances_threshold
        best_intersection_depths = None

        for idx, current_bbox in enumerate(self.container[slide_idx]['objects_bboxes']):
            if self.container[slide_idx]['mergeable'][idx]:
                # Check bbox intersection
                dilated_bbox = current_bbox.copy()
                dilated_bbox[self.orthogonal_orientation, 0] -= self.dilation_width // 2 # dilate current_bbox
                dilated_bbox[self.orthogonal_orientation, 1] += self.dilation_width // 2

                if not bboxes_intersected(component_bbox, dilated_bbox, axes=(self.orthogonal_orientation, -1)):
                    continue

                # Check closiness of upper and lower intersection border points - TODO: add more points
                current_component_coords = self.container[slide_idx]['objects_coords'][idx]
                current_component_depth_ranges = (current_component_coords[:, -1].min(),
                                                  current_component_coords[:, -1].max())

                intersection_depths = (max(component_depth_ranges[0], current_component_depth_ranges[0]),
                                       min(component_depth_ranges[1], current_component_depth_ranges[1]))

                step = np.clip(intersection_depths[1]-intersection_depths[0], 1, depth_iteration_step)

                borders_distance = depthwise_distances(component_coords, current_component_coords,
                                                       depths_ranges=intersection_depths, step=step,
                                                       axis=self.orthogonal_orientation,
                                                       max_threshold=min_distance)

                if borders_distance < min_distance: # min_distance initialization value is a threshold
                    min_distance = borders_distance
                    best_intersection_depths = intersection_depths

                    closest_component_coords = current_component_coords
                    merged_idx = idx

                    if borders_distance == 0:
                        break

        if closest_component_coords is not None:
            closest_component_coords = dilate_coords(closest_component_coords, axis=self.orthogonal_orientation,
                                                     max_value=self.shape[self.orthogonal_orientation]-1)
            intersection_height = best_intersection_depths[1] - best_intersection_depths[0]

            if (intersection_height < 0.2*len(component_coords)/self.dilation_width) or \
               (intersection_height < 0.2*len(closest_component_coords)/self.dilation_width):
                # Coords overlap too small -> different components
                closest_component_coords = None
                merged_idx = None
            else:
                # Components are close, find splitting depths
                intersection_borders = best_intersection_depths
                component_borders = np.min(component_coords[:, -1]), np.max(component_coords[:, -1])
                new_component_borders = np.min(closest_component_coords[:, -1]), np.max(closest_component_coords[:, -1])

                # Find differences between borders and intersected part
                borders_diffs_for_new = (intersection_borders[0] - new_component_borders[0],
                                         new_component_borders[1] - intersection_borders[1])
                borders_diffs_for_main = (intersection_borders[0] - component_borders[0],
                                          component_borders[1] - intersection_borders[1])

                # Split or not
                # Get split for the new part
                if borders_diffs_for_new[0] > depths_threshold:
                    item_split_indices[0] = intersection_borders[0]

                if borders_diffs_for_new[1] > depths_threshold:
                    item_split_indices[1] = intersection_borders[1]

                # Get split for the old part
                if borders_diffs_for_main[0] > depths_threshold:
                    component_split_indices[0] = intersection_borders[0]

                if borders_diffs_for_main[1] > depths_threshold:
                    component_split_indices[1] = intersection_borders[1]

        return closest_component_coords, merged_idx, item_split_indices, component_split_indices

    def find_connected_prototypes(self, prototypes, intersection_ratio=None, contour_threshold=10, axis=2):
        """ Find prototypes which can be connected as puzzles.

        ..!!..
        Parameters
        ----------
        intersection_ratio : float
            Prototypes contours intersection ratio to decide that prototypes are not close.
        contour_threshold : int
            Amount of points in contour to decide that prototypes are not close.
        """
        margin = 1 # local constant for code prettifying

        if intersection_ratio is None:
            intersection_ratio = 0.5 if axis in (2, -1) else 0.95

        to_concat = defaultdict(list) # owner -> items
        concated_with = {} # item -> owner

        bboxes = [np.column_stack([np.min(coords, axis=0), np.max(coords, axis=0)]) for coords in prototypes]

        for j, coords_1 in enumerate(prototypes):
            bbox_1 = bboxes[j]

            for k, coords_2 in enumerate(prototypes[j+1:]):
                bbox_2 = bboxes[k + j + 1]

                low_intersection_border, high_intersection_border = bboxes_adjoin(bbox_1, bbox_2, axis=axis)

                if low_intersection_border is None:
                    continue

                # Get area of interest
                intersection_coords_1 = coords_1[(coords_1[:, axis] >= low_intersection_border - margin) & \
                                                (coords_1[:, axis] <= high_intersection_border + margin)]

                intersection_coords_2 = coords_2[(coords_2[:, axis] >= low_intersection_border - margin) & \
                                                (coords_2[:, axis] <= high_intersection_border + margin)]

                is_first_upper = (bbox_1[axis, 0] < bbox_2[axis, 0]) or (bbox_1[axis, 1] < bbox_2[axis, 1])

                # Prepare data for contouring
                if axis not in (-1, 2):
                    # For contour finding we apply groupby which works only for the last axis, so we swap axes coords
                    intersection_coords_1[:, [-1, axis]] = intersection_coords_1[:, [axis, -1]]
                    intersection_coords_2[:, [-1, axis]] = intersection_coords_2[:, [axis, -1]]

                    # Groupby implementation needs sorted data
                    intersection_coords_1 = intersection_coords_1[intersection_coords_1[:, 0].argsort()]
                    intersection_coords_2 = intersection_coords_2[intersection_coords_2[:, 0].argsort()]

                # Find object contours in the area of interest
                contour_1_coords = find_border(coords=intersection_coords_1, find_lower_border=is_first_upper,
                                               projection_axis=self.orthogonal_orientation)
                contour_2_coords = find_border(coords=intersection_coords_2, find_lower_border=~is_first_upper,
                                               projection_axis=self.orthogonal_orientation)

                # Simple check: if one data contour is much longer than other,
                # then we can't connect them as puzzle details
                length_ratio = min(len(contour_1_coords), len(contour_2_coords)) / \
                               max(len(contour_1_coords), len(contour_2_coords))

                if length_ratio < intersection_ratio:
                    continue

                # Shift one of the objects, making their borders intersected
                shift = 1 if is_first_upper else - 1
                contour_1_coords[:, -1] += shift

                # Evaluate objects heights for thresholding
                height_1 = contour_1_coords[:, -1].max() - contour_1_coords[:, -1].min() + 1
                height_2 = contour_2_coords[:, -1].max() - contour_2_coords[:, -1].min() + 1

                corrected_contour_threshold = contour_threshold

                # Flatten line-likable borders and objects with small borders and tighten thresholding restrictions
                if (len(contour_1_coords) <= contour_threshold+2*margin) or \
                   (len(contour_2_coords) <= contour_threshold+2*margin) or \
                   (height_1 < 3) or (height_2 < 3):
                    # Get the most sufficient depth for each object
                    depths, frequencies = np.unique(np.hstack([contour_1_coords[:, -1],
                                                               contour_2_coords[:, -1]]), return_counts=True)

                    sufficient_depth = depths[np.argmax(frequencies)]

                    contour_1_coords = contour_1_coords[contour_1_coords[:, -1] == sufficient_depth]
                    contour_2_coords = contour_2_coords[contour_2_coords[:, -1] == sufficient_depth]

                    if (len(contour_1_coords) == 0) or (len(contour_2_coords) == 0):
                        # Contours are intersected on another depth, with less amount of points
                        break

                    length_ratio = min(len(contour_2_coords), len(contour_1_coords)) / \
                                   max(len(contour_2_coords), len(contour_1_coords))

                    if length_ratio < intersection_ratio:
                        continue

                    if len(contour_1_coords) < 3 or len(contour_2_coords) < 3:
                        corrected_contour_threshold = 0
                    else:
                        corrected_contour_threshold = 1

                # Check that one component projection is inside another (for both)
                # Objects can be shifted on orthogonal_orientation axis, so apply dilation for coords
                ### Variant 1
                contour_1_coords_dilated = dilate_coords(coords=contour_1_coords, dilate=self.dilation_width,
                                                         axis=self.orthogonal_orientation)
                contour_2_coords_dilated = dilate_coords(coords=contour_2_coords, dilate=self.dilation_width,
                                                         axis=self.orthogonal_orientation)

                contour_1_coords_dilated_set = set([tuple(x) for x in contour_1_coords_dilated])
                contour_2_coords_dilated_set = set([tuple(x) for x in contour_2_coords_dilated])

                contour_1_coords_set = set([tuple(x) for x in contour_1_coords])
                contour_2_coords_set = set([tuple(x) for x in contour_2_coords])

                second_is_subset_of_first = len(contour_2_coords_set - contour_1_coords_dilated_set)
                first_is_subset_of_second = len(contour_1_coords_set - contour_2_coords_dilated_set)

    #             #### Variant 2 - TODO fix and check timings
    #             ind = np.lexsort((contour_1_coords[:, 2], contour_1_coords[:, 1], contour_1_coords[:, 0]))
    #             contour_1_coords = contour_1_coords[ind, :]

    #             ind = np.lexsort((contour_2_coords[:, 2], contour_2_coords[:, 1], contour_2_coords[:, 0]))
    #             contour_2_coords = contour_2_coords[ind, :]

    #             contour_1_coords_dilated = dilate_coords(coords=contour_1_coords, dilate=self.dilation_width,
    #                                                      axis=self.orthogonal_orientation)
    #             contour_2_coords_dilated = dilate_coords(coords=contour_2_coords, dilate=self.dilation_width,
    #                                                      axis=self.orthogonal_orientation)

    #             second_is_subset_of_first_ = n_differences_for_coords(contour_2_coords, contour_1_coords_dilated,
    #                                                                   max_threshold=corrected_contour_threshold)
    #             first_is_subset_of_second_ = n_differences_for_coords(contour_1_coords, contour_2_coords_dilated,
    #                                                                   max_threshold=corrected_contour_threshold)
    #             ####

                second_is_subset_of_first = second_is_subset_of_first < corrected_contour_threshold
                first_is_subset_of_second = first_is_subset_of_second < corrected_contour_threshold

                length_ratio = min(len(contour_2_coords_set), len(contour_1_coords_set)) / \
                               max(len(contour_2_coords_set), len(contour_1_coords_set))

                if (second_is_subset_of_first or first_is_subset_of_second) and (length_ratio >= intersection_ratio):                    
                    to_concat, concated_with = _add_link(item_i=j, item_j=k+j+1,
                                                         to_concat=to_concat, concated_with=concated_with)

        return to_concat

    def concat_connected_prototypes(self, prototypes, axis=None):
        """ Find and concat prototypes connected by the `axis`.

        ..!!..
        """
        # Concat coords and remove concated parts
        to_concat = self.find_connected_prototypes(prototypes=prototypes, axis=axis)

        remove_elements = []

        for where_to_concat_idx, what_to_concat_idxs in to_concat.items():
            coords = [prototypes[where_to_concat_idx]]

            for idx in what_to_concat_idxs:
                coords.append(prototypes[idx])

            prototypes[where_to_concat_idx] = np.vstack(coords)
            remove_elements.extend(what_to_concat_idxs)

        remove_elements.sort()
        remove_elements = remove_elements[::-1]

        for idx in remove_elements:
            _ = prototypes.pop(idx)

        return prototypes

    def run_concat(self, prototypes, iters=5):
        """ Only for tests. Will be removed. """
        previous_prototypes_amount = len(prototypes) + 100 # to avoid stopping after first concat
        print("Start amount: ", len(prototypes))

        for _ in range(iters):
            prototypes = self.concat_connected_prototypes(prototypes=prototypes, axis=self.orientation)

            print("After ilines concat: ", len(prototypes))

            if len(prototypes) < previous_prototypes_amount:
                previous_prototypes_amount = len(prototypes)
            else:
                break

            prototypes = self.concat_connected_prototypes(prototypes=prototypes, axis=-1)

            print("After depths concat: ",len(prototypes))

            if len(prototypes) < previous_prototypes_amount:
                previous_prototypes_amount = len(prototypes)
            else:
                break

        return prototypes

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
            new_items = to_concat[other_owner] + [other_owner]
            to_concat[owner] += new_items
            del to_concat[other_owner]

            # Update owner links
            for item in new_items:
                concated_with[item] = owner
    return to_concat, concated_with
