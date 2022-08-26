""" Extractor of fault surfaces from point of clouds. """

from builtins import StopIteration
from itertools import combinations

import tqdm
import numpy as np
from scipy.ndimage import measurements
from scipy.sparse.csgraph import connected_components

from ...geometry import SeismicGeometry
from ...utils import make_slices

class ComponentsLabeler:
    """ Label conneted components in the array (or in its region) by slices along the axis.

    Parameters
    ----------
    array : numpy.ndarray or an object with the same slicing

    ranges : tuple of slices
        Region of the array for labeling

    axis : int, optional
        Orientation of the array to label, by default 2

    Attributes
    ----------
    labels : numpy.array
        Aray with components labels
    n_objects : int
        The total number of components (except zero-background)
    objects : list
        Bounding boxes of labeled components
    sizes : list
        Size as the sum of values in the initial array for each component
    abs_sizes : list
        Size as the number of pixels for each component
    sorted_labels : list
        Labels sorted by sizes
    """
    def __init__(self, array, ranges, axis=2):
        self.array = array[ranges]
        self.ranges = ranges
        self.axis = axis

        self.labels = None
        self.n_objects = None
        self.objects = None
        self.sizes = None
        self.abs_sizes = None
        self.sorted_labels = None

        self.find_objects()

    def find_objects(self):
        """ Label array.  """
        structure = self._get_structure()
        labels, n_objects = measurements.label(self.array, structure=structure)
        objects = measurements.find_objects(labels)

        sizes, abs_sizes = self._get_objects_sizes(labels, objects)
        sorted_labels = sorted(range(n_objects+1), key=lambda x: sizes[x], reverse=True)

        self.labels = labels
        self.n_objects = n_objects
        self.objects = objects
        self.sizes = sizes
        self.abs_sizes = abs_sizes
        self.sorted_labels = sorted_labels

    def _get_structure(self):
        """ Create structure for `scipy.ndimage.measurements.label` for 2D components labeling. """
        structure = np.zeros((3, 3, 3))
        slices = [slice(None) for _ in range(3)]
        slices[self.axis] = 1
        structure[tuple(slices)] = 1
        return structure

    def _get_objects_sizes(self, labels, objects):
        """ Compute `sizes` and `abs_sizes` of components. The zero label (background) is isgnored. """
        coords = list(range(3))
        coords.pop(self.axis)
        sizes = [0]
        abs_sizes = [0]
        for idx, obj in enumerate(objects):
            mask = (labels[obj] == idx + 1)
            sizes.append(self.array[obj][mask].sum())
            abs_sizes.append(mask.sum())
        return sizes, abs_sizes

    def component_loc(self, idx):
        """ Find the position of the slide where component is located. """
        return self.objects[idx-1][self.axis].start

    def component_points(self, idx):
        """ Get point cloud of the component in coordinates of the initial array. """
        points = np.stack(np.where(self.labels[self.objects[idx-1]] == idx), axis=1)
        object_origin = [item.start for item in self.objects[idx-1]]
        array_origin = [item.start or 0 for item in self.ranges]
        return points + object_origin + array_origin

    def components_stats(self, a, b):
        """ Get sizes (in points) of each component and sizes of its intersection. """
        slices = [slice(min(i.start, j.start), max(i.stop, j.stop)) for i, j in zip(self.objects[b-1][:2], self.objects[a-1][:2])]
        b_mask = self.labels[slices[0], slices[1], self.component_loc(b)] == b
        a_mask = self.labels[slices[0], slices[1], self.component_loc(a)] == a
        intersection = np.logical_and(a_mask, b_mask)
        return a_mask.sum(), b_mask.sum(), intersection.sum()

    def get_neighbours(self, idx, direction=1):
        """ For component with label `idx` find intersected components on the next slide in the `direction`. """
        slices = self.objects[idx-1][:2]
        if (direction == 1 and self.component_loc(idx) == self.array.shape[2]-1) or (direction == -1 and self.component_loc(idx) == 0):
            return []
        mask = (self.labels[slices[0], slices[1], self.component_loc(idx)] == idx)
        components = np.unique(self.labels[slices[0], slices[1], self.component_loc(idx) + direction][mask])
        return components[components > 0]

class SurfacesExtractor:
    def __init__(self, cube, axis=0, ranges=None):
        """ Extract fault surfaces from array with fault labeles (probabilities or 0-1).
        Note assumption that the connected components on each depth slice do not have branches.

        The algorithm is based on three stages:
            - create patches (connected sequences of horizontal components (see :class:`.FaultPatch`)).
              Each patch has child patches and parent patches so all of them can be organized into graph.
            - merge patches around the holes (find cycles)
            - merge patches with large intersection of the bound components

        Parameters
        ----------
        cube : str, numpy.ndarray or an object with the same slicing
            Cube with fault probabilities
        axis : int, optional
            Direction to find components (across ilines or crosslines), by default 0 (ilines).
        ranges : tuple of slices, optional
            Region in cube to find surfaces, by default None (entire cube).
        """
        if isinstance(cube, str):
            cube = SeismicGeometry(cube)
        self.cube = cube
        self.ranges = make_slices(ranges, cube.shape)
        self.axis = axis

    def make_labels(self):
        """ Find connected components in depth slices and along one axis. """
        self.h_labeler = ComponentsLabeler(self.cube, self.ranges, axis=2)
        return self

    def next_candidate(self):
        """ Next component to be anchor of the new patch (see :class:`.FaultPatch`).

        Returns
        -------
        component_index : int
            The initial component of the new patch.
        direction : -1, 0 or 1
            Direction of the patch extension (up or down). 0 means extension in both directions.
        """
        while True:
            # TODO: more accurate sampling
            if np.random.random() < 0.9 and (len(self.candidates[1]) > 0 or len(self.candidates[-1]) > 0):
                direction = np.random.choice([-1, 1])
                if len(self.candidates[direction]) == 0:
                    direction = -direction
                candidates = list(self.candidates[direction])
                proba = np.array([self.h_labeler.abs_sizes[x] for x in candidates])
                anchor = candidates.pop(np.random.choice(np.arange(len(candidates)), p=proba / proba.sum()))
                self.candidates[direction] = set(candidates)
            else:
                anchor = next(self._components_iter)
                direction = 0

            if direction == 0:
                if anchor in self.extended_components[1]:
                    direction = -1
                elif anchor in self.extended_components[-1]:
                    direction = 1
                else:
                    break

            if direction != 0:
                if anchor not in self.extended_components[direction]:
                    break

        return anchor, direction

    def create_patches(self, n_patches, bar=True):
        """ Create fixed number of patches. They are extended from the largest horizontal components or from candidates
        from exsisted patches. At the end of the creation unit `connectivity_matrix` is created to join patches into
        large complex struces (trees of patches). """
        self.patches = {}
        self.patches_reverse = {}

        self.extended_components = {-1: set(), 1: set()}
        self.candidates = {-1: set(), 1: set()}
        self._components_iter = iter(self.h_labeler.sorted_labels)

        np.random.seed(42)

        for _ in tqdm.tqdm_notebook(range(n_patches), disable=(not bar)):
            try:
                anchor, direction = self.next_candidate()
            except StopIteration:
                break

            patch = FaultPatch(anchor, direction, self)
            if len(patch.components) == 1 and (direction == 0 or patch.up in self.extended_components[-direction]):
                continue
            if patch.up in self.patches:
                if patch.direction != self.patches[patch.up]:
                    continue
                else:
                    print("STRANGE!!!") # TODO: process

            self.patches[patch.up] = patch
            self.patches_reverse[patch.bottom] = patch

        self._labels_mapping = dict(enumerate(self.patches.keys()))
        self._labels_reverse = {v: k for k, v in self._labels_mapping.items()}
        self.connectivity_matrix = np.eye(len(self.patches))

        return self

    def make_children_tree(self, idx, depth, threshold):
        """ Get tree of children components which starts from one component of the fixed depth. """
        children = [idx]
        tree = {}
        for i in range(depth):
            children_ = set()
            for p in children:
                for c in self.patches[p].children:
                    if c in self.patches:
                        leaf = list(self.patches[p].components.values())[-1][0]
                        a, b, intersection = self.h_labeler.components_stats(c, leaf)
                        if intersection / min(a, b) >= threshold:
                            tree[c] = p
                            children_ |= {c}
            children = children_
        return tree

    def join_cycles(self, depth, threshold=0.9, bar=True):
        """ Find patches with two children and find common descendants. Such cycles in patches tree usually
        are around holes in fault surfaces. """
        cycles = []
        for idx in tqdm.tqdm_notebook(self.patches, disable=(not bar)):
            children = self.patches[idx].children
            for p1, p2 in combinations(children, 2):
                if p1 not in self.patches or p2 not in self.patches:
                    continue
                tree1 = self.make_children_tree(p1, depth, threshold)
                tree2 = self.make_children_tree(p2, depth, threshold)

                for item in set(tree1) & set(tree2):
                    if tree1[item] == tree2[item]:
                        continue
                    parent = tree1.get(item)
                    path1 = [item]
                    while parent is not None:
                        path1.append(parent)
                        parent = tree1.get(parent)

                    parent = tree2.get(item)
                    path2 = [item]
                    while parent is not None:
                        path2.append(parent)
                        parent = tree2.get(parent)

                    if len(set(path1) & set(path2)) == 1:
                        cycles.append(list(set([idx, *path1, *path2])))

            for cycle in cycles:
                idx = self._labels_reverse.get(cycle[0])
                if idx is not None:
                    for item in cycle[1:]:
                        idx_2 = self._labels_reverse[item]
                        self.connectivity_matrix[idx, idx_2] = 1
                        self.connectivity_matrix[idx_2, idx] = 1

        return self

    def merge_patches(self, thresholds, bar=True):
        """ Merge patches with large intersections. see :meth:`.FaultPatch.find_children_to_merge`. """
        for comp, patch in tqdm.tqdm_notebook(self.patches.items(), disable=(not bar)):
            for i in patch.find_children_to_merge(thresholds=thresholds):
                if i in self.patches:
                    idx = self._labels_reverse[comp]
                    idx_2 = self._labels_reverse[i]
                    self.connectivity_matrix[idx, idx_2] = 1
                    self.connectivity_matrix[idx_2, idx] = 1
        return self

    def make_faults(self, mode='h', depth_step=10, horizontal_step=40, merge_groups=True, threshold=0):
        """ Make fault points/sticks from groups of patches. """
        faults = []
        n_groups, groups = connected_components(self.connectivity_matrix)
        for idx in range(n_groups):
            group = [self._labels_mapping[item] for item in np.arange(len(self.patches))[groups == idx]]
            labels = [list(self.patches[patch].all_components) for patch in group]
            if merge_groups:
                labels = [sum(labels, [])]
            for group in labels:
                points = [self.h_labeler.component_points(i) for i in group]
                points_ = np.concatenate(points, axis=0)
                if np.sum(points_.ptp(axis=0)) > threshold:
                    if mode == 'v':
                        faults.append({'points': points_})
                    else:
                        sticks = sorted(points, key=lambda x: x[0, 2])
                        sticks = [stick[np.argsort(stick[:, self.axis])][::horizontal_step] for stick in sticks][::depth_step]
                        faults.append({'sticks': sticks})
        return faults


class FaultPatch:
    def __init__(self, anchor, direction, extractor):
        """ A sequence of horizontal components extended from anchor component.
        Each patch starts from anchor in one of the direction: up (-1) or bottom (+1).
        If zero, anchor is extended in both direction and then is merged into one patch.
        When we say "intersection" of two components on the sequential slides we mean intersection
        of their 2D masks (the word "contact" is more accurate).

        Anchor extension is iterative and stops in 3 cases:
            - in the intersection of the currently extended component with the next depth slide
              there are more then one component
            - the next slide has only one component in intersection but it has other component
              in intersection with the current slide
            - the next slide has only one component but it was already extended in that direction

        Components from intersection on the last step will be added to candidates fot the next fault
        patch anchors.

        Examples (for direction = 1):
                    parent                               ---------
                    anchor/up                        -----------------            |
                                                   --------------------           |  patch
                    bottom                        -----------------------         |
                    candidates/children          ---------         ---------
                                                ------                --------

                    ===========================================================================

                    anchor/up         |        -----------------
                                patch |      --------------------
                    bottom            |    -----------------------     ------------   candidate
                    candidate/child       ------------------------------------

                    ============================================================================

        Parameters
        ----------
        anchor : int
            Index of the components to extend
        direction : -1, 0 or 1
            Direction of the patch extension (up or down). 0 means extension in both directions.
        extractor : SurfacesExtractor

        """
        self.anchor = anchor
        self.extractor = extractor
        self.h_labeler = extractor.h_labeler
        self.direction = direction

        self.extend()

    def extend(self):
        """ Extend fault patch from anchor. """
        if self.direction == 0:
            patch_down = FaultPatch(self.anchor, 1, self.extractor)
            patch_up = FaultPatch(self.anchor, -1, self.extractor)
            self.components = {**patch_up.components, **patch_down.components}
            self.parents = patch_up.parents
            self.children = patch_down.children
        else:
            idx = self.anchor
            branch = {self.h_labeler.component_loc(idx): np.array([idx])}

            while True:
                neighbours = self.h_labeler.get_neighbours(idx, self.direction)
                neighbours_ = []
                if len(neighbours) != 1:
                    break
                if neighbours[0] in self.extractor.extended_components[self.direction]:
                    break

                neighbours_ = self.h_labeler.get_neighbours(neighbours[0], -self.direction)
                if len(neighbours_) > 1:
                    break

                branch[self.h_labeler.component_loc(neighbours[0])] = neighbours
                self.extractor.extended_components[1] |= {idx}
                self.extractor.extended_components[-1] |= {idx, neighbours[-1]}

                idx = neighbours[0]

            self.components = dict(sorted(branch.items(), key=lambda x: x[0]))

            self.parents = list(self.h_labeler.get_neighbours(self.anchor, -self.direction))
            self.children = list(neighbours)
            if self.direction == -1:
                self.parents, self.children = self.children, self.parents

            self.extractor.candidates[self.direction] |= set(neighbours)
            self.extractor.candidates[-self.direction] |= set(neighbours_)

        self.up = list(self.components.values())[0][0]
        self.bottom = list(self.components.values())[-1][0]

    def find_children_to_merge(self, thresholds=(0.5, 0.9)):
        """ Find children components that have large intersections with bottom components.

        Parameters
        ----------
        thresholds : tuple, optional
            Thresholds as a ratios of components sizes, by default (0.5, 0.9).
            For each pair of bottom component and child the first item is the ratio of the components intersection
            to the size of the smallest item in pair. The second is the ratio of the components intersection
            to the size of the largest item in pair.

        Returns
        -------
        list
            sorted by size list of children components filtered by thresholds.
        """
        merge = {}
        for leaf in self.children:
            if leaf in self.extractor.patches:
                comp = self.all_components[-1]
                A, B, intersection = self.extractor.h_labeler.components_stats(leaf, comp)
                if min(A, B) > 20 and intersection / (max(A, B)) >= thresholds[0] and intersection / (min(A, B)) >= thresholds[1]:
                    merge[leaf] = intersection / (max(A, B))
        return list(sorted(merge, key=lambda x: merge[x]))

    def to_fault(self):
        """ Make horizontal fault stciks from patch. """
        labels = np.concatenate(list(self.components.values())).astype(int)
        points = [self.h_labeler.component_points(i) for i in labels]
        sticks = sorted(points, key=lambda x: x[0, 2])
        return {'sticks': [stick[np.argsort(stick[:, 1])][::30] for stick in sticks]}

    def __repr__(self):
        return f"{self.parents} ---> {self.up} -> ... -> {self.bottom} ---> {self.children}"

    @property
    def all_components(self):
        """ All components included into patch. """
        return np.concatenate(list(self.components.values())).astype(int)
