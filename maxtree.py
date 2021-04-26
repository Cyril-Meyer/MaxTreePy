"""
Max-Tree computation and filters.

This work use some of the MaMPy original code https://github.com/Yt-trium/MaMPy
Authors:
C. Meyer (MaMPy & MaxTreePy)
K. Masson (MaMPy)
"""

import numpy as np
import numba
from numba import jit
import math
from collections import namedtuple


# Named Tuple for the max-tree structure
MaxTreeStructure = namedtuple("MaxTreeStructure", ["parent", "S"])


@jit(nopython=True)
def find_pixel_parent(parents, index):
    """
    Given an image containing pixel's parent and a pixel id, returns the id of its parent id.
    The parent is also named as root. A pixel is the root of itself if parents[index] == index.
    """

    root = parents[index]

    # Assign the root of the given pixel to the root of its parent.
    if root != index:
        parents[index] = find_pixel_parent(parents, root)
        return parents[index]
    else:
        return root


@jit(nopython=True)
def canonize(image, parents, nodes_order):
    """
    Makes sure all nodes of a max tree are valid.
    """
    for pi in nodes_order:
        root = parents[pi]

        if image[root] == image[parents[root]]:
            parents[pi] = parents[root]


@jit(nopython=True)
def get_neighbors_2d(connectivity, shape, pi, size):
    """
    Return the indexes of the neighbors with the given connectivity and position
    """
    neighbors = []

    # 4-connectivity
    pi_row = math.floor(pi / shape[1])
    pi_top = pi - shape[1]
    pi_bot = pi + shape[1]
    pi_lft = pi - 1
    pi_rgt = pi + 1

    if pi_top >= 0:
        neighbors.append(pi_top)
    if pi_bot < size:
        neighbors.append(pi_bot)
    if math.floor(pi_lft / shape[1]) == pi_row:
        neighbors.append(pi_lft)
    if math.floor(pi_rgt / shape[1]) == pi_row:
        neighbors.append(pi_rgt)

    # 8-connectivity
    if connectivity == 8:
        pi_top_lef = pi_top - 1
        pi_top_rgt = pi_top + 1
        pi_bot_lef = pi_bot - 1
        pi_bot_rgt = pi_bot + 1

        if math.floor(pi_top_lef / shape[1]) == pi_row - 1:
            neighbors.append(pi_top_lef)
        if math.floor(pi_top_rgt / shape[1]) == pi_row - 1:
            neighbors.append(pi_top_rgt)
        if math.floor(pi_bot_lef / shape[1]) == pi_row + 1:
            neighbors.append(pi_bot_lef)
        if math.floor(pi_bot_rgt / shape[1]) == pi_row + 1:
            neighbors.append(pi_bot_rgt)

    return neighbors


@jit(nopython=True)
def get_neighbors_3d(connectivity, shape, pi, size):
    neighbors = []

    # 6-connectivity
    pi_row = math.floor(pi / shape[1])
    pi_slc = math.floor(pi / shape[2])

    pi_far = pi + shape[2]
    pi_cls = pi - shape[2]
    pi_top = pi - shape[1]
    pi_bot = pi + shape[1]
    pi_lft = pi - 1
    pi_rgt = pi + 1

    if pi_top >= 0:
        neighbors.append(pi_top)
    if pi_bot < size:
        neighbors.append(pi_bot)
    if math.floor(pi_lft / shape[1]) == pi_row:
        neighbors.append(pi_lft)
    if math.floor(pi_rgt / shape[1]) == pi_row:
        neighbors.append(pi_rgt)
    if math.floor(pi_far / shape[2]) == pi_slc:
        neighbors.append(pi_far)
    if math.floor(pi_cls / shape[2]) == pi_slc:
        neighbors.append(pi_cls)

    # 18-connectivity
    if (connectivity >= 18):
        raise NotImplementedError
    # 26-connectivity
    if (connectivity == 26):
        raise NotImplementedError

    return neighbors


@jit(nopython=True)
def get_neighbors(connectivity, shape, pi, size):
    """
    :param connectivity: connectivity of the maxtree : acceptable value : 2D 4/8 - 3D 6/18/26 (default 4 and 6)
    :param shape: shape of the input image
    :param pi: current index in the flatten input image
    :return: indexes of the neighbors with the given connectivity and position
    """

    # 2D
    if len(shape) == 2:
        return get_neighbors_2d(connectivity, shape, pi, size)

    # 3D
    if len(shape) == 3:
        # test needed
        raise NotImplementedError
        # return get_neighbors_3d(connectivity, shape, pi, size)


@jit(nopython=True)
def maxtree_berger_union_by_rank(input, connectivity):
    """
    Union-find with union-by-rank based max-tree algorithm.
    Algorithm 3 in the paper [1].
    :param input: numpy ndarray of a single channel image
    :param connectivity: connectivity of the maxtree : acceptable value : 2D 4/8 - 3D 6/18/26 (default 4 and 6)
    :return: the maxtree of the image (parent and S vector pair)
    """

    input_flat = input.flatten()
    resolution = input_flat.size

    # Unique value telling if a pixel is defined in the max tree or not.
    undefined_node = resolution + 2

    # We generate an extra vector of pixels that order nodes downard.
    # This vector allow to traverse the tree both upward and downard
    # without having to sort childrens of each node.
    # Initially, we sort pixel by increasing value and add indices in it.
    sorted_pixels = input_flat.argsort()

    # We store in the parent node of each pixel in an image.
    # To do so we use the index of the pixel (x + y * width).
    parents = np.full(
        resolution,
        fill_value=undefined_node,
        dtype=np.uint32)

    ranks = np.full(
        resolution,
        fill_value=0,
        dtype=np.uint32)

    reprs = np.full(
        resolution,
        fill_value=0,
        dtype=np.uint32)

    # zparents make root finding much faster.
    zparents = parents.copy()

    # We go through sorted pixels in the reverse order.
    for pi in sorted_pixels[::-1]:
        # Make a node.
        # By default, a pixel is its own parent.
        parents[pi] = pi
        zparents[pi] = pi
        ranks[pi] = 0
        reprs[pi] = pi

        zp = pi

        neighbors = get_neighbors(connectivity, input.shape, pi, input.size)

        # Filter neighbors.
        neighbors = [n for n in neighbors if parents[n] != undefined_node]

        # Go through neighbors.
        for nei_pi in neighbors:
            zn = find_pixel_parent(zparents, nei_pi)

            if zn != zp:
                parents[reprs[zn]] = pi

                if ranks[zp] < ranks[zn]:
                    # Swap them.
                    zp, zn = zn, zp

                # Merge sets.
                zparents[zn] = zp
                reprs[zp] = pi

                if ranks[zp] == ranks[zn]:
                    ranks[zp] += 1

    canonize(input_flat, parents, sorted_pixels)

    return MaxTreeStructure(parents, sorted_pixels)


@jit(nopython=True)
def maxtree_berger_union_by_rank_level_compression(input, connectivity):
    """
    Union-find with union-by-rank based max-tree algorithm with level compression.
    Algorithm 5 in the paper [1].
    :param input: numpy ndarray of a single channel image
    :param connectivity: connectivity of the maxtree : acceptable value : 2D 4/8 - 3D 6/18/26 (default 4 and 6)
    :return: the maxtree of the image (parent and S vector pair)
    """

    input_flat = input.flatten()
    resolution = input_flat.size

    # Unique value telling if a pixel is defined in the max tree or not.
    undefined_node = resolution + 2

    # We generate an extra vector of pixels that order nodes downard.
    # This vector allow to traverse the tree both upward and downard
    # without having to sort childrens of each node.
    # Initially, we sort pixel by increasing value and add indices in it.
    sorted_pixels = input_flat.argsort()

    # We store in the parent node of each pixel in an image.
    # To do so we use the index of the pixel (x + y * width).
    parents = np.full(
        resolution,
        fill_value=undefined_node,
        dtype=np.uint32)

    # zparents make root finding much faster.
    zparents = parents.copy()

    j = resolution - 1

    # We go through sorted pixels in the reverse order.
    for pi in sorted_pixels[::-1]:
        # Make a node.
        # By default, a pixel is its own parent.
        parents[pi] = pi
        zparents[pi] = pi

        zp = pi

        neighbors = get_neighbors(connectivity, input.shape, pi, input.size)

        # Filter neighbors.
        neighbors = [n for n in neighbors if parents[n] != undefined_node]

        # Go through neighbors.
        for nei_pi in neighbors:
            zn = find_pixel_parent(zparents, nei_pi)

            if zn != zp:
                if input_flat[zp] == input_flat[zn]:
                    zp, zn = zn, zp

                # Merge sets.
                zparents[zn] = zp
                parents[zn] = zp

                sorted_pixels[j] = zn
                j -= 1

    canonize(input_flat, parents, sorted_pixels)

    return MaxTreeStructure(parents, sorted_pixels)


@jit(nopython=True)
def maxtree(input, connectivity=None):
    """
    Compute the max-tree of a 2D/3D image
    :param input: numpy ndarray of a single channel image
                  Good practice : use a numpy fixed-size dtype (https://www.numpy.org/devdocs/user/basics.types.html)
    :param connectivity: connectivity of the maxtree : acceptable value : 2D 4/8 - 3D 6/18/26 (default 4 and 6)
    """
    # Check input
    if input.ndim not in [2, 3]:
        raise ValueError("Input image is not a 2D or 3D array")

    if input.ndim == 2:
        if connectivity is None or connectivity not in [4, 8]:
            connectivity = 4
    if input.ndim == 3:
        if connectivity is None or connectivity not in [6, 18, 26]:
            connectivity = 6

    return maxtree_berger_union_by_rank(input, connectivity)
    # return maxtree_berger_union_by_rank_level_compression(input, connectivity)


@jit(nopython=True)
def direct_filter(maxtree_p_s, input, attribute, λ):
    """
    The parameters order follows the order given in the article [1]
    :param maxtree_p_s: the maxtree of the image (parent and S vector pair)
    :param input: numpy ndarray of a single channel image
    :param attribute: the attribute associated with the maxtree
    :param λ: attribute threashold
    :return: the filtered image
    """

    ima = input.flatten()

    out = np.full(
        ima.shape,
        fill_value=0,
        dtype=input.dtype)

    proot = maxtree_p_s.S[0]

    if attribute[proot] < λ:
        out[proot] = 0
    else:
        out[proot] = ima[proot]

    for p in maxtree_p_s.S:
        q = maxtree_p_s.parent[p]

        if ima[q] == ima[p]:
            out[p] = out[q]     # p not canonical
        elif attribute[p] < λ:
            out[p] = out[q]     # Criterion failed
        else:
            out[p] = ima[p]     # Criterion pass

    return out.reshape(input.shape)


@jit(nopython=True)
def get_area_attribute(input, maxtree_p_s):
    # Compute area attribute
    area_attribute = np.full(input.size,
                             fill_value=1,
                             dtype=np.uint32)

    # Everything except the first item, reversed
    # > np.arange(8)[:0:-1]
    # array([7, 6, 5, 4, 3, 2, 1])
    for p in maxtree_p_s.S[:0:-1]:
        q = maxtree_p_s.parent[p]
        area_attribute[q] += area_attribute[p]

    return area_attribute


@jit(nopython=True)
def area_filter(input, threshold, maxtree_p_s=None):
    """
    :param input: numpy ndarray of a single channel image
    :param threshold: threshold of the filter (minimum area to keep)
    :param maxtree_p_s: the maxtree of the image (parent and S vector pair)
    :return: numpy ndarray of the image
    """
    # Check input
    if input.ndim not in [2, 3]:
        raise ValueError("Input image is not a 2D or 3D array")

    if threshold < 1:
        raise ValueError("Threshold less than 1")

    if maxtree_p_s is None:
        maxtree_p_s = maxtree(input)

    if maxtree_p_s.parent.size != maxtree_p_s.S.size:
        raise ValueError("Invalid max-tree")

    if maxtree_p_s.S.size != input.size:
        raise ValueError("Image and max-tree doesn't match")

    # Compute area attribute
    area_attribute = get_area_attribute(input, maxtree_p_s)

    # Apply Filter
    return direct_filter(maxtree_p_s, input, area_attribute, threshold)


@jit(nopython=True)
def get_contrast_attribute(input, maxtree_p_s):
    # Compute contrast attribute
    pixel_values = input.flatten()
    contrast_attribute = np.full(input.size,
                             fill_value=0,
                             dtype=np.uint16)

    # Everything except the first item, reversed
    # > np.arange(8)[:0:-1]
    # array([7, 6, 5, 4, 3, 2, 1])
    for p in maxtree_p_s.S[:0:-1]:
        q = maxtree_p_s.parent[p]
        contrast_attribute[q] = max(contrast_attribute[q], pixel_values[p] - pixel_values[q] + contrast_attribute[p])

    return contrast_attribute


@jit(nopython=True)
def contrast_filter(input, threshold, maxtree_p_s=None):
    """
    :param input: numpy ndarray of a single channel image
    :param threshold: threshold of the filter (minimum contrast to keep)
    :param maxtree_p_s: the maxtree of the image (parent and S vector pair)
    :return: numpy ndarray of the image
    """
    # Check input
    if input.ndim not in [2, 3]:
        raise ValueError("Input image is not a 2D or 3D array")

    if threshold < 1:
        raise ValueError("Threshold less than 1")

    if maxtree_p_s is None:
        maxtree_p_s = maxtree(input)

    if maxtree_p_s.parent.size != maxtree_p_s.S.size:
        raise ValueError("Invalid max-tree")

    if maxtree_p_s.S.size != input.size:
        raise ValueError("Image and max-tree doesn't match")

    # Compute contrast attribute
    contrast_attribute = get_contrast_attribute(input, maxtree_p_s)

    # Apply Filter
    return direct_filter(maxtree_p_s, input, contrast_attribute, threshold)


@jit(nopython=True)
def attribute_map(maxtree_p_s, input, attribute, dtype=np.float64):
    ima = input.flatten()

    out = np.full(
        ima.shape,
        fill_value=0,
        dtype=dtype)

    proot = maxtree_p_s.S[0]

    out[proot] = attribute[proot]
    for p in maxtree_p_s.S:
        q = maxtree_p_s.parent[p]
        if ima[q] == ima[p]:
            out[p] = attribute[q]
        else:
            out[p] = attribute[p]

    # out = attribute

    return out.reshape(input.shape)


def main():
    import imageio
    import matplotlib.pyplot as plt

    # image_input = imageio.imread(uri="examples/images/circuit.png", as_gray=True).astype(dtype=np.uint8)
    # image_input = imageio.imread(uri="examples/images/i3_slice68.png", as_gray=True).astype(dtype=np.uint8)
    image_input = imageio.imread(uri="examples/images/test.png", as_gray=True).astype(dtype=np.uint8)
    # image_input = image_input / image_input.max()
    # image_output = area_filter(image_input, 500)
    # image_output = contrast_filter(image_input, 30)

    mt = maxtree(image_input, connectivity=4)
    attr_area = get_area_attribute(image_input, mt)
    attr_cont = get_contrast_attribute(image_input, mt)
    image_output_1 = attribute_map(mt, image_input, attr_area)
    image_output_2 = attribute_map(mt, image_input, attr_cont)

    plt.imshow(image_input, cmap="gray")
    plt.show()
    plt.imshow(image_output_1, cmap="gray")
    plt.show()
    plt.imshow(image_output_2, cmap="gray")
    plt.show()

    print(image_input.min(), image_input.max(), image_input.max() - image_input.min())
    print(image_output_1.min(), image_output_1.max())
    print(image_output_2.min(), image_output_2.max())


if __name__ == "__main__":
    main()
