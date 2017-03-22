# import the necessary packages

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.future import graph
from skimage import io, filters, color
import matplotlib.pyplot as plt
import argparse
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

# load the image and convert it to a floating point data type
img = img_as_float(io.imread(args["image"]))

def weight_boundary(graph, src, dst, n):
    default = {'weight': 0.0, 'count': 0}
    count_src = graph[src].get(n, default)['count']
    count_dst = graph[dst].get(n, default)['count']

    weight_src = graph[src].get(n, default)['weight']
    weight_dst = graph[dst].get(n, default)['weight']

    count = count_src + count_dst
    return {
        'count': count,
        'weight': (count_src * weight_src + count_dst * weight_dst)/count
    }

def merge_boundary(graph, src, dst):
    pass

edges = filters.sobel(color.rgb2gray(img))

''' Increase the segment number to get finer slicing result '''
labels = slic(img, compactness=30, n_segments=400, sigma=5)

g = graph.rag_boundary(labels, edges)

graph.show_rag(labels, g, img)
plt.title('Initial RAG')

''' Lower the thresh value to get finer merge result (more approximate to the slicing grid) '''
labels2 = graph.merge_hierarchical(labels, g, thresh=0.0005, rag_copy=False,
                                   in_place_merge=True,
                                   merge_func=merge_boundary,
                                   weight_func=weight_boundary)

graph.show_rag(labels, g, img)
plt.title('RAG after hierarchical merging')

plt.figure()
out = color.label2rgb(labels2, img, kind='avg')
plt.imshow(out)
plt.title('Final segmentation')

plt.show()
