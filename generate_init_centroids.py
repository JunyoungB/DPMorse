import random, copy, math, sys, os
import numpy as np
import time, shutil, itertools

MAX_SINGLE_TRAILED_TIMES = 10
MAX_BATCH_TRAILED_TIMES = 20

def euclidean_distance(x,y):
    return math.sqrt(np.dot(x-y,(x-y).conj()))

def randomly_generate_init_centroids(k, min_max, dim):
    centroids_list = []
    for i in range(k):    
        curr_centroid = np.array([random.uniform(min_max[f][0], min_max[f][1]) for f in range(dim)], np.float)
        centroids_list.append(curr_centroid)
    
    return centroids_list


def check_closeness(centroids_list, centroid, radius):

    check_result = True
    for cen in centroids_list:
        dist = euclidean_distance(cen, centroid)
        if dist < 2*radius:
            check_result = False
            break

    return check_result

def propose_smart_single_init_centroid(min_max, radius, proposed_centroids_list):
    dim = len(min_max)
    trail_count = 0

    vertex_list = []

    for v in itertools.product(*min_max):
        vertex_list.append(np.array(v, float))

    while trail_count < MAX_SINGLE_TRAILED_TIMES:
        proposed_centroid = np.array([random.uniform(min_max[f][0], min_max[f][1]) for f in range(dim)], np.float)

        closeness_to_centroids = check_closeness(proposed_centroids_list, proposed_centroid, radius)
        closeness_to_vertices = check_closeness(vertex_list, proposed_centroid, radius * 0.5)
        if closeness_to_centroids == True and closeness_to_vertices == True:#
            return trail_count, proposed_centroid
        else:
            trail_count += 1

    return trail_count, None


def propose_smart_init_centroids_list(num_clusters, min_max, radius):

    proposed_centroids_list = []
    batch_trail_count = 0

    while batch_trail_count < MAX_BATCH_TRAILED_TIMES and len(proposed_centroids_list) < num_clusters:
        trail_count, proposed_centroid = propose_smart_single_init_centroid(min_max, radius, proposed_centroids_list)
        if proposed_centroid is None:
            batch_trail_count += 1
        else:
            proposed_centroids_list.append(proposed_centroid)

    if batch_trail_count >= MAX_BATCH_TRAILED_TIMES:
        return batch_trail_count, None
    else:
        return batch_trail_count, proposed_centroids_list

def do_smart_generate_init_centroids_binary_search(num_clusters, min_max, radius_lo, radius_hi):

    proposed_centroid_list = None
    while radius_lo < radius_hi:
        radius_mid = (radius_lo + radius_hi) * 0.5
        trail_count, proposed_centroid_list = propose_smart_init_centroids_list(num_clusters, min_max, radius_mid)

        if trail_count < 3:
            radius_lo = radius_mid
        elif trail_count > 0.8 * MAX_BATCH_TRAILED_TIMES:
            radius_hi = radius_mid
        else:
            break

    return proposed_centroid_list, radius_mid

def smart_generate_init_centroids_binary_search(num_clusters, min_max):

    north_pole = np.array([min_max[f][1] for f in range(len(min_max))])
    south_pole = np.array([min_max[f][0] for f in range(len(min_max))])
    max_radius = 0.5 * math.sqrt(((north_pole - south_pole) ** 2).sum())

    radius_lo = 0.0
    radius_hi = max_radius

    centroids_list, radius = do_smart_generate_init_centroids_binary_search(num_clusters, min_max, radius_lo, radius_hi)

    return centroids_list, radius

def batch_smartly_generate_init_centroids_binary_search(k, min_max):

    cl_list = []
    radius_list = []

    for i in range(0, 1):
        cl, radius = smart_generate_init_centroids_binary_search(k, min_max)
        cl_list.append(cl)
        radius_list.append(radius)

    return cl_list[np.argmax(radius_list)], radius_list[np.argmax(radius_list)]