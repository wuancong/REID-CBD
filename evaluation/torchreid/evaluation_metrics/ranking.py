from __future__ import absolute_import
import numpy as np
from ..utils import to_numpy


def _unique_sample(ids_dict, num):
    mask = np.zeros(num, dtype=np.bool)
    for _, indices in ids_dict.items():
        i = np.random.choice(indices)
        mask[i] = True
    return mask


def cmc(distmat, query_ids, gallery_ids,
        query_cams, gallery_cams, topk=100):
    distmat = to_numpy(distmat)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute CMC for each query
    ret = np.zeros(topk)
    num_valid_queries = 0
    for i in range(distmat.shape[0]):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        if not np.any(matches[i, valid]):
            continue
        num_valid_queries += 1
        index = np.nonzero(matches[i, valid])[0]
        if index[0] < topk:
            ret[index[0]] += 1
    if num_valid_queries == 0:
        raise RuntimeError("No valid query")
    return ret.cumsum() / num_valid_queries


def mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams):
    # Ensure numpy array
    distmat = to_numpy(distmat)
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)

    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    aps = []

    for i in range(distmat.shape[0]):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        y_true = matches[i, valid]
        if not np.any(y_true):
            continue
        aps.append(np.mean((np.arange(y_true.sum())+1) / (np.arange(len(y_true))[y_true]+1)))

    return np.mean(aps)
