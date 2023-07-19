import numpy as np
def eval_cmc_map(dist, gallery_labels, probe_labels, gallery_views=None, probe_views=None):
    """
    :param dist: 2-d np array, shape=(num_gallery, num_probe), distance matrix.
    :param gallery_labels: np array, shape=(num_gallery,)
    :param probe_labels:
    :param gallery_views: np array, shape=(num_gallery,) if specified, for any probe image,
    the gallery correct matches from the same view are ignored.
    :param probe_views: must be specified if gallery_views are specified.
    :return:
    CMC: np array, shape=(num_gallery,). Measured by percentage
    MAP: np array, shape=(1,). Measured by percentage
    
    sample:
    cmc, map = eval_cmc_map(distmat.numpy().T, np.array(gallery_ids), np.array(query_ids),
                            np.array(gallery_cams), np.array(query_cams))
    """
    is_view_sensitive = False
    num_gallery = gallery_labels.shape[0]
    num_probe = probe_labels.shape[0]
    if gallery_views is not None or probe_views is not None:
        assert gallery_views is not None and probe_views is not None, \
            'gallery_views and probe_views must be specified together. \n'
        is_view_sensitive = True
    cmc = np.zeros((num_gallery, num_probe))
    ap = np.zeros((num_probe,))
    for i in range(num_probe):
        cmc_ = np.zeros((num_gallery,))
        dist_ = dist[:, i]
        probe_label = probe_labels[i]
        gallery_labels_ = gallery_labels
        if is_view_sensitive:
            probe_view = probe_views[i]
            is_from_same_view = gallery_views == probe_view
            is_correct = gallery_labels == probe_label
            should_be_excluded = is_from_same_view & is_correct
            dist_ = dist_[~should_be_excluded]
            gallery_labels_ = gallery_labels_[~should_be_excluded]
        ranking_list = np.argsort(dist_)
        inference_list = gallery_labels_[ranking_list]
        positions_correct_tuple = np.nonzero(probe_label == inference_list)
        positions_correct = positions_correct_tuple[0]
        pos_first_correct = positions_correct[0]
        cmc_[pos_first_correct:] = 1
        cmc[:, i] = cmc_

        num_correct = positions_correct.shape[0]
        for j in range(num_correct):
            last_precision = float(j) / float(positions_correct[j]) if j != 0 else 1.0
            current_precision = float(j+1) / float(positions_correct[j]+1)
            ap[i] += (last_precision + current_precision) / 2.0 / float(num_correct)

    CMC = np.mean(cmc, axis=1)
    MAP = np.mean(ap)
    return CMC, MAP