from __future__ import print_function, absolute_import
import time
from collections import OrderedDict
import torch
import torch.nn.functional as F
from .evaluation_metrics import cmc, mean_ap
from .feature_extraction import extract_cnn_feature
from .utils.meters import AverageMeter

torch.multiprocessing.set_sharing_strategy('file_system')


def extract_features(model, data_loader, print_freq=1000):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, item in enumerate(data_loader):
            imgs, fnames, pids = item[0], item[1], item[2]
            data_time.update(time.time() - end)

            outputs = extract_cnn_feature(model, imgs)
            for fname, output, pid in zip(fnames, outputs, pids):
                features[fname] = output
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features, labels


def pairwise_distance(query=None, gallery=None, metric=None):
    x = query
    y = gallery
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    if metric is not None:
        x = metric.transform(x)
        y = metric.transform(y)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
             torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m = torch.addmm(dist_m, x, y.t(), beta=1, alpha=-2)
    return dist_m


def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10, 20)):
    if query is not None and gallery is not None:
        query_ids = [item[1] for item in query]
        gallery_ids = [item[1] for item in gallery]
        query_cams = [item[2] for item in query]
        gallery_cams = [item[2] for item in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))
    cmc_scores = cmc(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('CMC Scores:')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'
              .format(k, cmc_scores[k - 1]))
    return cmc_scores, mAP


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, query, gallery, query_data, gallery_data, metric=None, cmc_flag=False):
        features_query, _ = extract_features(self.model, query)
        features_gallery, _ = extract_features(self.model, gallery)

        features_query = torch.cat([features_query[f].unsqueeze(0) for f, _, _, _ in query_data], 0)
        features_query = F.normalize(features_query, p=2, dim=1)
        features_gallery = torch.cat([features_gallery[f].unsqueeze(0) for f, _, _, _ in gallery_data], 0)
        features_gallery = F.normalize(features_gallery, p=2, dim=1)

        distmat = pairwise_distance(features_query, features_gallery, metric=metric)
        results = evaluate_all(distmat, query=query_data, gallery=gallery_data)
        return results
