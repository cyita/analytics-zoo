from bigdl.nn.criterion import *


class MultiBoxCriterion(Criterion):
    def __init__(self, loc_weight=1.0, n_classes=21,
                 share_location=True,
                 overlap_threshold=0.5,
                 bg_label_ind=0,
                 use_difficult_gt=True,
                 neg_pos_ratio=3.0,
                 neg_overlap=0.5, bigdl_type="float"):
        super(MultiBoxCriterion, self).__init__(None, bigdl_type, loc_weight, n_classes,
                                                share_location, overlap_threshold, bg_label_ind,
                                                use_difficult_gt, neg_pos_ratio, neg_overlap)
