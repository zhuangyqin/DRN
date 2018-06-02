# import unittest
# import deeplab._init_paths
# from deeplab.core.loader import TrainDataLoader
# from utils.PrefetchergroupIter import PrefetchergroupIter
# from utils.load_data import load_gt_segdb, merge_segdb
# from utils.create_logger import create_logger
# from deeplab.config.config import config, update_config
# import argparse
# import os
# import mxnet as mx
#
#
# #
# class TestPrefetcher(unittest.TestCase):
#
#     def setUp(self):
#         cfg = "./experiments/deeplab/cfgs/deeplab_test_symbol.yaml"
#
#         # update config
#         update_config(cfg)
#         logger, final_output_path = create_logger(config.output_path, cfg, config.dataset.image_set)
#         image_sets = [iset for iset in config.dataset.image_set.split('+')]
#         segdbs = [
#             load_gt_segdb(config.dataset.dataset, image_set, config.dataset.root_path, config.dataset.dataset_path,
#                           result_path=None, flip=config.TRAIN.FLIP)
#             for image_set in image_sets]
#         segdb = merge_segdb(segdbs)
#
#         ctx = [mx.gpu(3)]
#         input_batch_size =1
#         sym =None
#         self.train_data = TrainDataLoader(sym, segdb, config, batch_size=input_batch_size, crop_height=config.TRAIN.CROP_HEIGHT, crop_width=config.TRAIN.CROP_WIDTH,
#                                  shuffle=config.TRAIN.SHUFFLE, ctx=ctx)
#
#
#     def test_loader(self):
#
#         if not isinstance(self.train_data,PrefetchergroupIter):
#             self.train_data = PrefetchergroupIter(self.train_data)
#
#         for batch in self.train_data:
#             print batch.index
#
#
# if __name__ == '__main__':
#     unittest.main()
