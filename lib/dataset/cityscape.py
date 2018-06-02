# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Zheng Zhang
# --------------------------------------------------------
import cv2
import cPickle
import os
import numpy as np
import itertools

from imdb import IMDB
from PIL import Image
from collections import namedtuple


class CityScape(IMDB):

    def __init__(self, image_set, root_path, dataset_path, result_path=None):
        """
        fill basic information to initialize imdb
        :param image_set: leftImg8bit_train, etc
        :param root_path: 'selective_search_data' and 'cache'
        :param dataset_path: data and results
        :return: imdb object
        """
        image_set_main_folder, image_set_sub_folder= image_set.split('_', 1)
        super(CityScape, self).__init__('cityscape', image_set, root_path, dataset_path, result_path)  # set self.name

        self.image_set_main_folder = image_set_main_folder
        self.image_set_sub_folder = image_set_sub_folder
        self.root_path = root_path
        self.data_path = dataset_path
        self.num_classes = 19
        self.image_set_index = self.load_image_set_index()
        self.num_images = len(self.image_set_index)
        self.has_label = True
        if image_set.endswith('test'):
            self.has_label =False
        print 'num_images', self.num_images

        self.config = {'comp_id': 'comp4',
                       'use_diff': False,
                       'min_size': 2}

    def load_image_set_index(self):
        """
        find out which indexes correspond to given image set
        :return: the indexes of given image set
        """

        #Collection all subfolders
        image_set_main_folder_path = os.path.join(self.data_path, self.image_set_main_folder, self.image_set_sub_folder)
        image_name_set = [filename for parent, dirname, filename in os.walk(image_set_main_folder_path)]
        image_name_set = list(itertools.chain.from_iterable(image_name_set))
        index_set = ['' for x in range(len(image_name_set))]
        valid_index_count = 0
        for i, image_name in enumerate(image_name_set):
            splited_name_set = image_name.split('_')
            ext_split = splited_name_set[len(splited_name_set) - 1].split('.')
            ext = ext_split[len(ext_split)-1]
            if splited_name_set[len(splited_name_set) - 1] != 'flip.png' and ext == 'png':
                index_set[valid_index_count] = "_".join(splited_name_set[:len(splited_name_set)-1])
                valid_index_count += 1

        return index_set[:valid_index_count]

    def image_path_from_index(self, index):
        """
        find the image path from given index
        :param index: the given index
        :return: the image path
        """
        index_folder = index.split('_')[0]
        image_file = os.path.join(self.data_path, self.image_set_main_folder, self.image_set_sub_folder, index_folder, index + '_' + self.image_set_main_folder + '.png')
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def annotation_path_from_index(self, index):
        """
        find the gt path from given index
        :param index: the given index
        :return: the image path
        """
        index_folder = index.split('_')[0]
        if self.image_set.endswith('extra'):
            label_folder = "gtCoarse"
            label_filename = index + '_gtCoarse_labelIds.png'
        else:
            label_folder = "gtFine"
            label_filename = index + '_gtFine_labelIds.png'

        image_file = os.path.join(self.data_path, label_folder, self.image_set_sub_folder, index_folder,label_filename)
        if self.image_set_sub_folder == 'test':
            image_file = None
        else:
            assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def ID2trainID(self):

        Label = namedtuple('Label', [
            'name',
            'id',
            'trainId',
            'category',
            'categoryId',
            'hasInstances',
            'ignoreInEval',
            'color',
        ])

        labels = [
            #       name       id    trainId   category            catId     hasInstances   ignoreInEval   color
            Label('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
            Label('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
            Label('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
            Label('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
            Label('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
            Label('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
            Label('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
            Label('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
            Label('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
            Label('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
            Label('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
            Label('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
            Label('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
            Label('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
            Label('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
            Label('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
            Label('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
            Label('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
            Label('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
            Label('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
            Label('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
            Label('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
            Label('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
            Label('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
            Label('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
            Label('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
            Label('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
            Label('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
            Label('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
            Label('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
            Label('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
            Label('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
            Label('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
            Label('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
            Label('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
        ]

        ID2trainID_dic = {label.id: label for label in labels}

        ID2trainID = 255 * np.ones((256,))
        for l in ID2trainID_dic:
            if l in (-1, 255):
                continue
            ID2trainID[l] = ID2trainID_dic[l].trainId
        return ID2trainID.astype(np.int32)

    def trainID2ID(self):
        # a label and all meta information
        Label = namedtuple('Label', [

            'name',
            'id',
            'trainId',
            'category',
            'categoryId',
            'hasInstances',
            'ignoreInEval',
            'color',
        ])

        labels = [
            #       name       id    trainId   category            catId     hasInstances   ignoreInEval   color
            Label('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
            Label('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
            Label('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
            Label('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
            Label('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
            Label('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
            Label('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
            Label('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
            Label('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
            Label('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
            Label('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
            Label('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
            Label('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
            Label('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
            Label('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
            Label('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
            Label('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
            Label('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
            Label('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
            Label('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
            Label('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
            Label('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
            Label('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
            Label('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
            Label('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
            Label('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
            Label('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
            Label('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
            Label('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
            Label('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
            Label('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
            Label('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
            Label('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
            Label('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
            Label('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
        ]

        trainID2ID_dic = {label.trainId: label for label in labels}

        # for train_ID to the ID
        trainID2ID = 255 * np.ones((256,))
        for l in trainID2ID_dic:
            if l in (-1, 255):
                continue
            trainID2ID[l] = trainID2ID_dic[l].id

        return trainID2ID.astype(np.int32)

    def sequence_path_from_index(self,index,sequence_length,sequence_length_range):

        index_folder = index.split('_')[0]
        second_id = index.split('_')[1]
        image_file_list = list()
        frame_id_ref = int(index.split('_')[2])
        relative_time_list = np.random.choice(np.arange(1,sequence_length_range),replace=False,size=sequence_length-1)
        relative_time_list.sort()
        relative_time_list=np.insert(relative_time_list,0,0)
        for id in relative_time_list:
            id  = frame_id_ref - id
            frame_id = '%06d'%(id)
            image_file = os.path.join(self.data_path, 'leftImg8bit_sequence', self.image_set_sub_folder, index_folder,
                                      index_folder+'_'+second_id+'_'+frame_id + '_' + self.image_set_main_folder + '.png')
            assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
            image_file_list.insert(0, image_file)
        return image_file_list


    def load_sequence_segdb_from_index(self,index,sequence_length,sequence_length_range):

        seg_rec = dict()
        seg_rec['image'] = self.image_path_from_index(index)

        seg_rec['sequence'] =self.sequence_path_from_index(index,sequence_length,sequence_length_range)

        im = cv2.imread(seg_rec['image'])
        size = im.shape

        seg_rec['height'] = size[0]
        seg_rec['width'] = size[1]

        seg_rec['seg_cls_path'] = self.annotation_path_from_index(index)
        seg_rec['flipped'] = False
        seg_rec['ID2trainID'] = self.ID2trainID()
        #
        return seg_rec

    def load_segdb_from_index(self, index):
        """
        load segdb from given index
        :param index: given index
        :return: segdb
        """
        seg_rec = dict()
        seg_rec['image'] = self.image_path_from_index(index)


        im = cv2.imread(seg_rec['image'])
        size = im.shape

        seg_rec['height'] = size[0]
        seg_rec['width'] = size[1]
        if self.has_label:
            seg_rec['seg_cls_path'] = self.annotation_path_from_index(index)
        seg_rec['flipped'] = False
        seg_rec['ID2trainID'] = self.ID2trainID().astype(np.uint8)
        #
        return seg_rec

    def gt_segdb(self):
        """
        return ground truth image regions database
        :return: imdb[image_index]['', 'flipped']
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_segdb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                segdb = cPickle.load(fid)
            print '{} gt segdb loaded from {}'.format(self.name, cache_file)
            return segdb

        gt_segdb = [self.load_segdb_from_index(index) for index in self.image_set_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_segdb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt segdb to {}'.format(cache_file)

        return gt_segdb

    def gt_sequence_segdb(self,sequence_length,sequence_length_range):
        sequence_length_str = "%02d"%sequence_length
        cache_file = os.path.join(self.cache_path, self.name + '_gt_sequence_'+sequence_length_str+'_segdb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                segdb = cPickle.load(fid)
            print '{} gt segdb loaded from {}'.format(self.name, cache_file)
            return segdb

        gt_sequence_segdb = [self.load_sequence_segdb_from_index(index,sequence_length,sequence_length_range) for index in self.image_set_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_sequence_segdb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt segdb to {}'.format(cache_file)

        return gt_sequence_segdb

    def getpallete(self):
        """
        this function is to get the colormap for visualizing the segmentation mask
        :param num_cls: the number of visulized class
        :return: the pallete
        """
        # n = num_cls
        pallete_raw = np.zeros((256, 3)).astype('uint8')
        # pallete = np.zeros((n, 3)).astype('uint8')

        pallete_raw[5, :] =  [111,  74,   0]
        pallete_raw[6, :] =  [ 81,   0,  81]
        pallete_raw[7, :] =  [128,  64, 128]
        pallete_raw[8, :] =  [244,  35, 232]
        pallete_raw[9, :] =  [250, 170, 160]
        pallete_raw[10, :] = [230, 150, 140]
        pallete_raw[11, :] = [ 70,  70,  70]
        pallete_raw[12, :] = [102, 102, 156]
        pallete_raw[13, :] = [190, 153, 153]
        pallete_raw[14, :] = [180, 165, 180]
        pallete_raw[15, :] = [150, 100, 100]
        pallete_raw[16, :] = [150, 120,  90]
        pallete_raw[17, :] = [153, 153, 153]
        pallete_raw[18, :] = [153, 153, 153]
        pallete_raw[19, :] = [250, 170,  30]
        pallete_raw[20, :] = [220, 220,   0]
        pallete_raw[21, :] = [107, 142,  35]
        pallete_raw[22, :] = [152, 251, 152]
        pallete_raw[23, :] = [ 70, 130, 180]
        pallete_raw[24, :] = [220,  20,  60]
        pallete_raw[25, :] = [255,   0,   0]
        pallete_raw[26, :] = [  0,   0, 142]
        pallete_raw[27, :] = [  0,   0,  70]
        pallete_raw[28, :] = [  0,  60, 100]
        pallete_raw[29, :] = [  0,   0,  90]
        pallete_raw[30, :] = [  0,   0, 110]
        pallete_raw[31, :] = [  0,  80, 100]
        pallete_raw[32, :] = [  0,   0, 230]
        pallete_raw[33, :] = [119,  11,  32]

        pallete_raw = pallete_raw.reshape(-1)

        return pallete_raw

    def evaluate_segmentations(self, pred_segmentations = None):
        """
        top level evaluations
        :param pred_segmentations: the pred segmentation result
        :return: the evaluation results
        """
        if not (pred_segmentations is None):
            self.write_segmentation_result(pred_segmentations)

        info = self._py_evaluate_segmentation()
        return info


    def get_confusion_matrix(self, gt_label, pred_label, class_num):
        """
        Calcute the confusion matrix by given label and pred
        :param gt_label: the ground truth label
        :param pred_label: the pred label
        :param class_num: the nunber of class
        :return: the confusion matrix
        """

        index = (gt_label * class_num + pred_label).astype(np.int64)
        label_count = np.bincount(index)
        confusion_matrix = np.zeros((class_num, class_num))

        for i_label in range(class_num):
            for i_pred_label in range(class_num):
                cur_index = i_label * class_num + i_pred_label
                if cur_index < len(label_count):
                    confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

        return confusion_matrix

    def _evaluateOneImage(self,gt_label, pred_label, class_num):


        Invalid_Mask = np.zeros(gt_label.shape)
        Invalid_Mask[np.where(gt_label == 255)] = 1
        Image_Sub = Invalid_Mask * 255 + (gt_label == pred_label) * 127

        ignore_index = gt_label != 255
        seg_gt = gt_label[ignore_index]
        seg_pred = pred_label[ignore_index]
        one_confusion_matrix = self.get_confusion_matrix(seg_gt, seg_pred, class_num)
        pos = one_confusion_matrix.sum(1)
        res = one_confusion_matrix.sum(0)

        tp = np.diag(one_confusion_matrix)

        IU_array = (tp / np.maximum(1.0, pos + res - tp))
        one_mean_IU = IU_array[IU_array!=0].mean()

        return Image_Sub, IU_array, one_mean_IU

    def _py_evaluate_segmentation(self):
        """
        This function is a wrapper to calculte the metrics for given pred_segmentation results
        :return: the evaluation metrics
        """
        res_file_folder = os.path.join(self.result_path, 'results', 'prediction')
        sub_file_folder = os.path.join(self.result_path, 'results', 'difference')
        result_analyzation_file  =  os.path.join(self.result_path, 'results', 'analysis.txt')

        if os.path.exists(result_analyzation_file):
            os.remove(result_analyzation_file)

        if not os.path.exists(sub_file_folder):
            os.makedirs(sub_file_folder)

        confusion_matrix = np.zeros((self.num_classes,self.num_classes))
        print "len(self.image_set_index)",len(self.image_set_index)
        for i, index in enumerate(self.image_set_index):
            seg_gt_info = self.load_segdb_from_index(index)

            seg_gt = np.array(Image.open(seg_gt_info['seg_cls_path'])).astype(np.int64)
            ID2trainID= self.ID2trainID()

            # convert
            if ID2trainID is not None:
                seg_gt = ID2trainID[seg_gt]

            seg_pathes = os.path.split(seg_gt_info['image'])
            res_image_name = seg_pathes[1][:-len('_leftImg8bit.png')]
            res_subfolder_name = os.path.split(seg_pathes[0])[-1]
            res_save_folder = os.path.join(res_file_folder, res_subfolder_name)
            res_save_path = os.path.join(res_save_folder, res_image_name + '.png')

            seg_pred = np.array(Image.open(res_save_path)).astype(np.int64)
            if ID2trainID is not None:
                seg_pred = ID2trainID[seg_pred]
            seg_pred = cv2.resize(seg_pred, (seg_gt.shape[1], seg_gt.shape[0]), interpolation=cv2.INTER_NEAREST)

            sub_save_path = os.path.join(sub_file_folder, res_image_name + '.png')
            sub_image, oneIU, oneMeanIU = self._evaluateOneImage(seg_gt, seg_pred, self.num_classes)
            Image.fromarray(sub_image.astype(np.uint8)).save(sub_save_path)

            with open(result_analyzation_file,'a') as f:
                np.set_printoptions(formatter={'float': '{:5.2f}'.format}, linewidth=200)
                f.writelines(res_image_name +'\t' + str(oneIU*100) + '\t' +'mean:'+str(oneMeanIU)+'\n')

            ignore_index = seg_gt != 255
            seg_gt = seg_gt[ignore_index].astype(np.int64)
            seg_pred = seg_pred[ignore_index].astype(np.int64)

            confusion_matrix += self.get_confusion_matrix(seg_gt, seg_pred, self.num_classes)


        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)

        IU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IU = IU_array.mean()

        with open(result_analyzation_file, 'a') as f:
            f.writelines('-----------------------------------------------------------------------'+'\n')
            np.set_printoptions(formatter={'float': '{:5.2f}'.format}, linewidth=200)
            f.writelines(res_image_name + '\t' + str(mean_IU * 100) + '\t' + 'mean:' + str(mean_IU)+'\n')

        return {'meanIU':mean_IU, 'IU_array':IU_array}

    def write_segmentation_result(self, segmentation_results):
        """
        Write the segmentation result to result_file_folder
        :param segmentation_results: the prediction result
        :param result_file_folder: the saving folder
        :return: [None]
        """
        res_file_folder = os.path.join(self.result_path, 'results','prediction')
        if not os.path.exists(res_file_folder):
            os.makedirs(res_file_folder)

        pallete = self.getpallete()
        trainID2ID = self.trainID2ID()

        for i, index in enumerate(self.image_set_index):
            # seg_image_info =self.image_path_from_index(index)
            seg_gt_info = self.load_segdb_from_index(index)
            image_pathes = os.path.split(seg_gt_info['image'])
            res_image_name = image_pathes[1][:-len('_leftImg8bit.png')]
            res_subfolder_name = os.path.split(image_pathes[0])[-1]
            res_save_folder = os.path.join(res_file_folder, res_subfolder_name)
            res_save_path = os.path.join(res_save_folder, res_image_name + '.png')

            if not os.path.exists(res_save_folder):
                os.makedirs(res_save_folder)
            
            if trainID2ID is not None:
                segmentation_result = np.uint8(trainID2ID[np.squeeze(np.copy(segmentation_results[i]))])
            else:
                segmentation_result = np.uint8(np.squeeze(np.copy(segmentation_results[i])))

            segmentation_result = Image.fromarray(segmentation_result)
            segmentation_result.putpalette(pallete)
            segmentation_result.save(res_save_path)