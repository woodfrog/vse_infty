"""Flickr dataset loader"""

import torch
import torch.utils.data as data
from dataloader import KVReader
from lib.utils.hdfs_utils import pickle_load, numpy_load, hopen
import os
import os.path as osp
import numpy as np
from imageio import imread, imsave
import cv2
import random
import math
import pdb

import logging

logger = logging.getLogger(__name__)


class FlickrRawHDFSDataset(data.IterableDataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split, tokenizer, opt, train, num_readers, shuffle):
        self.tokenizer = tokenizer
        self.opt = opt
        self.train = train
        self.data_path = data_path
        self.shuffle = shuffle
        assert data_path.startswith('hdfs://')

        loc_cap = osp.join(data_path, 'precomp')
        loc_image = osp.join(data_path, 'precomp')
        loc_mapping = osp.join(data_path, 'id_mapping.pkl')
        loc_rois = osp.join(data_path, 'f30k_rois_all.pkl')

        self.id_to_path = pickle_load(loc_mapping)
        self.image_base = 'hdfs:///home/byte_arnold_hl_vc/user/chenjiacheng/data/f30k/packed_f30k_images/f30k'

        self.num_readers = num_readers
        self.keys = KVReader(self.image_base, num_readers).list_keys()
        self.kv_reader_batch = 512
        # this is only used for single worker mode
        # if opt.workers == 0:
        # self.reader = KVReader(self.image_base, num_readers)

        # Set related parameters according to the pre-trained backbone **
        assert 'backbone' in opt.precomp_enc_type
        self.backbone_source = opt.backbone_source
        if 'vsepp' not in self.backbone_source and 'small' not in self.backbone_source and (
                            self.backbone_source == 'detector' or self.backbone_source == 'wsl' or 'imagenet' in self.backbone_source):
            self.input_mode = 'large'
            self.min_scale = 300
            self.max_scale = 500
            self.crop_size = 500
            if 'detector' in self.backbone_source:
                self.pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])
            else:
                self.imagenet_mean = [0.485, 0.456, 0.406]
                self.imagenet_std = [0.229, 0.224, 0.225]
        elif 'vsepp' in self.backbone_source or 'small' in self.backbone_source:
            self.input_mode = 'small'
            self.base_target_size = 256
            self.crop_ratio = 0.875
            self.train_scale_rate = 1
            if hasattr(opt, 'input_scale_factor') and opt.input_scale_factor != 1:
                self.base_target_size = int(self.base_target_size * opt.input_scale_factor)
                logger.info('Input mode small: scaled by factor {}'.format(opt.input_scale_factor))
            if 'detector' in self.backbone_source:
                self.pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])
            else:
                self.imagenet_mean = [0.485, 0.456, 0.406]
                self.imagenet_std = [0.229, 0.224, 0.225]
        else:
            self.input_mode = 'unknown'
            raise ValueError('Invalid backbone type {}'.format(self.backbone_source))

        # Captions
        self.captions = []
        with hopen(osp.join(loc_cap, '%s_caps.txt' % data_split), 'r') as f:
            for line in f:
                self.captions.append(line.decode('utf-8').strip())

        # Images
        with hopen(osp.join(loc_image, '{}_ids.txt'.format(data_split)), 'r') as f:
            image_ids = f.readlines()
            self.images = [int(x.strip()) for x in image_ids]

        self.image_rois = pickle_load(loc_rois)

        self.length = len(self.captions)
        self.all_index = np.arange(self.length)
        if self.train:
            self.shuffle_index()

        # Set the paramater for every reader
        self.start = 0
        self.end = self.length

        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        num_images = len(self.images)

        if num_images != self.length:
            self.im_div = 5
        else:
            self.im_div = 1

    def shuffle_index(self):
        self.all_index = np.random.permutation(self.all_index)
        logger.info('Shuffle global dataset index')

    def __iter__(self):
        all_index_worker = self.all_index[self.start:self.end]  # all index on this worker
        if self.shuffle:
            all_index_worker = np.random.permutation(all_index_worker)
        num_read_iter = len(all_index_worker) // self.kv_reader_batch + 1

        for i in range(num_read_iter):
            all_index_iter = list()
            caption_targets = list()
            caption_cloze_labels = list()
            img_keys = list()

            # self.reset_training_image_scale()

            # process caption and get the corresponding image key
            for index in all_index_worker[i * self.kv_reader_batch: (i + 1) * self.kv_reader_batch]:
                # Convert caption (string) to word ids.
                caption = self.captions[index]
                caption_tokens = self.tokenizer.basic_tokenizer.tokenize(caption)
                if hasattr(self.opt, 'drop') and self.opt.drop:
                    target, cloze_label = process_caption(self.tokenizer, caption_tokens, self.train)
                else:
                    target, cloze_label = process_caption(self.tokenizer, caption_tokens, False)
                caption_targets.append(target)
                caption_cloze_labels.append(cloze_label)
                all_index_iter.append(index)

                img_index = index // self.im_div
                image_path = self.id_to_path[self.images[img_index]]
                img_key = image_path  # get the key from the saved path in the pickle file
                img_keys.append(img_key)

            # read images and do preprocessing & data aug
            batch_values = self.reader.read_many(img_keys)
            for idx_i, value in enumerate(batch_values):
                im_in = np.array(imread(value))
                rois = self.image_rois[img_keys[idx_i]]
                blobs, processed_rois, im_scale_x, im_scale_y = self._process_image_with_entities(im_in,
                                                                                                  rois,
                                                                                                  mode='box')
                image = torch.Tensor(blobs)
                image = image.permute(2, 0, 1)
                processed_rois = torch.Tensor(processed_rois)
                target = caption_targets[idx_i]
                index = all_index_iter[idx_i]
                img_index = index // self.im_div
                cloze_label = caption_cloze_labels[idx_i]
                yield image, processed_rois, target, index, img_index, cloze_label

    def __len__(self):
        return self.length

    def _process_image_with_entities(self, im_in, entities, mode='box'):
        """Converts an image into a network input, with pre-processing including re-scaling, padding, etc, and data
        augmentation. The entity information(boxes or masks) are adjusted accordingly together with the image
        Arguments:
          im_in (ndarray): input image
          entities: a list of bounding boxes (x1, y1, x2, y2), specifying the location of entities in the image
          mode: 'box' | 'mask', the type of entity information
        """
        if mode not in ['box', 'mask']:
            raise ValueError('Invalid mode for image processing with entities {}'.format(mode))

        if len(im_in.shape) == 2:
            im_in = im_in[:, :, np.newaxis]
            im_in = np.concatenate((im_in, im_in, im_in), axis=2)

        if 'detector' in self.backbone_source:
            im_in = im_in[:, :, ::-1]
        im = im_in.astype(np.float32, copy=True)

        if self.input_mode == 'large':
            # make a copy of the RoIs
            image_entities = np.copy(entities)

            # 2. Re-scale and adjust rois accordingly
            im_shape = im.shape
            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])

            im_scale = float(self.min_scale) / im_size_min
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > self.max_scale:
                im_scale = float(self.max_scale) / float(im_size_max)
            im_scale_x = im_scale_y = im_scale
            processed_im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                                      interpolation=cv2.INTER_LINEAR)
            # rescale the bbox as well
            if mode == 'box':
                image_entities = self._rescale_rois(image_entities, im_scale_x, im_scale_y)
            else:
                rescaled = list()
                for i in range(image_entities.shape[0]):
                    rescaled.append(cv2.resize(image_entities[i], None, None, fx=im_scale, fy=im_scale,
                                               interpolation=cv2.INTER_LINEAR))
                image_entities = np.stack(rescaled, axis=0)

            # 3. Pad the image
            # Pad the image if the shortest side is smaller than the crop size
            h, w = processed_im.shape[0], processed_im.shape[1]
            if h < self.crop_size:
                pad_top = round((self.crop_size - h) / 2)
                pads = ((pad_top, self.crop_size - h - pad_top), (0, 0), (0, 0))
                processed_im = np.pad(processed_im, pads, 'constant', constant_values=0)
                if mode == 'box':
                    image_entities[:, 1] = image_entities[:, 1] + pad_top
                    image_entities[:, 3] = image_entities[:, 3] + pad_top
                else:
                    # Pad the masks accordingly
                    pads_mask = ((0, 0), (pad_top, self.crop_size - h - pad_top), (0, 0))
                    image_entities = np.pad(image_entities, pads_mask, 'constant', constant_values=0)
            if w < self.crop_size:
                pad_left = round((self.crop_size - w) / 2)
                pads = ((0, 0), (pad_left, self.crop_size - w - pad_left), (0, 0))
                processed_im = np.pad(processed_im, pads, 'constant', constant_values=0)
                if mode == 'box':
                    image_entities[:, 0] = image_entities[:, 0] + pad_left
                    image_entities[:, 2] = image_entities[:, 2] + pad_left
                else:
                    pads_mask = ((0, 0), (0, 0), (pad_left, self.crop_size - w - pad_left))
                    image_entities = np.pad(image_entities, pads_mask, 'constant', constant_values=0)

            # 4. Crop a patch from the re-scaled image with specified crop size
            processed_im, image_entities, crop_box = self._crop(processed_im, image_entities, self.crop_size,
                                                                self.train, mode)
            if mode == 'mask':
                image_entities = self._downsample_mask(image_entities, scale=32)
        else:
            assert self.input_mode == 'small'
            # make a copy of the RoIs
            image_entities = np.copy(entities)

            if self.train:
                target_size = self.base_target_size * self.train_scale_rate
            else:
                target_size = self.base_target_size

            # 2. Random crop when in training mode, elsewise just skip
            if self.train:
                crop_ratio = np.random.random() * 0.4 + 0.6
                crop_size_h = int(im.shape[0] * crop_ratio)
                crop_size_w = int(im.shape[1] * crop_ratio)
                #if np.random.random() > 0.5:  # randomly choose a aspect ratio
                #    crop_size_w = min(int(crop_size_h * 3.0 / 4), im.shape[1])
                #else:
                #    crop_size_w = min(int(crop_size_h * 4.0 / 3), im.shape[1])
                processed_im, image_entities, crop_box = self._crop(im, image_entities, crop_size_h, crop_size_w,
                                                                    random=True, mode=mode)
            else:
                processed_im = im

            # 3. Resize to the target resolution
            im_shape = processed_im.shape
            im_scale_x = float(target_size) / im_shape[1]
            im_scale_y = float(target_size) / im_shape[0]
            processed_im = cv2.resize(processed_im, None, None, fx=im_scale_x, fy=im_scale_y,
                                      interpolation=cv2.INTER_LINEAR)
            if mode == 'box':
                image_entities = self._rescale_rois(image_entities, im_scale_x, im_scale_y)
            else:
                raise NotImplementedError('224 imagenet pretrained model not used for entity mask yet.')

        if self.train:
            if np.random.random() > 0.5:
                processed_im, image_entities = self._hori_flip(processed_im, image_entities, mode)

        # 1. Normalization
        if 'detector' in self.backbone_source:
            processed_im = self._detector_norm(processed_im)
        elif 'simclr' in self.backbone_source:
            processed_im = processed_im.astype(np.float32)
            processed_im /= 255
        else:
            processed_im = self._imagenet_norm(processed_im)

        # self._test_entities(processed_im, image_entities, mode)
        return processed_im, image_entities, im_scale_x, im_scale_y

    def _imagenet_norm(self, im_in):
        im_in = im_in.astype(np.float32)
        im_in = im_in / 255
        for i in range(im_in.shape[-1]):
            im_in[:, :, i] = (im_in[:, :, i] - self.imagenet_mean[i]) / self.imagenet_std[i]
        return im_in

    def _detector_norm(self, im_in):
        im_in = im_in.astype(np.float32)
        im_in -= self.pixel_means
        return im_in

    @staticmethod
    def _crop(im, entities, crop_size_h, crop_size_w, random, mode):
        h, w = im.shape[0], im.shape[1]
        if random:
            if w - crop_size_w == 0:
                x_start = 0
            else:
                x_start = np.random.randint(w - crop_size_w, size=1)[0]
            if h - crop_size_h == 0:
                y_start = 0
            else:
                y_start = np.random.randint(h - crop_size_h, size=1)[0]
        else:
            x_start = (w - crop_size_w) // 2
            y_start = (h - crop_size_h) // 2

        cropped_im = im[y_start:y_start + crop_size_h, x_start:x_start + crop_size_w, :]
        crop_box = np.array(
            [x_start, y_start, x_start + crop_size_w - 1, y_start + crop_size_h - 1])  # x1, y1, x2, y2

        if mode == 'box':
            # Clip the rois according to the crop bbox
            entities[:, 0] = np.clip(entities[:, 0], crop_box[0], crop_box[2]) - crop_box[0]
            entities[:, 2] = np.clip(entities[:, 2], crop_box[0], crop_box[2]) - crop_box[0]
            entities[:, 1] = np.clip(entities[:, 1], crop_box[1], crop_box[3]) - crop_box[1]
            entities[:, 3] = np.clip(entities[:, 3], crop_box[1], crop_box[3]) - crop_box[1]
        else:
            raise ValueError('Invalid mode for cropping: {}'.format(mode))

        return cropped_im, entities, crop_box

    @staticmethod
    def _rescale_rois(rois, scale_x, scale_y):
        if scale_x == scale_y:
            rois = rois * scale_x
        else:
            rois[:, 0] = rois[:, 0] * scale_x
            rois[:, 2] = rois[:, 2] * scale_x
            rois[:, 1] = rois[:, 1] * scale_y
            rois[:, 3] = rois[:, 3] * scale_y
        return rois

    @staticmethod
    def _hori_flip(im, entities, mode='box'):
        im = np.fliplr(im).copy()
        width = im.shape[1]
        if mode == 'box':
            new_entities = entities.copy()
            new_entities[:, 0] = width - 1 - entities[:, 2]
            new_entities[:, 2] = width - 1 - entities[:, 0]
        else:
            new_entities = np.fliplr(np.transpose(entities, (1, 2, 0))).copy()
            new_entities = np.transpose(new_entities, (2, 0, 1))
        return im, new_entities


class FlickrPrecompHDFSDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split, tokenizer, opt, train):
        self.tokenizer = tokenizer
        self.opt = opt
        self.train = train
        self.data_path = data_path
        assert data_path.startswith('hdfs://')

        loc_cap = osp.join(data_path, 'precomp')
        loc_image = osp.join(data_path, 'precomp')

        # Captions
        self.captions = []
        with hopen(osp.join(loc_cap, '%s_caps.txt' % data_split), 'r') as f:
            for line in f:
                self.captions.append(line.decode('utf-8').strip())
        # Image features
        self.images = numpy_load(os.path.join(loc_image, '%s_ims.npy' % data_split))
        # self.images = np.load(os.path.join('data/coco_precomp/', '%s_ims.npy' % data_split))

        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        num_images = len(self.images)

        if num_images != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

    def __getitem__(self, index):
        # handle the image redundancy
        img_index = index // self.im_div
        caption = self.captions[index]
        caption_tokens = self.tokenizer.basic_tokenizer.tokenize(caption)

        # Convert caption (string) to word ids.
        if hasattr(self.opt, 'drop') and self.opt.drop:
            target, cloze_label = process_caption(self.tokenizer, caption_tokens, self.train)
        else:
            target, cloze_label = process_caption(self.tokenizer, caption_tokens, False)
        image = self.images[img_index]
        if self.train:
            num_features = image.shape[0]
            rand_list = np.random.rand(num_features)
            image = image[np.where(rand_list > 0.20)]
        image = torch.Tensor(image)
        return image, target, index, img_index, cloze_label

    def __len__(self):
        return self.length


def process_caption(tokenizer, tokens, train=True):
    output_tokens = []
    output_label = []
    deleted_idx = []

    for i, token in enumerate(tokens):
        sub_tokens = tokenizer.wordpiece_tokenizer.tokenize(token)
        prob = random.random()

        if prob < 0.20 and train:  # mask/remove the tokens only during training
            prob /= 0.20

            # 50% randomly change token to mask token
            if prob < 0.5:
                for sub_token in sub_tokens:
                    output_tokens.append("[MASK]")
            # 10% randomly change token to random token
            elif prob < 0.6:
                for sub_token in sub_tokens:
                    output_tokens.append(random.choice(list(tokenizer.vocab.keys())))
                    # -> rest 10% randomly keep current token
            else:
                for sub_token in sub_tokens:
                    output_tokens.append(sub_token)
                    deleted_idx.append(len(output_tokens) - 1)

            # append current token to output (we will predict these later)
            for sub_token in sub_tokens:
                try:
                    output_label.append(tokenizer.vocab[sub_token])
                except KeyError:
                    # For unknown words (should not occur with BPE vocab)
                    output_label.append(tokenizer.vocab["[UNK]"])
                    print("Cannot find sub_token '{}' in vocab. Using [UNK] insetad".format(sub_token))
        else:
            for sub_token in sub_tokens:
                # no masking token (will be ignored by loss function later)
                output_tokens.append(sub_token)
                output_label.append(0)

    if len(deleted_idx) != 0:
        output_tokens = [output_tokens[i] for i in range(len(output_tokens)) if i not in deleted_idx]
        output_label = [output_label[i] for i in range(len(output_label)) if i not in deleted_idx]

    output_tokens = ['[CLS]'] + output_tokens + ['[SEP]']
    output_label = [0] + output_label + [0]
    target = tokenizer.convert_tokens_to_ids(output_tokens)
    target = torch.Tensor(target)
    output_label = torch.Tensor(output_label)
    return target, output_label


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    # Avoid "cannot pickle KVReader object" error
    dataset.reader = KVReader(dataset.image_base, dataset.num_readers)
    overall_start = dataset.start
    overall_end = dataset.end
    per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)
    logger.info('set start and end to be {}:{}'.format(dataset.start, dataset.end))


def collate_fn_normal(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    if len(data[0]) == 5:
        # Sort a data list by caption length
        data.sort(key=lambda x: len(x[1]), reverse=True)
        images, captions, ids, img_ids, cloze_labels = zip(*data)

        # Merge images (convert tuple of 3D tensor to 4D tensor)
        # images = torch.stack(images, 0)
        img_lengths = [len(image) for image in images]
        all_images = torch.zeros(len(images), max(img_lengths), images[0].size(-1))
        for i, image in enumerate(images):
            end = img_lengths[i]
            all_images[i, :end] = image[:end]
        img_lengths = torch.Tensor(img_lengths)

        # Merget captions (convert tuple of 1D tensor to 2D tensor)
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()
        cloze_labels_collate = torch.zeros(len(captions), max(lengths)).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]
            cloze_labels_collate[i, :end] = cloze_labels[i][:end]

        return all_images, img_lengths, targets, lengths, ids, cloze_labels_collate
    else:
        data.sort(key=lambda x: len(x[2]), reverse=True)
        images, rois, captions, ids, img_ids, cloze_labels = zip(*data)
        # Merge images (convert tuple of 3D tensor to 4D tensor)
        images = torch.stack(images, 0)
        rois = torch.stack(rois, 0)
        rois_padding = torch.zeros(rois.size(0), rois.size(1), 1)
        rois = torch.cat([rois_padding, rois], dim=-1)

        # Merget captions (convert tuple of 1D tensor to 2D tensor)
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()
        cloze_labels_collate = torch.zeros(len(captions), max(lengths)).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]
            cloze_labels_collate[i, :end] = cloze_labels[i][:end]
        return images, rois, targets, lengths, ids, cloze_labels_collate


def get_loader(data_path, data_split, tokenizer, opt, batch_size=100,
               shuffle=True, num_workers=2, num_readers=32, train=True):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    if opt.precomp_enc_type == 'basic':
        dset = FlickrPrecompHDFSDataset(data_path, data_split, tokenizer, opt, train)
        data_loader = torch.utils.data.DataLoader(dataset=dset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  pin_memory=True,
                                                  collate_fn=collate_fn_normal,
                                                  num_workers=num_workers)
    else:
        dset = FlickrRawHDFSDataset(data_path, data_split, tokenizer, opt, train, num_readers, shuffle)
        data_loader = torch.utils.data.DataLoader(dataset=dset,
                                                  batch_size=batch_size,
                                                  num_workers=num_workers,
                                                  worker_init_fn=worker_init_fn,
                                                  collate_fn=collate_fn_normal,
                                                  pin_memory=True,
                                                  multiprocessing_context='spawn')
    return data_loader


def get_loaders(data_path, tokenizer, batch_size, workers, opt):
    train_loader = get_loader(data_path, 'train', tokenizer, opt,
                              batch_size, True, workers)
    val_loader = get_loader(data_path, 'dev', tokenizer, opt,
                            batch_size, False, workers, train=False)
    return train_loader, val_loader


def get_train_loader(data_path, tokenizer, batch_size, workers, opt, shuffle):
    train_loader = get_loader(data_path, 'train', tokenizer, opt,
                              batch_size, shuffle, workers)
    return train_loader


def get_test_loader(split_name, data_name, tokenizer, batch_size,
                    workers, opt):
    test_loader = get_loader(opt.data_path, split_name, tokenizer, opt,
                             batch_size, False, workers, train=False)
    return test_loader


if __name__ == '__main__':
    data_path = 'hdfs:///home/byte_arnold_hl_vc/user/chenjiacheng/data/coco/'
