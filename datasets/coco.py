"""COCO dataset loader"""

import torch
import torch.utils.data as data
import os
import nltk
import numpy as np
from imageio import imread, imsave
import cv2
import pickle
import random
import pdb


class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split, vocab, opt, train):
        self.vocab = vocab
        self.opt = opt
        self.train = train
        self.data_path = data_path
        loc_cap = data_path + '/'
        if hasattr(opt, 'use_lmdb') and opt.use_lmdb:
            self.use_lmdb = True
        else:
            self.use_lmdb = False

        if opt.precomp_enc_type == 'basic':
            loc_image = data_path + '/'
        else:
            loc_image = data_path + '/original_updown/'
            self.coco_image_base = '../../my_vse/data/coco/images'
            loc_mapping = data_path + '/original_updown/id_mapping.pkl'
            with open(loc_mapping, 'rb') as f:
                self.id_to_path = pickle.load(f)

        # ** Need to set these parameters according to the pre-trained backbone **
        if 'backbone' in opt.precomp_enc_type:
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
                self.target_size = 256
                self.crop_size = 224
                if hasattr(opt, 'input_scale_factor') and opt.input_scale_factor != 1:
                    self.target_size = int(self.target_size * opt.input_scale_factor)
                    self.crop_size = int(self.crop_size * opt.input_scale_factor)
                    print('Input mode small: scaled by factor {}'.format(opt.input_scale_factor))
                if self.backbone_source == 'vsepp_detector_fix':
                    self.input_mode = 'large'
                    self.min_scale = 600
                    self.max_scale = 1000
                    self.crop_size = 1000
                    self.pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])
                elif 'detector' in self.backbone_source:
                    self.pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])
                else:
                    self.imagenet_mean = [0.485, 0.456, 0.406]
                    self.imagenet_std = [0.229, 0.224, 0.225]
            else:
                self.input_mode = 'unknown'
                raise ValueError('Invalid backbone type {}'.format(self.backbone_source))

        # Captions
        self.captions = []
        with open(loc_cap + '%s_caps.txt' % data_split, 'rb') as f:
            for line in f:
                self.captions.append(line.strip())
        # Image features
        if opt.precomp_enc_type == 'basic':
            if not self.use_lmdb:
                self.images = np.load(loc_image + '%s_ims.npy' % data_split)
            else:  # use lmdb to read in the data
                import lmdb
                id_path = os.path.join(data_path, '{}_ids.txt'.format(data_split))
                db_path = os.path.join(loc_image, '{}_ims'.format(data_split))
                with open(id_path, 'r') as id_f:
                    ids = id_f.readlines()
                self.data_ids = [int(item.strip()) for item in ids]
                self._lmdb_env = lmdb.open(db_path, subdir=os.path.isdir(db_path), readonly=True, lock=False,
                                           readahead=True, map_size=1099511627776 * 2, max_readers=100)
                self._txn = self._lmdb_env.begin(write=False)
                print('Set up lmdb env')
        elif 'backbone' in opt.precomp_enc_type:
            with open(loc_image + '{}_rois.pkl'.format(data_split), 'rb') as f:
                self.images = pickle.load(f)
        else:
            raise ValueError('Invalid encoder type parameter {}'.format(opt.precomp_enc_type))

        self.length = len(self.captions)
        # self.length = len(self.images)  # this is used only for pre-extracting image features with fixed extractors

        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if not self.use_lmdb:
            num_images = len(self.images)
        else:
            num_images = len(self.data_ids)

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

        # Convert caption (string) to word ids.
        if hasattr(self.opt, 'drop') and self.opt.drop:
            target, cloze_label = self.process_caption(caption, drop=self.train)
        else:
            target, cloze_label = self.process_caption(caption, drop=False)

        if self.opt.precomp_enc_type == 'basic':
            if not self.use_lmdb:
                image = torch.Tensor(self.images[img_index])
            else:
                image_id = str(self.data_ids[img_index]).encode('ascii')
                image = pickle.loads(self._txn.get(image_id))
                image = torch.Tensor(image)
            return image, target, index, img_index, cloze_label
        else:
            image_path, image_rois = self.images[img_index]
            if not isinstance(image_path,
                              str):  # When the first element is an ID instead of the file path, get the path
                image_path = os.path.join(self.coco_image_base, self.id_to_path[image_path])
            im_in = np.array(imread(image_path))
            blobs, processed_rois, im_scale_x, im_scale_y = self._process_image_with_entities(im_in, image_rois,
                                                                                              mode='box')
            image = torch.Tensor(blobs)
            image = image.permute(2, 0, 1)
            processed_rois = torch.Tensor(processed_rois)
            return image, processed_rois, target, index, img_index, cloze_label

    def process_caption(self, caption, drop=False):
        vocab = self.vocab
        if not drop:
            tokens = nltk.tokenize.word_tokenize(
                str(caption).lower())
            caption = list()
            caption.append(vocab('<start>'))
            caption.extend([vocab(token) for token in tokens])
            caption.append(vocab('<end>'))
            target = torch.Tensor(caption)
            return target, target
        else:
            # Convert caption (string) to word ids.
            tokens = ['<start>', ]
            tokens.extend(nltk.tokenize.word_tokenize(str(caption).lower()))
            tokens.append('<end>')
            output_label = []
            for i, token in enumerate(tokens):
                prob = random.random()
                if prob < 0.15:
                    prob /= 0.15
                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = vocab.word2idx['<mask>']
                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.randrange(len(vocab))
                    # 10% randomly change token to current token
                    else:
                        tokens[i] = vocab(token)
                    output_label.append(vocab(token))
                else:
                    tokens[i] = vocab(token)
                    output_label.append(0)
            target = torch.Tensor(tokens)
            output_label = torch.Tensor(output_label)
            return target, output_label

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

            # 2. Resize
            im_shape = im.shape
            im_scale_x = float(self.target_size) / im_shape[1]
            im_scale_y = float(self.target_size) / im_shape[0]
            processed_im = cv2.resize(im, None, None, fx=im_scale_x, fy=im_scale_y,
                                      interpolation=cv2.INTER_LINEAR)
            if mode == 'box':
                image_entities = self._rescale_rois(image_entities, im_scale_x, im_scale_y)
            else:
                raise NotImplementedError('224 imagenet pretrained model not used for entity mask yet.')

            # 3. Random crop when in training mode, elsewise just skip
            if self.train:
                assert self.crop_size < self.target_size
                processed_im, image_entities, crop_box = self._crop(processed_im, image_entities, self.crop_size,
                                                                    random=True, mode=mode)

        if self.train:
            if np.random.random() > 0.5:
                processed_im, image_entities = self._hori_flip(processed_im, image_entities, mode)

        # 1. Normalization
        if 'detector' in self.backbone_source:
            processed_im = self._detector_norm(processed_im)
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
    def _crop(im, entities, crop_size, random, mode):
        h, w = im.shape[0], im.shape[1]
        if random:
            if w - crop_size == 0:
                x_start = 0
            else:
                x_start = np.random.randint(w - crop_size, size=1)[0]
            if h - crop_size == 0:
                y_start = 0
            else:
                y_start = np.random.randint(h - crop_size, size=1)[0]
        else:
            x_start = (w - crop_size) // 2
            y_start = (h - crop_size) // 2

        cropped_im = im[y_start:y_start + crop_size, x_start:x_start + crop_size, :]
        crop_box = np.array(
            [x_start, y_start, x_start + crop_size - 1, y_start + crop_size - 1])  # x1, y1, x2, y2

        if mode == 'box':
            # Clip the rois according to the crop bbox
            entities[:, 0] = np.clip(entities[:, 0], crop_box[0], crop_box[2]) - crop_box[0]
            entities[:, 2] = np.clip(entities[:, 2], crop_box[0], crop_box[2]) - crop_box[0]
            entities[:, 1] = np.clip(entities[:, 1], crop_box[1], crop_box[3]) - crop_box[1]
            entities[:, 3] = np.clip(entities[:, 3], crop_box[1], crop_box[3]) - crop_box[1]
        elif mode == 'mask':
            entities = entities[:, y_start:y_start + crop_size, x_start:x_start + crop_size]
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

    def _downsample_mask(self, masks, scale):
        times = int(np.log2(scale))
        size_x = masks.shape[2]
        size_y = masks.shape[1]
        for i in range(times):
            size_x = (size_x + 1) // 2
            size_y = (size_y + 1) // 2
        resized_masks = cv2.resize(np.transpose(masks, (1, 2, 0)), (size_x, size_y), interpolation=cv2.INTER_LINEAR)
        if len(resized_masks.shape) == 2:
            resized_masks = resized_masks[:, :, np.newaxis]
        resized_masks = np.transpose(resized_masks, (2, 0, 1))
        return resized_masks


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
        images = torch.stack(images, 0)

        # Merget captions (convert tuple of 1D tensor to 2D tensor)
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()
        cloze_labels_collate = torch.zeros(len(captions), max(lengths)).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]
            cloze_labels_collate[i, :end] = cloze_labels[i][:end]

        return images, targets, lengths, ids, cloze_labels_collate
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


def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2, train=True):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_path, data_split, vocab, opt, train)

    collate_fn = collate_fn_normal
    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn,
                                              num_workers=num_workers)
    return data_loader


def get_loaders(data_name, vocab, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    train_loader = get_precomp_loader(dpath, 'train', vocab, opt,
                                      batch_size, True, workers)
    val_loader = get_precomp_loader(dpath, 'dev', vocab, opt,
                                    batch_size, False, workers, train=False)
    return train_loader, val_loader


def get_test_loader(split_name, data_name, vocab, batch_size,
                    workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    test_loader = get_precomp_loader(dpath, split_name, vocab, opt,
                                     batch_size, False, workers, train=False)
    return test_loader
