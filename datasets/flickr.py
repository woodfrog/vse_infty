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
        loc_image = data_path + '/'
        loc_mapping = './data/f30k/id_mapping.pkl'
        self.image_base = './data/f30k/flickr30k-images'

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
                if 'detector' in self.backbone_source:
                    self.pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])
                else:
                    self.imagenet_mean = [0.485, 0.456, 0.406]
                    self.imagenet_std = [0.229, 0.224, 0.225]
            else:
                raise ValueError('Invalid backbone type {}'.format(self.backbone_source))

        # Captions
        self.captions = []
        with open(loc_cap + '%s_caps.txt' % data_split, 'rb') as f:
            for line in f:
                self.captions.append(line.strip())
        # Image features
        if opt.precomp_enc_type == 'basic':
            self.images = np.load(loc_image + '%s_ims.npy' % data_split)
        elif 'backbone' in opt.precomp_enc_type:
            with open(loc_image + '{}_ids.txt'.format(data_split), 'r') as f:
                image_ids = f.readlines()
                self.images = [int(x.strip()) for x in image_ids]
        else:
            raise ValueError('Invalid encoder type parameter {}'.format(opt.precomp_enc_type))

        self.length = len(self.captions)
        # self.length = len(self.images)  # this is used only for pre-extracting image features with fixed extractors

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

        # Convert caption (string) to word ids.
        if hasattr(self.opt, 'drop') and self.opt.drop:
            target, cloze_label = self.process_caption(caption, drop=self.train)
        else:
            target, cloze_label = self.process_caption(caption, drop=False)

        if self.opt.precomp_enc_type == 'basic':
            image = torch.Tensor(self.images[img_index])
        else:
            image_id = self.images[img_index]
            image_path = os.path.join(self.image_base, self.id_to_path[image_id])
            im_in = np.array(imread(image_path))
            blobs, im_scale_x, im_scale_y = self._process_image(im_in)
            image = torch.Tensor(blobs)
            image = image.permute(2, 0, 1)
        return image, None, target, index, img_index, cloze_label

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

    def _process_image(self, im_in):
        if len(im_in.shape) == 2:
            im_in = im_in[:, :, np.newaxis]
            im_in = np.concatenate((im_in, im_in, im_in), axis=2)

        assert self.input_mode == 'small'
        # 1. Normalization
        if 'detector' in self.backbone_source:
            im_orig = self._detector_norm(im_in)
        else:
            im_orig = self._imagenet_norm(im_in)

        # 2. Resize
        im_shape = im_orig.shape
        im_scale_x = float(self.target_size) / im_shape[1]
        im_scale_y = float(self.target_size) / im_shape[0]
        processed_im = cv2.resize(im_orig, None, None, fx=im_scale_x, fy=im_scale_y,
                                  interpolation=cv2.INTER_LINEAR)

        # 3. Random crop when in training mode, elsewise just skip
        if self.train:
            assert self.crop_size < self.target_size
            processed_im, crop_box = self._crop(processed_im, self.crop_size, random=True)

        return processed_im, im_scale_x, im_scale_y

    def _imagenet_norm(self, im_in):
        im_orig = im_in.astype(np.float32, copy=True)
        im_orig = im_orig / 255
        for i in range(im_orig.shape[-1]):
            im_orig[:, :, i] = (im_orig[:, :, i] - self.imagenet_mean[i]) / self.imagenet_std[i]
        return im_orig

    def _detector_norm(self, im_in):
        im = im_in[:, :, ::-1]
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= self.pixel_means
        return im_orig

    @staticmethod
    def _crop(im, crop_size, random):
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

        return cropped_im, crop_box

    @staticmethod
    def _hori_flip(im):
        im = np.fliplr(im).copy()
        return im


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
