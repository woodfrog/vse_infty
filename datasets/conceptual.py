from dataflow import LMDBSerializer, MapDataComponent, BatchData, LocallyShuffleData, TestDataSpeed, MapData, \
    MultiProcessMapDataZMQ
import cv2
import numpy as np
import nltk
import random

import pdb


def conceptual_data(lmdb_source, batch_size, vocab, norm_mode, train, num_workers):
    ds = LMDBSerializer.load(lmdb_source, shuffle=False)
    if train:
        ds = LocallyShuffleData(ds, 50000)
    im_mapper = conceptual_im_func_factory(norm_mode=norm_mode, train=train, target_size=256, crop_size=224)
    cap_mapper = conceptual_cap_cloze_func_factory(vocab)
    mapper = mapper_image_cap_factory(im_mapper, cap_mapper)
    if train:
        ds = MultiProcessMapDataZMQ(ds, num_workers, mapper)
    else:
        ds = MapData(ds, mapper)
    ds = BatchData(ds, batch_size, use_list=True)
    ds = MapData(ds, collate_fn_cloze)
    return ds


def conceptual_data_multi_label(lmdb_source, batch_size, norm_mode, train, num_workers, label2idx):
    ds = LMDBSerializer.load(lmdb_source, shuffle=False)
    if train:
        ds = LocallyShuffleData(ds, 50000)
    ds = MapData(ds, lambda x: (x[0], x[2]))  # skip the caption here.
    im_mapper = conceptual_im_func_factory(norm_mode=norm_mode, train=train)
    label_mapper = conceptual_label_func_factory(label2idx=label2idx)
    mapper = mapper_image_cap_factory(im_mapper, label_mapper)
    if train:
        ds = MultiProcessMapDataZMQ(ds, num_workers, mapper)
    else:
        ds = MapData(ds, mapper)
    ds = BatchData(ds, batch_size, use_list=True)
    ds = MapData(ds, collate_fn_label)
    return ds


def collate_fn_cap(data):
    # Sort a data list by caption length
    images, captions = data
    data = [[im, cap] for im, cap in zip(images, captions)]
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = np.stack(images, 0)
    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = np.zeros((len(captions), max(lengths))).astype(np.int32)
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    new_data = [images, targets, lengths]
    return new_data


def collate_fn_label(data):
    # Sort a data list by caption length
    images, labels = data
    data = [[im, label] for im, label in zip(images, labels)]
    images, labels = zip(*data)
    # Merge images and labels (convert tuple of 3D tensor to 4D tensor)
    images = np.stack(images, 0)
    labels = np.stack(labels, 0)
    new_data = [images, labels]
    return new_data


def collate_fn_cloze(data):
    # Sort a data list by caption length
    images, captions = data
    data = [[im, cap] for im, cap in zip(images, captions)]
    data.sort(key=lambda x: len(x[1][0]), reverse=True)
    images, captions = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = np.stack(images, 0)

    # # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap[0]) for cap in captions]
    targets = np.zeros((len(captions), max(lengths))).astype(np.int32)
    cloze_labels = np.zeros((len(captions), max(lengths))).astype(np.int32)
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[0][:end]
        cloze_labels[i, :end] = cap[1][:end]
    new_data = [images, targets, lengths, cloze_labels]
    return new_data


def mapper_image_cap_factory(func_im, func_cap):
    def mapper(ds):
        im = cv2.imdecode(ds[0], cv2.IMREAD_COLOR)
        if im is None:
            return None
        else:
            return func_im(im), func_cap(ds[1])

    return mapper


def conceptual_cap_func_factory(vocab):
    def tokenize_caption(cap):
        tokens = nltk.tokenize.word_tokenize(str(cap).lower())
        caption = list()
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        return np.array(caption)

    return tokenize_caption


def conceptual_cap_cloze_func_factory(vocab):
    """
    The factory of mapping function for caption, which should also generate the bert-like label for the caption.
    Randomized masking / swapping
    :param vocab: The vocab should contain the typical token for MASK
    :return: a mapping function, built in accordance with the given vocab
    """

    def build_caption_for_cloze(cap):
        tokens = ['<start>',]
        tokens.extend(nltk.tokenize.word_tokenize(str(cap).lower()))
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
        return tokens, output_label

    return build_caption_for_cloze


def conceptual_label_func_factory(label2idx):
    def convert_label(labels):
        label_ids = [label2idx[label] for label in labels]
        # Convert labels into target tensor
        label_target = np.zeros(len(label2idx))
        label_target[label_ids] = 1
        return label_target

    return convert_label


def conceptual_im_func_factory(norm_mode, train, target_size=256, crop_size=224):
    def _conceptual_image_preprocess(x):
        if x is None:
            return None
        im = _im_normalization(x, norm_mode)
        im_shape = im.shape
        im_scale_x = float(target_size) / im_shape[1]
        im_scale_y = float(target_size) / im_shape[0]
        processed_im = cv2.resize(im, None, None, fx=im_scale_x, fy=im_scale_y,
                                  interpolation=cv2.INTER_LINEAR)
        if train:
            processed_im = _crop_im(processed_im, crop_size)

        processed_im = np.transpose(processed_im, (2, 0, 1))
        return processed_im

    return _conceptual_image_preprocess


def _im_normalization(im, mode):
    if mode == 'detector':
        # keep the BGR format
        norm_im = im.astype(np.float32, copy=True)
        norm_im -= np.array([[[102.9801, 115.9465, 122.7717]]])
    elif mode == 'imagenet':
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        norm_im = im.astype(np.float32, copy=True)
        norm_im = norm_im[:, :, ::-1]  # Convert from BGR to RGB
        norm_im = norm_im / 255
        for i in range(norm_im.shape[-1]):
            norm_im[:, :, i] = (norm_im[:, :, i] - imagenet_mean[i]) / imagenet_std[i]
    else:
        raise ValueError('Invalid mode {}'.format(mode))
    return norm_im


def _crop_im(im, crop_size):
    h, w = im.shape[0], im.shape[1]
    if w == crop_size:
        x_start = 0
    else:
        x_start = np.random.randint(w - crop_size, size=1)[0]
    if h == crop_size:
        y_start = 0
    else:
        y_start = np.random.randint(h - crop_size, size=1)[0]
    cropped_im = im[y_start:y_start + crop_size, x_start:x_start + crop_size, :]
    return cropped_im


def _hori_flip(im):
    flipped_im = np.fliplr(im).copy()
    return flipped_im


if __name__ == '__main__':
    ds = conceptual_data(lmdb_source='./data/conceptual/conceptual_training.lmdb', batch_size=128, vocab=vocab,
                         norm_mode='imagenet', train=True)
    ds.reset_state()
    for i, x in enumerate(ds):
        pdb.set_trace()
        # print(item.shape)
    TestDataSpeed(ds).start()
