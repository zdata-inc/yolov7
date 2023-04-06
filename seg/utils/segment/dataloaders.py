# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Dataloaders
"""

import collections
import copy
import json
import os
from pathlib import Path
import random
import re

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, distributed

from ..augmentations import augment_hsv, copy_paste, letterbox
from ..dataloaders import InfiniteDataLoader, LoadImagesAndLabels, seed_worker
from ..general import LOGGER, xyn2xy, xywhn2xyxy, xyxy2xywhn
from ..torch_utils import torch_distributed_zero_first
from .augmentations import mixup, random_perspective


def create_dataloader(path,
                      imgsz,
                      batch_size,
                      stride,
                      single_cls=False,
                      hyp=None,
                      augment=False,
                      cache=False,
                      pad=0.0,
                      rect=False,
                      rank=-1,
                      workers=8,
                      image_weights=False,
                      quad=False,
                      prefix='',
                      shuffle=False,
                      mask_downsample_ratio=1,
                      overlap_mask=False,
                      data_dict=None):
    if rect and shuffle:
        LOGGER.warning('WARNING: --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = LoadImagesAndLabelsAndMasks(
            path,
            imgsz,
            batch_size,
            augment=augment,  # augmentation
            hyp=hyp,  # hyperparameters
            rect=rect,  # rectangular batches
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix,
            downsample_ratio=mask_downsample_ratio,
            overlap=overlap_mask,
            data_dict=data_dict)

    #dataset = FramePairDataset(dataset)
    dataset = ChangeDataAugDataset(dataset)

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    #loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    loader = DataLoader
    # generator = torch.Generator()
    # generator.manual_seed(0)
    return loader(
        dataset,
        batch_size=None,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=True,
        #collate_fn=LoadImagesAndLabelsAndMasks.collate_fn4 if quad else LoadImagesAndLabelsAndMasks.collate_fn,
        worker_init_fn=seed_worker,
        # generator=generator,
    ), dataset


class ChangeDataAugDataset(Dataset):
    def __init__(self, org_dataset):
        """ Augments an input dataset (`org_dataset`) for the purposes of
        creating a Change Detection dataset.  For each image in the dataset (the
        'target' image), take a random selection of objects from a randomly
        chosen secondary image and copy-paste those objects into the target
        image. Then pair this augmented version of the image with the original
        image for the purposes of modeling it with a change detection model.
        
        The augmented and original target image are paired and added to the
        ChangeDataAugDataset twice. The first time, they are added as a tuple
        with the augmented image first and the original image second. Then in
        the augmented image, the copy-pasted objects are flagged as 'deletions'
        to indicate that they are subsequently deleted. The second time they are
        added as a tuple with the original image first and the augmented image
        second. In this case the copy-pasted objects are annotated as
        'additions' in the second image.
        """

        self.paired_items = []

        for im_id in range(len(org_dataset)):
            # The copy-paste function assumes a different ordering of tensor
            # dimensions, so the below manipulations make the appropriate
            # changes. Further down we reverse these changes after augmentation
            im = org_dataset[im_id][0].transpose(0, 2).transpose(0, 1).cpu().numpy()
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            # Translate the bounding boxes into pixel coordinates.
            im_labels = org_dataset[im_id][1]
            im1_h, im1_w = im.shape[:2]
            im_labels[:, 2:] = xywhn2xyxy(im_labels[:, 2:], w=im1_w, h=im1_h)

            # Translate the polygon segments into pixel coordinates.
            segments = org_dataset.segments[im_id]
            padw, padh = org_dataset[im_id][7]
            w, h = org_dataset[im_id][8]
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]

            # Choose a random image from the dataset to copy objects from
            im2_id = random.randint(0, len(org_dataset)-1)
            while im2_id == im_id:
                im2_id = random.randint(0, len(org_dataset)-1)
            
            # Similar to above, prepare the dimensions of this image so that it
            # is compatible with the assumptions of copy_paste()
            im2 = org_dataset[im2_id][0].transpose(0, 2).transpose(0, 1).cpu().numpy()
            im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

            # Translate bboxes into pixel coordinates
            padw, padh = org_dataset[im2_id][7]
            w, h = org_dataset[im2_id][8]
            im2_labels = org_dataset[im2_id][1]
            im2_labels[:, 2:] = torch.tensor(xywhn2xyxy(org_dataset.labels[im2_id][:, 1:],
                                                        w=w, h=h, padw=padw, padh=padh))

            # Translate polygon segments into pixel coordinates
            im2_segments = org_dataset.segments[im2_id]
            im2_segments = [xyn2xy(x, w, h, padw, padh) for x in im2_segments] 

            # This code here demonstrates how to reverse the translation into
            # pixel coordinates.  It indicates how the w/h values (which come
            # from the original image) don't include padding, while the
            # org_dataset[im2_id] representation includes padding. To translate
            # from org_dataset.labels[imd2_id] to pixel coordinates, you need
            # to consider the original size of the image with w/h. But if we
            # were to translate from org_dataset[im2_id][1] then those
            # coordinates already factor in the padding, and so we would uset
            # im2.shape[1] and im2.shape[0] as width and height, respectively.
            #ratio = org_dataset[im2_id][9]
            #cycle = xyxy2xywhn(xywhn2xyxy(org_dataset.labels[im2_id][:, 1:], w=ratio[0]*w, h=ratio[1]*h, padw=padw, padh=padh), w=im2.shape[1], h=im2.shape[0], clip=True, eps=1e-3)[:5]

            # Do the copy-paste augmentation of some labels
            im_aug, labels, segments, cp_labels, cp_segments = copy_paste(
                    im, im_labels, segments, im2, im2_labels, im2_segments)

            # Create a pair of images that represents an example of change
            # detection. The images are identical, except the first image
            # contains some objects copy-pasted from a completely different
            # image. The second image does not have them. In the first image,
            # those objects are flagged as being 'deleted' (removed).
            example = copy.deepcopy(list(org_dataset[im_id]))
            # Undo the image transformation we did earlier
            example[0] = torch.tensor(cv2.cvtColor(im_aug, cv2.COLOR_RGB2BGR)).transpose(0, 1).transpose(0, 2)
            if cp_labels.size != 0: # If there we copy-pasted images (sometimes there aren't)
                # Re-normalize the bounding boxes for the copy-pasted objects
                cp_labels[:, 2:] = torch.tensor(xyxy2xywhn(
                        cp_labels[:, 2:], w=im1_w, h=im1_h, clip=True, eps=1e-3))
                # Combine copy-pasted objects with the objects that were originally there
                example[1] = torch.concatenate((example[1], torch.tensor(cp_labels))) 
                # Flag these as being the first image of the two-image pair
                example[1][:, 0] = 0 
                # Create masks from segments of the copy-pasted objects
                cp_masks = polygons2masks(im.shape[:2], cp_segments, color=1,
                        downsample_ratio=org_dataset.downsample_ratio)
                # Combine copy-pasted masks with the original masks
                example[4] = torch.concatenate((example[4], torch.tensor(cp_masks)))
                # Flag the added objects as being deletions, but not additions (because they disappear between the first image and the second)
                example[5] = torch.concatenate((example[5], torch.tensor([False]*len(cp_labels))))
                example[6] = torch.concatenate((example[6], torch.tensor([True]*len(cp_labels))))
            # The second image in the pair is an identical copy of the first but without the copy-pasted objects
            example2 = copy.deepcopy(list(org_dataset[im_id]))
            example2[1][:, 0] = 1
            self.paired_items.append((example, example2))

            # Create another pair where the copy-pasted item(s) occur in the second image. Do the same logic as above but flag them as additions instead of deletions.
            example = copy.deepcopy(list(org_dataset[im_id]))
            example[1][:, 0] = 0
            example2 = copy.deepcopy(list(org_dataset[im_id]))
            example2[0] = torch.tensor(cv2.cvtColor(im_aug, cv2.COLOR_RGB2BGR)).transpose(0, 1).transpose(0, 2)
            if cp_labels.size != 0:
                cp_labels[:, 2:] = torch.tensor(xyxy2xywhn(cp_labels[:, 2:], w=im1_w, h=im1_h))
                example2[1] = torch.concatenate((example2[1], torch.tensor(cp_labels)))
                example2[1][:, 0] = 1
                cp_masks = polygons2masks(im.shape[:2], cp_segments, color=1, downsample_ratio=org_dataset.downsample_ratio)
                example[4] = torch.concatenate((example[4], torch.tensor(cp_masks)))
                example[5] = torch.concatenate((example[5], torch.tensor([True]*len(cp_labels))))
                example[6] = torch.concatenate((example[6], torch.tensor([False]*len(cp_labels))))
            self.paired_items.append((example, example2))

        self.labels = org_dataset.labels
        self.shapes = org_dataset.shapes

    def __getitem__(self, i):
        return LoadImagesAndLabelsAndMasks.collate_fn(self.paired_items[i])

    def __len__(self):
        return len(self.paired_items)

class FramePairDataset(Dataset):
    def __init__(self, org_dataset):

        def sort_key(item):
            """ Extract frame ID so we can sort frames chronologically."""
            item_id = int(re.match(r'.*-(\d+).png', item[2])[1])
            return item_id

        self.org_dataset = org_dataset
        camera_items = collections.defaultdict(list)
        sorted_items = sorted(self.org_dataset, key=sort_key)
        # Gather a list of sorted frames for each camera
        for item in sorted_items:
            cam_id = re.match(r'cam(\d+)-.*', Path(item[2]).name)[1]
            camera_items[cam_id].append(item)

        # For each camera, pair adjacent frames, then unify them all into one
        # list at the end.
        self.paired_items = []
        for cam_id in camera_items:
            items_copy = copy.deepcopy(camera_items[cam_id])
            for item in items_copy:
                # Indicate that this is actually the second example.
                item[1][:, 0] = 1
            self.paired_items.extend(list(zip(camera_items[cam_id], items_copy[1:])))
            # Create a set of pairs of frames where the latter frame comes
            # first
            swap_pairs = list(zip(camera_items[cam_id][1:], items_copy))
            # Then make the dels the adds and the adds the dels.
            swap_pairs = [(frame_b[:-2] + (frame_b[-1],) + (frame_b[-2],),
                           frame_a[:-2] + (frame_a[-1],) + (frame_a[-2],))
                           for (frame_b, frame_a) in swap_pairs]
            self.paired_items.extend(swap_pairs)
        self.labels = self.org_dataset.labels
        self.shapes = self.org_dataset.shapes

    def __getitem__(self, i):
        return LoadImagesAndLabelsAndMasks.collate_fn(self.paired_items[i])

    def __len__(self):
        return len(self.paired_items)


class LoadImagesAndLabelsAndMasks(LoadImagesAndLabels):  # for training/testing

    def __init__(
        self,
        path,
        img_size=640,
        batch_size=16,
        augment=False,
        hyp=None,
        rect=False,
        image_weights=False,
        cache_images=False,
        single_cls=False,
        stride=32,
        pad=0,
        prefix="",
        downsample_ratio=1,
        overlap=False,
        data_dict=None,
    ):
        super().__init__(path, img_size, batch_size, augment, hyp, rect, image_weights, cache_images, single_cls,
                         stride, pad, prefix)
        self.downsample_ratio = downsample_ratio
        self.overlap = overlap
        with open(Path(path).parent / 'categories.json') as f:
            obj = json.loads(f.read())
            self.label2id = {item['name']: item['id']
                             for item in obj['categories']}
            self.id2label = {item['id']: item['name']
                             for item in obj['categories']}


        # Normalize the labels and convert the label IDs assigned by COCO to
        # reflect what is in the --data configuration json.
        self.datadict_label2id = {label: id_ for id_, label in
                                  data_dict['names'].items()}

    def get_raw_obj_id(self, id_):
        """ Given an object ID, convert it into an ID indicating the 'raw'
        object. For example, if the label 'del_can' has an ID in
        categories.json as 15, and 'can' has
        an ID 3 in the data YAML, then convert input 15 into 3. That is, once
        we discount the deletion/addition status what is the ID of the object
        class?
        """
        label = self.id2label[id_]
        if 'add_' in label or 'del_' in label:
            raw_label = re.match(r'.*_(.+)', label)[1]
            return self.datadict_label2id[raw_label]
        return self.datadict_label2id[label]

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        masks = []
        if mosaic:
            # Load mosaic
            img, labels, segments = self.load_mosaic(index)
            shapes = None

            # MixUp augmentation
            if random.random() < hyp["mixup"]:
                img, labels, segments = mixup(img, labels, segments, *self.load_mosaic(random.randint(0, self.n - 1)))

        else:
            # Load image
            img, (h0, w0), (h, w) = self.load_image(index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            # [array, array, ....], array.shape=(num_points, 2), xyxyxyxy
            segments = self.segments[index].copy()
            if len(segments):
                for i_s in range(len(segments)):
                    segments[i_s] = xyn2xy(
                        segments[i_s],
                        ratio[0] * w,
                        ratio[1] * h,
                        padw=pad[0],
                        padh=pad[1],
                    )
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                img, labels, segments = random_perspective(
                    img,
                    labels,
                    segments=segments,
                    degrees=hyp["degrees"],
                    translate=hyp["translate"],
                    scale=hyp["scale"],
                    shear=hyp["shear"],
                    perspective=hyp["perspective"],
                    return_seg=True,
                )

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1e-3)
            if self.overlap:
                masks, sorted_idx = polygons2masks_overlap(img.shape[:2],
                                                           segments,
                                                           downsample_ratio=self.downsample_ratio)
                masks = masks[None]  # (640, 640) -> (1, 640, 640)
                labels = labels[sorted_idx]
            else:
                masks = polygons2masks(img.shape[:2], segments, color=1, downsample_ratio=self.downsample_ratio)

        masks = (torch.from_numpy(masks) if len(masks) else torch.zeros(1 if self.overlap else nl, img.shape[0] //
                                                                        self.downsample_ratio, img.shape[1] //
                                                                        self.downsample_ratio))
        # TODO: albumentations support
        if self.augment:
            # Albumentations
            # there are some augmentation that won't change boxes and masks,
            # so just be it for now.
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img, hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"])

            # Flip up-down
            if random.random() < hyp["flipud"]:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]
                    masks = torch.flip(masks, dims=[1])

            # Flip left-right
            if random.random() < hyp["fliplr"]:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]
                    masks = torch.flip(masks, dims=[2])

            # Cutouts  # labels = cutout(img, labels, p=0.5)

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        # Create per-label binary flags to indicate if the object is an
        # addition
        adds = torch.tensor([True if 'add_' in self.id2label[label.item()] else False
                             for label in labels_out[:, 1]])

        # Create per-label binary flags to indicate if the object will be
        # deleted
        dels = torch.tensor([True if 'del_' in self.id2label[label.item()] else False
                             for label in labels_out[:, 1]])
        # Normalize labels
        labels_out[:, 1] = torch.tensor([self.get_raw_obj_id(id_.item()) for id_ in labels_out[:, 1]])

        return (torch.from_numpy(img), labels_out, self.im_files[index],
                shapes, masks, adds, dels, pad, (w, h), ratio)

    def load_mosaic(self, index):
        # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
        labels4, segments4 = [], []
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y

        # 3 additional image indices
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            labels, segments = self.labels[index].copy(), self.segments[index].copy()

            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            labels4.append(labels)
            segments4.extend(segments)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, 1:], *segments4):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img4, labels4 = replicate(img4, labels4)  # replicate

        # Augment
        img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp["copy_paste"])
        img4, labels4, segments4 = random_perspective(img4,
                                                      labels4,
                                                      segments4,
                                                      degrees=self.hyp["degrees"],
                                                      translate=self.hyp["translate"],
                                                      scale=self.hyp["scale"],
                                                      shear=self.hyp["shear"],
                                                      perspective=self.hyp["perspective"],
                                                      border=self.mosaic_border)  # border to remove
        return img4, labels4, segments4

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes, masks, adds, dels, pad, (w, h), ratio = zip(*batch)  # transposed
        batched_masks = torch.cat(masks, 0)
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes, batched_masks, torch.cat(adds, 0), torch.cat(dels, 0)


def polygon2mask(img_size, polygons, color=1, downsample_ratio=1):
    """
    Args:
        img_size (tuple): The image size.
        polygons (np.ndarray): [N, M], N is the number of polygons,
            M is the number of points(Be divided by 2).
    """
    mask = np.zeros(img_size, dtype=np.uint8)
    polygons = np.asarray(polygons)
    polygons = polygons.astype(np.int32)
    shape = polygons.shape
    polygons = polygons.reshape(shape[0], -1, 2)
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = (img_size[0] // downsample_ratio, img_size[1] // downsample_ratio)
    # NOTE: fillPoly firstly then resize is trying the keep the same way
    # of loss calculation when mask-ratio=1.
    mask = cv2.resize(mask, (nw, nh))
    return mask


def polygons2masks(img_size, polygons, color, downsample_ratio=1):
    """
    Args:
        img_size (tuple): The image size.
        polygons (list[np.ndarray]): each polygon is [N, M],
            N is the number of polygons,
            M is the number of points(Be divided by 2).
    """
    masks = []
    for si in range(len(polygons)):
        mask = polygon2mask(img_size, [polygons[si].reshape(-1)], color, downsample_ratio)
        masks.append(mask)
    return np.array(masks)


def polygons2masks_overlap(img_size, segments, downsample_ratio=1):
    """Return a (640, 640) overlap mask."""
    masks = np.zeros((img_size[0] // downsample_ratio, img_size[1] // downsample_ratio),
            dtype=np.int32 if len(segments) > 255 else np.uint8)
    areas = []
    ms = []
    for si in range(len(segments)):
        mask = polygon2mask(
            img_size,
            [segments[si].reshape(-1)],
            downsample_ratio=downsample_ratio,
            color=1,
        )
        ms.append(mask)
        areas.append(mask.sum())
    areas = np.asarray(areas)
    index = np.argsort(-areas)
    ms = np.array(ms)[index]
    for i in range(len(segments)):
        mask = ms[i] * (i + 1)
        masks = masks + mask
        masks = np.clip(masks, a_min=0, a_max=i + 1)
    return masks, index
