import matplotlib.pyplot as plt
from PIL import Image
import glob
import spectral
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
import os
import pickle
import torch
from tqdm import tqdm
from skimage.morphology import binary_erosion
from skimage.morphology import disk
from scipy.signal import savgol_filter
from sklearn.metrics import balanced_accuracy_score
import cv2


black_list = [
    "1dai/WT_6_2019_10_07_13_08_01",
    "1dai/P22_6_2019_10_07_13_59_25",
    "4dai/PO1_2_2019_10_10_08_20_38",
    "4dai/PO1_4_2019_10_10_08_35_06",
    "4dai/PO1_5_2019_10_10_08_42_23",
    "4dai/PO1_7_2019_10_10_08_56_55",
    "4dai/PO22_3_2019_10_10_07_22_07",
    "4dai/PO22_7_2019_10_10_07_51_11"
]

genotypes_g = ["P01", "P22", "WT"]
labels_dai_g = ["1", "2", "3", "4", "5"]
inoculated_g = [0, 1]
np.random.seed(42)
torch.random.manual_seed(42)

wavelength = [
    247.778
    , 248.401
    , 249.024
    , 249.647
    , 250.269
    , 250.892
    , 251.515
    , 252.137
    , 252.76
    , 253.383
    , 254.005
    , 254.628
    , 255.251
    , 255.874
    , 256.496
    , 257.119
    , 257.742
    , 258.364
    , 258.987
    , 259.61
    , 260.232
    , 260.855
    , 261.478
    , 262.101
    , 262.723
    , 263.346
    , 263.969
    , 264.591
    , 265.214
    , 265.837
    , 266.46
    , 267.082
    , 267.705
    , 268.328
    , 268.95
    , 269.573
    , 270.196
    , 270.818
    , 271.441
    , 272.064
    , 272.687
    , 273.309
    , 273.932
    , 274.555
    , 275.177
    , 275.8
    , 276.423
    , 277.045
    , 277.668
    , 278.291
    , 278.914
    , 279.536
    , 280.159
    , 280.782
    , 281.404
    , 282.027
    , 282.65
    , 283.272
    , 283.895
    , 284.518
    , 285.141
    , 285.763
    , 286.386
    , 287.009
    , 287.631
    , 288.254
    , 288.877
    , 289.5
    , 290.122
    , 290.745
    , 291.368
    , 291.99
    , 292.613
    , 293.236
    , 293.858
    , 294.481
    , 295.104
    , 295.727
    , 296.349
    , 296.972
    , 297.595
    , 298.217
    , 298.84
    , 299.463
    , 300.085
    , 300.708
    , 301.331
    , 301.954
    , 302.576
    , 303.199
    , 303.822
    , 304.444
    , 305.067
    , 305.69
    , 306.313
    , 306.935
    , 307.558
    , 308.181
    , 308.803
    , 309.426
    , 310.049
    , 310.671
    , 311.294
    , 311.917
    , 312.54
    , 313.162
    , 313.785
    , 314.408
    , 315.03
    , 315.653
    , 316.276
    , 316.898
    , 317.521
    , 318.144
    , 318.767
    , 319.389
    , 320.012
    , 320.635
    , 321.257
    , 321.88
    , 322.503
    , 323.125
    , 323.748
    , 324.371
    , 324.994
    , 325.616
    , 326.239
    , 326.862
    , 327.484
    , 328.107
    , 328.73
    , 329.353
    , 329.975
    , 330.598
    , 331.221
    , 331.843
    , 332.466
    , 333.089
    , 333.711
    , 334.334
    , 334.957
    , 335.58
    , 336.202
    , 336.825
    , 337.448
    , 338.07
    , 338.693
    , 339.316
    , 339.938
    , 340.561
    , 341.184
    , 341.807
    , 342.429
    , 343.052
    , 343.675
    , 344.297
    , 344.92
    , 345.543
    , 346.166
    , 346.788
    , 347.411
    , 348.034
    , 348.656
    , 349.279
    , 349.902
    , 350.524
    , 351.147
    , 351.77
    , 352.393
    , 353.015
    , 353.638
    , 354.261
    , 354.883
    , 355.506
    , 356.129
    , 356.751
    , 357.374
    , 357.997
    , 358.62
    , 359.242
    , 359.865
    , 360.488
    , 361.11
    , 361.733
    , 362.356
    , 362.978
    , 363.601
    , 364.224
    , 364.847
    , 365.469
    , 366.092
    , 366.715
    , 367.337
    , 367.96
    , 368.583
    , 369.206
    , 369.828
    , 370.451
    , 371.074
    , 371.696
    , 372.319
    , 372.942
    , 373.564
    , 374.187
    , 374.81
    , 375.433
    , 376.055
    , 376.678
    , 377.301
    , 377.923
    , 378.546
    , 379.169
    , 379.791
    , 380.414
    , 381.037
    , 381.66
    , 382.282
    , 382.905
    , 383.528
    , 384.15
    , 384.773
    , 385.396
    , 386.019
    , 386.641
    , 387.264
    , 387.887
    , 388.509
    , 389.132
    , 389.755
    , 390.377
    , 391
    , 391.623
    , 392.246
    , 392.868
    , 393.491
    , 394.114
    , 394.736
    , 395.359
    , 395.982
    , 396.604
    , 397.227
    , 397.85
    , 398.473
    , 399.095
    , 399.718
    , 400.341
    , 400.963
    , 401.586
    , 402.209
    , 402.831
    , 403.454
    , 404.077
    , 404.7
    , 405.322
    , 405.945
    , 406.568
    , 407.19
    , 407.813
    , 408.436
    , 409.059
    , 409.681
    , 410.304
    , 410.927
    , 411.549
    , 412.172
    , 412.795
    , 413.417
    , 414.04
    , 414.663
    , 415.286
    , 415.908
    , 416.531
    , 417.154
    , 417.776
    , 418.399
    , 419.022
    , 419.644
    , 420.267
    , 420.89
    , 421.513
    , 422.135
    , 422.758
    , 423.381
    , 424.003
    , 424.626
    , 425.249
    , 425.872
    , 426.494
    , 427.117
    , 427.74
    , 428.362
    , 428.985
    , 429.608
    , 430.23
    , 430.853
    , 431.476
    , 432.099
    , 432.721
    , 433.344
    , 433.967
    , 434.589
    , 435.212
    , 435.835
    , 436.457
    , 437.08
    , 437.703
    , 438.326
    , 438.948
    , 439.571
    , 440.194
    , 440.816
    , 441.439
    , 442.062
    , 442.684
    , 443.307
    , 443.93
    , 444.553
    , 445.175
    , 445.798
    , 446.421
    , 447.043
    , 447.666
    , 448.289
    , 448.912
    , 449.534
    , 450.157
    , 450.78
    , 451.402
    , 452.025
    , 452.648
    , 453.27
    , 453.893
    , 454.516
    , 455.139
    , 455.761
    , 456.384
    , 457.007
    , 457.629
    , 458.252
    , 458.875
    , 459.497
    , 460.12
    , 460.743
    , 461.366
    , 461.988
    , 462.611
    , 463.234
    , 463.856
    , 464.479
    , 465.102
    , 465.725
    , 466.347
    , 466.97
    , 467.593
    , 468.215
    , 468.838
    , 469.461
    , 470.083
    , 470.706
    , 471.329
    , 471.952
    , 472.574
    , 473.197
    , 473.82
    , 474.442
    , 475.065
    , 475.688
    , 476.31
    , 476.933
    , 477.556
    , 478.179
    , 478.801
    , 479.424
    , 480.047
    , 480.669
    , 481.292
    , 481.915
    , 482.538
    , 483.16
    , 483.783
    , 484.406
    , 485.028
    , 485.651
    , 486.274
    , 486.896
    , 487.519
    , 488.142
    , 488.765
    , 489.387
    , 490.01
    , 490.633
    , 491.255
    , 491.878
    , 492.501
    , 493.123
    , 493.746
    , 494.369
    , 494.992
    , 495.614
    , 496.237
    , 496.86
    , 497.482
    , 498.105
    , 498.728
    , 499.35
    , 499.973
    , 500.596
    , 501.219
    , 501.841
]


class LeafDataset(Dataset):
    def __init__(self, data_path="/media/disk2/datasets/anna/Messungen/Current_UV_Gerste",
                 mode='train', test_size=0.95, genotype=None, inoculated=None, dai=None,
                 signature_pre_clip=0, signature_post_clip=200, max_num_balanced_inoculated=-1,
                 savgol_filter_params=(7, 3), num_samples_file=-1, n_splits=5, split=0,
                 superpixel=False, bags=False):
        print("Using wavelength ({}) from {} nm to {} nm".format(len(wavelength),
                                                                 wavelength[signature_pre_clip:-signature_post_clip][0],
                                                                 wavelength[signature_pre_clip:-signature_post_clip][
                                                                     -1]))
        self.wavelength = wavelength[signature_pre_clip:-signature_post_clip]
        self.max_num_balanced_inoculated = max_num_balanced_inoculated
        self.signature_clip = [signature_pre_clip, -signature_post_clip]
        self.num_samples_file = num_samples_file
        self.superpixel = superpixel
        self.bags = bags

        # load data
        base_path_dataset_parsed = os.path.join(data_path, "../../Downloads/UV_Gerste/parsed_data")
        self.data_memmaps, self.data = _load_preprocessed_data(data_path, base_path_dataset_parsed,
                                                               genotype=genotype,
                                                               inoculated=inoculated,
                                                               dai=dai,
                                                               max_num_balanced_inoculated=self.max_num_balanced_inoculated,
                                                               num_samples_file = self.num_samples_file,
                                                               superpixel=superpixel, bags=bags)
        if test_size == 0:
            train_index = np.arange(len(self.data))
            test_index = np.arange(len(self.data))
        else:
            sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0)

            splits = [(train, test) for train, test in sss.split(self.data, np.zeros([len(self.data)]))]
            train_index, test_index = splits[split]
            print("Splitting data for cross validation, n_splits: {}, split: {}".format(n_splits, split))

        if mode == 'train':
            self.indices = train_index
        elif mode == 'test':
            self.indices = test_index
        elif mode == 'eval':
            self.indices = train_index.tolist()
            self.indices += test_index.tolist()
            self.indices = np.array(self.indices)
        else:
            raise ValueError("mode not supported")

        self.savgol_filter_params = savgol_filter_params
        if bags:
            self.input_size = len(self.__getitem__(0)[0][1])
        else:
            self.input_size = len(self.__getitem__(0)[0])
        self.hyperparams = {
            'genotype': genotype,
            'inoculated': inoculated,
            'dai': dai,
            'signature_pre_clip': signature_pre_clip,
            'signature_post_clip': signature_post_clip,
            'test_size': test_size,
            'max_num_balanced_inoculated': max_num_balanced_inoculated,
            'input_size': self.input_size,
            'savgol_filter_params': self.savgol_filter_params,
            'num_samples_file': self.num_samples_file,
            'n_splits': n_splits,
            'split': split,
            'superpixel': superpixel
        }

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.superpixel:
            return self.getSuperpixel(idx)
        else:
            return self.getPixel(idx)

    def getPixel(self, idx):
        sample = self.data[self.indices[idx]]
        hs_img = self.data_memmaps[sample["path"]][0]
        """
        "hs_img": img,
        "hs_img_idx": len(data_hs)-1,
        "path": hs_img_path,
        "pos": (x, y),
        "label_genotype": bbox_obj_dict["label_genotype"],
        "label_dai": bbox_obj_dict["label_dai"],
        "label_inoculated": bbox_obj_dict["label_inoculated"],
        "label_obj": bbox_obj_dict["label_obj"],
        "label_running": bbox_obj_dict["label_running"],            
        """
        label = (sample["label_genotype"], sample["label_dai"], sample["label_inoculated"],
                 sample["label_obj"], sample["label_running"])
        res_sample = hs_img[sample["pos"][0], sample["pos"][1], :].squeeze()
        res_sample = self.normalize(res_sample)
        #
        # res_sample += -0.5
        return res_sample, label

    def getSuperpixel(self, idx):
        samples = self.data[self.indices[idx]]
        if self.bags:
            hs_img = self.data_memmaps[samples[0]["path"]][0]
            """
            "hs_img": img,
            "hs_img_idx": len(data_hs)-1,
            "path": hs_img_path,
            "pos": ((x0,x1), (y0,y1)),
            "mask": ...
            "label_genotype": bbox_obj_dict["label_genotype"],
            "label_dai": bbox_obj_dict["label_dai"],
            "label_inoculated": bbox_obj_dict["label_inoculated"],
            "label_obj": bbox_obj_dict["label_obj"],
            "label_running": bbox_obj_dict["label_running"],            
            """
            bag_instances, bag_labels = [], []
            for sample in samples:
                label = (sample["label_genotype"], sample["label_dai"], sample["label_inoculated"],
                         sample["label_obj"], sample["label_running"], sample['mask'])

                res_sample = hs_img[sample["pos"][0][0]:sample["pos"][0][1], sample["pos"][1][0]:sample["pos"][1][1], :]
                #res_sample = res_sample[sample["mask"].astype(int) == 1]
                #res_sample = np.mean(res_sample, axis=(0,))
                #res_sample = self.normalize(res_sample)
                #bag_instances.append(torch.Tensor(res_sample))
                bag_instances.append(torch.from_numpy(res_sample.transpose((2, 0, 1))))
                #bag_instances.append(res_sample)
                #bag_labels.append(label[5])
                bag_labels.append(label[2])
            bag_instances = torch.stack(bag_instances)
            bag_labels = torch.Tensor([np.max(bag_labels)])
        else:
            hs_img = self.data_memmaps[samples["path"]][0]
            bag_labels = (samples["label_genotype"], samples["label_dai"], samples["label_inoculated"],
                     samples["label_obj"], samples["label_running"])

            res_sample = hs_img[samples["pos"][0][0]:samples["pos"][0][1], samples["pos"][1][0]:samples["pos"][1][1], :]
            res_sample = res_sample[samples["mask"].astype(int) == 1]
            res_sample = np.mean(res_sample, axis=(0,))
            bag_instances = self.normalize(res_sample)
        #res_sample = self.crop(res_sample, (25, 750))
        #res_sample = torch.from_numpy(res_sample.transpose((2, 0, 1)))
        #
        # res_sample += -0.5
        return bag_instances, bag_labels

    def crop(self, res_sample, tw, th):
        w, h = res_sample.shape[:-1]
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))

        return res_sample[x1:x1+tw, y1:y1+th, :]

    def normalize(self, res_sample):
        res_sample = res_sample[self.signature_clip[0]:self.signature_clip[1]]
        res_sample = savgol_filter(res_sample, self.savgol_filter_params[0], self.savgol_filter_params[1])

        return res_sample

    def test_full_image(self, forward, mask_erosion_value=5, num_images_per_class=2):
        """
        data_hs[hs_img_path] = (img, {"label_genotype": bbox_obj_dict["label_genotype"],
                                          "label_dai": bbox_obj_dict["label_dai"],
                                          "label_inoculated": bbox_obj_dict["label_inoculated"],
                                          "label_running": bbox_obj_dict["label_running"]})

        :return:
        """

        selected_inoculated = self.hyperparams["inoculated"]*num_images_per_class
        #print(list(self.data_memmaps.keys())[0])
        #exit()

        res_samples = []
        for image_key in sorted(list(self.data_memmaps.keys())):
            sample = self.data_memmaps[image_key]
            sample_meta = sample[1]
            #print(sample_meta["label_genotype"], sample_meta["label_inoculated"], print(sample_meta["label_dai"]))
            if self.hyperparams["genotype"] is None or sample_meta["label_genotype"] in self.hyperparams["genotype"]:
                if self.hyperparams["dai"] is None or sample_meta["label_dai"] in self.hyperparams["dai"]:
                    for sample_bbox_filename in sample_meta['bbox_filename']:
                        if self.hyperparams["inoculated"] is None or (sample_meta["label_inoculated"] in self.hyperparams["inoculated"] and sample_meta[
                                "label_inoculated"] in selected_inoculated):
                            selected_inoculated.remove(sample_meta["label_inoculated"])
                            img = sample[0]
                            bbox_obj_dict = pickle.load(open(sample_bbox_filename, "rb"))
                            mask = bbox_obj_dict["mask"]
                            if mask_erosion_value > 0:
                                mask = cv2.copyMakeBorder(mask, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=0)
                                selem = disk(mask_erosion_value)
                                mask = binary_erosion(mask, selem)
                                mask = mask[30:-30, 30:-30]
                            [(min_y, min_x), (max_y, max_x)] = bbox_obj_dict["bbox"]
                            classes = np.zeros(shape=[img.shape[0], img.shape[1]]).squeeze()
                            classes[min_x:max_x, min_y:max_y] = mask
                            masked_pred_img = self.pixelwise_eval(img, classes, bbox_obj_dict["bbox"], forward)

                            res_samples.append({'img': img[min_x:max_x, min_y:max_y, :],
                                                'pred': masked_pred_img[min_x:max_x, min_y:max_y],
                                                'mask': mask,
                                                'label': sample_meta["label_inoculated"]
                                                })

                            if len(selected_inoculated) == 0:
                                break
        return res_samples

    def pixelwise_eval(self, img, masked_img, bbox, forward):
        [(min_y, min_x), (max_y, max_x)] = bbox
        masked_pred_img = np.zeros(shape=[img.shape[0], img.shape[1]]).squeeze()
        #masked_pred_img[:,:] = -1
        for x in tqdm(range(min_x, max_x)):
            for y in range(min_y, max_y):
                # if pixel is in segmentation component add
                if masked_img[x, y] == 1:
                    signature = self.normalize(img[x, y])
                    signature = np.expand_dims(signature, axis=0)
                    out = forward(signature)
                    out = out.detach().cpu().numpy().squeeze()
                    out = np.argmax(out)
                    masked_pred_img[x, y] = out + 1
        return masked_pred_img


def _load_preprocessed_data(base_path_dataset, base_path_dataset_parsed, genotype=None, inoculated=None, dai=None,
                            max_num_balanced_inoculated=-1, mask_erosion_value=5, num_samples_file=-1, superpixel=False,
                            bags=False):
    current_path = os.path.join(base_path_dataset_parsed, "*.p")

    filenames = sorted(list(set(glob.glob(current_path))))

    data_hs = dict()
    data_pos = []

    balance_inoculated_label_max_num = np.inf if max_num_balanced_inoculated == -1 else max_num_balanced_inoculated
    balance_inoculated_label_current = [0, 0]
    cnt_files_used = 0
    for filename in tqdm(filenames):
        data_pos_file = []
        bbox_obj_dict = pickle.load(open(filename, "rb"))

        if genotype is not None and bbox_obj_dict["label_genotype"] not in genotype:
            continue
        if inoculated is not None and bbox_obj_dict["label_inoculated"] not in inoculated:
            continue
        if dai is not None and bbox_obj_dict["label_dai"] not in dai:
            continue
        [(min_y, min_x), (max_y, max_x)] = bbox_obj_dict["bbox"]

        hs_img_path = os.path.join(os.path.join(base_path_dataset, "{}dai".format(bbox_obj_dict["label_dai"])),
                                   bbox_obj_dict["filename"] + "/data.hdr")
        img = spectral.open_image(hs_img_path)
        classes = np.zeros(shape=[img.shape[0], img.shape[1]]).squeeze()

        mask = bbox_obj_dict["mask"]
        if mask_erosion_value > 0:
            mask = cv2.copyMakeBorder(mask, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=0)
            selem = disk(mask_erosion_value)
            mask = binary_erosion(mask, selem)
            mask = mask[30:-30, 30:-30]
        classes[min_x:max_x, min_y:max_y] = mask

        # view = spectral.imshow(img, classes=classes)
        # view.set_display_mode('overlay')
        # view.class_alpha = 0.5
        # input("")
        # print(bbox_obj_dict["bbox"])
        # print(bbox_obj_dict["filename"])
        # tmp_mask = bbox_obj_dict["mask"]
        # plt.figure(figsize=(20, 10))
        # plt.imshow(bbox_obj_dict["mask"])
        # plt.figure(figsize=(20, 10))
        # plt.imshow(bbox_obj_dict["image"])

        # plt.figure(figsize=(20, 10))
        # plt.imshow(tmp_mask_erod)

        # plt.show()
        # exit()
        # 1 / 0
        if hs_img_path not in list(data_hs.keys()):
            data_hs[hs_img_path] = (img, {"label_genotype": bbox_obj_dict["label_genotype"],
                                          "label_dai": bbox_obj_dict["label_dai"],
                                          "label_inoculated": bbox_obj_dict["label_inoculated"],
                                          "label_running": bbox_obj_dict["label_running"],
                                          "bbox_filename": [filename]})
        else:
            data_hs[hs_img_path][1]["bbox_filename"].append(filename)
            #print(data_hs[hs_img_path][1]["bbox_filename"])
        # in bbox
        x_diff = max_x - min_x
        new_x = int(round((x_diff - 25) / 2.))
        min_x = min_x + new_x
        max_x = min_x + 25
        y_diff = max_y - min_y
        new_y = int(round((y_diff - 631) / 2.))
        min_y = min_y + new_y
        max_y = min_y + 631
        if superpixel:
            step_size = 7
            for y in range(min_y + step_size, max_y - step_size, step_size*2):
                max_x_ = max_x + 1 if max_x + 1 < classes.shape[0] else max_x # fix for extreme case on border if bbox
                super_pixel_x_range = np.arange(min_x, max_x_)
                #print(classes.shape, super_pixel_x_range)
                test_classes = classes[super_pixel_x_range, y]
                if np.sum(test_classes) > 0:
                    data_pos_file.append(
                        {
                            # "hs_img_idx": len(data_hs) - 1,
                            "path": hs_img_path,
                            "pos": ((super_pixel_x_range[0], super_pixel_x_range[-1]), (y-step_size, y + step_size+1)),
                            "mask": classes[super_pixel_x_range[0]:super_pixel_x_range[-1],
                                    y - step_size:y + step_size + 1],
                            "label_genotype": bbox_obj_dict["label_genotype"],
                            "label_dai": bbox_obj_dict["label_dai"],
                            "label_inoculated": bbox_obj_dict["label_inoculated"],
                            "label_obj": bbox_obj_dict["label_obj"],
                            "label_running": bbox_obj_dict["label_running"],
                        }
                    )
        else:
            for x in range(min_x, max_x):
                for y in range(min_y, max_y):
                    # if pixel is in segmentation component add
                    if classes[x, y] == 1:
                            data_pos_file.append(
                                {
                                    # "hs_img_idx": len(data_hs) - 1,
                                    "path": hs_img_path,
                                    "pos": (x, y),
                                    "label_genotype": bbox_obj_dict["label_genotype"],
                                    "label_dai": bbox_obj_dict["label_dai"],
                                    "label_inoculated": bbox_obj_dict["label_inoculated"],
                                    "label_obj": bbox_obj_dict["label_obj"],
                                    "label_running": bbox_obj_dict["label_running"],
                                }
                            )

        selected_file_samples = np.arange(len(data_pos_file))
        #print(len(data_pos_file))
        if num_samples_file > 0:
            np.random.shuffle(selected_file_samples)
            selected_file_samples = selected_file_samples[:num_samples_file]

        if balance_inoculated_label_current[bbox_obj_dict["label_inoculated"]] + len(selected_file_samples) <= balance_inoculated_label_max_num:
            balance_inoculated_label_current[bbox_obj_dict["label_inoculated"]] += len(selected_file_samples)

            selected_file_samples = [d for (i, d) in enumerate(data_pos_file) if i in selected_file_samples]
            if bags:
                data_pos += [selected_file_samples]
            else:
                data_pos += selected_file_samples

            cnt_files_used += 1

    print("Leafs used", cnt_files_used, "total #sample", len(data_pos))
    print("Dataset num sample per class", balance_inoculated_label_current)
    return data_hs, data_pos


def _load_preprocessed_test():
    base_path_dataset = "/media/disk2/datasets/anna/Messungen/Current_UV_Gerste"
    base_path_dataset_parsed = "/media/disk2/datasets/anna/Messungen/Current_UV_Gerste/parsed_data"

    current_path = os.path.join(base_path_dataset_parsed, "*.p")

    filenames = sorted(list(set(glob.glob(current_path))))
    bbox_obj_dict = pickle.load(open(filenames[0], "rb"))

    """
    bbox_obj_dict["id"] = "_"
    bbox_obj_dict["label_genotype"] = current_label_genotype_
    bbox_obj_dict["label_dai"] = current_label_dai
    bbox_obj_dict["label_inoculated"] = label_inoculated
    bbox_obj_dict["label_obj"] = {"label": obj_label, "idx": obj_label_idx}
    bbox_obj_dict["label_running"] = filename_idx
    bbox_obj_dict["filename"] = os.path.basename(os.path.dirname(fp_img))
    [(min_y, min_x), (max_y, max_x)] = bbox_pixels[obj_label_idx]
    bbox_obj_dict["bbox"] = [(min_y, min_x), (max_y, max_x)]
    bbox_obj_dict["mask"] = labeledimg[min_x:max_x, min_y:max_y]
    bbox_obj_dict["image"] = img[min_x:max_x, min_y:max_y]
    """

    [(min_y, min_x), (max_y, max_x)] = bbox_obj_dict["bbox"]

    # TODO load HS and extract labeled
    img = spectral.open_image(os.path.join(os.path.join(base_path_dataset, "{}dai".format(bbox_obj_dict["label_dai"])),
                                           bbox_obj_dict["filename"] + "/data.hdr"))
    classes = np.zeros_like(img[:, :, 0]).squeeze()
    print([(min_y, min_x), (max_y, max_x)])
    print(classes.shape)
    classes[min_x:max_x, min_y:max_y] = bbox_obj_dict["mask"]

    view = spectral.imshow(img, classes=classes)
    view.set_display_mode('overlay')
    view.class_alpha = 0.5
    input("")
    print(bbox_obj_dict["bbox"])
    print(bbox_obj_dict["filename"])
    """plt.figure(figsize=(20, 10))
    plt.imshow(bbox_obj_dict["mask"])
    plt.figure(figsize=(20, 10))
    plt.imshow(bbox_obj_dict["image"])
    plt.show()"""

    # print(filenames)


def _test_dataset():
    dataset = LeafDataset()
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for i, (sample, y) in enumerate(tqdm(data_loader)):
        # print(len(sample.squeeze()))
        plt.plot(sample.squeeze().cpu().detach().numpy())
        if i > 500:
            break

    plt.show()


def _test_dataset_WT():
    colors = ['g', 'r']
    dataset = LeafDataset(genotype=["WT"], inoculated=None, dai=["5"], signature_pre_clip=100)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    samples = [[], []]
    for i, (sample, y) in enumerate(tqdm(data_loader)):
        if y[2] == 0:
            if len(samples[0]) < 200:
                samples[0].append(sample)
        else:
            if len(samples[1]) < 200:
                samples[1].append(sample)
        if len(samples[0]) >= 20 and len(samples[1]) >= 20:
            break
        # label_inoculated = y[2]
        # print(len(sample.squeeze()))
    # for sample in samples[0]:
    #    plt.plot(sample.squeeze().cpu().detach().numpy(), c=colors[0])
    # for sample in samples[1]:
    #    plt.plot(sample.squeeze().cpu().detach().numpy(), c=colors[1])
    std = torch.std(torch.stack(samples[0]), dim=(0,)).squeeze().cpu().detach().numpy()

    plt.plot(torch.mean(torch.stack(samples[0]), dim=(0,)).squeeze().cpu().detach().numpy(), c=colors[0])
    plt.plot(torch.mean(torch.stack(samples[0]), dim=(0,)).squeeze().cpu().detach().numpy() + std, c=colors[0])
    plt.plot(torch.mean(torch.stack(samples[0]), dim=(0,)).squeeze().cpu().detach().numpy() - std, c=colors[0])

    std = torch.std(torch.stack(samples[1]), dim=(0,)).squeeze().cpu().detach().numpy()
    plt.plot(torch.mean(torch.stack(samples[1]), dim=(0,)).squeeze().cpu().detach().numpy(), c=colors[1])
    plt.plot(torch.mean(torch.stack(samples[1]), dim=(0,)).squeeze().cpu().detach().numpy() + std, c=colors[1])
    plt.plot(torch.mean(torch.stack(samples[1]), dim=(0,)).squeeze().cpu().detach().numpy() - std, c=colors[1])

    plt.show()


def _preprocessing():
    # based_on = "png"
    based_on = "hs"
    base_path_dataset = "/media/disk2/datasets/anna/Messungen/Current_UV_Gerste"
    base_path_dataset_parsed = "/media/disk2/datasets/anna/Messungen/Current_UV_Gerste/parsed_data"

    labels_genotype = [["P01", "PO1"], ["P22", "PO22"], ["WT"]]
    labels_dai = ["1", "2", "3", "4", "5"]
    # fp_img = "/media/disk2/datasets/anna/Messungen/Current_UV_Gerste/1dai/P01_1-_2019_10_07_10_00_47/image.png"
    # img = Image.open(fp_img)
    # img = np.array(img)[:, :, 0]

    # plt.figure()
    # plt.imshow(img)
    # plt.colorbar()

    # bboximg, bbox_pixels, labeledimg, obj_labels = comp_bbox(img, thresh=25)
    # save complete bbox image

    # save (pickle dict) single bbox mask with meta information position in image

    # save

    for current_label_genotype in tqdm(labels_genotype):
        current_label_genotype_ = current_label_genotype[0]
        for current_label_dai in tqdm(labels_dai):
            filenames = list()
            for current_label_genotype_check in current_label_genotype:
                current_path = os.path.join(base_path_dataset,
                                            "{}dai/{}*/data.hdr".format(current_label_dai,
                                                                        current_label_genotype_check))
                filenames += sorted(list(set(glob.glob(current_path))))

            for filename_idx, filename in enumerate(filenames):
                tmp = False
                for blacked_listed_filename in black_list:
                    if blacked_listed_filename in filename:
                        tmp = True
                if tmp:
                    continue
                if based_on == "png":
                    fp_img = filename.replace("data.hdr", "image.png")
                    img = Image.open(fp_img)
                    img = np.array(img)[:, :, 0]
                    bboximg, bbox_pixels, labeledimg, obj_labels = comp_bbox(img, thresh=30)
                elif based_on == "hs":
                    fp_img = filename
                    img_hs = spectral.open_image(fp_img)
                    img_hs = img_hs[:, :, :]
                    img = np.mean(img_hs, axis=2)
                    bboximg, bbox_pixels, labeledimg, obj_labels = comp_bbox_pseudo_rgb(img, thresh=np.mean(img))
                else:
                    raise ValueError("unkown format")
                # thresh = np.mean(img) + 5.
                # print(np.std(img))

                labeledimg[labeledimg != 0] = 1

                label_inoculated = 0 if "-_" in os.path.basename(os.path.dirname(filename)) else 1
                filepath_parsed = os.path.join(base_path_dataset_parsed,
                                               "{}label_{}dai_{}type_{}idx.png".format(
                                                   label_inoculated,
                                                   current_label_dai,
                                                   current_label_genotype_,
                                                   filename_idx
                                               ))
                plt.imshow(bboximg)
                plt.savefig(filepath_parsed, bbox_inches='tight')
                plt.close()
                plt.clf()
                # continue
                for obj_label_idx, obj_label in enumerate(obj_labels):
                    bbox_obj_dict = dict()
                    bbox_obj_dict["id"] = "_"
                    bbox_obj_dict["label_genotype"] = current_label_genotype_
                    bbox_obj_dict["label_dai"] = current_label_dai
                    bbox_obj_dict["label_inoculated"] = label_inoculated
                    bbox_obj_dict["label_obj"] = {"label": obj_label, "idx": obj_label_idx}
                    bbox_obj_dict["label_running"] = filename_idx
                    bbox_obj_dict["filename"] = os.path.basename(os.path.dirname(fp_img))
                    bbox_obj_dict["file_path"] = fp_img
                    [(min_y, min_x), (max_y, max_x)] = bbox_pixels[obj_label_idx]
                    bbox_obj_dict["bbox"] = [(min_y, min_x), (max_y, max_x)]
                    bbox_obj_dict["mask"] = labeledimg[min_x:max_x, min_y:max_y]
                    bbox_obj_dict["image"] = img[min_x:max_x, min_y:max_y]
                    # plt.figure(figsize=(20, 10))
                    # plt.imshow(bbox_obj_dict["mask"])
                    # plt.figure(figsize=(20, 10))
                    # plt.imshow(bbox_obj_dict["image"])
                    # plt.show()
                    pickle.dump(bbox_obj_dict,
                                open(filepath_parsed.replace(".png", "_{}obj.p".format(obj_label_idx)),
                                     "wb"))
                    # if os.path.isfile(filepath_parsed.replace("/parsed_data/",
                    #                                          "/parsed_data_debugControl/").replace(".png",
                    #                                                                                "_{}obj.p".format(
                    #                                                                                    obj_label_idx))):
                    #    pickle.dump(bbox_obj_dict,
                    #                open(filepath_parsed.replace(".png", "_{}obj.p".format(obj_label_idx)),
                    #                     "wb"))

    # plt.figure(figsize=(20, 10))
    # plt.imshow(labeledimg)
    # plt.figure(figsize=(20, 10))
    # plt.imshow(bboximg)
    # plt.show()


def _test_hyperparams_dataset():
    from uv_dataset.hyperparams.hyperparams import get_param_class, dict_classes_keys
    for class_key in sorted(dict_classes_keys):
        #models_to_test = [x for x in models_to_test_all if class_key in x]

        param_class = get_param_class(class_key)
        print(class_key)
        dataset_train = LeafDataset(data_path="/media/disk2/datasets/anna/Messungen/Current_UV_Gerste",
                                    genotype=param_class.genotype, inoculated=param_class.inoculated,
                                    dai=param_class.dai,
                                    test_size=0,
                                    signature_pre_clip=param_class.signature_pre_clip,
                                    signature_post_clip=param_class.signature_post_clip,
                                    max_num_balanced_inoculated=param_class.max_num_balanced_inoculated,
                                    num_samples_file=param_class.num_samples_file,
                                    mode='train',
                                    n_splits=5,
                                    split=0,
                                    superpixel=param_class.superpixel)

if __name__ == '__main__':
    _test_hyperparams_dataset()
    #pass
    # _test_dataset_WT()
    # _load_preprocessed_test()
    # _preprocessing()
