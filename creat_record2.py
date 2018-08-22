import json
import io
import cv2
import numpy as np
import random
import math
from scipy.ndimage import filters
from PIL import Image, ImageEnhance
from sklearn.neighbors import NearestNeighbors
import paddle.fluid as fluid
import paddle
import time
import sys


def json_read(filename_list):
    annotation_1s = {}
    for filename in filename_list:
        len_num = len(annotation_1s)
        f = io.open(filename, encoding='utf-8')
        baidu_data = json.load(f)
        info = baidu_data["info"]
        annotations = baidu_data["annotations"]  # [0]["name"]
        lenth = len(annotations)
        for annotation in annotations:
            if annotation["annotation"] != [] and len(annotation["annotation"][0]) == 2:
                annotation["type"] = 1
            else:
                annotation["type"] = 0
            id = annotation["id"] + len_num
            annotation_1s[id] = {"name": annotation["name"], "num": annotation["num"], "type": annotation["type"],
                                 "ignore_region": annotation["ignore_region"], "annotation": annotation["annotation"]}
    # ##print(annotation_1s)
    return annotation_1s


def dataugment_cut(annotation, image_path, h_new, w_new):
    for id in annotation.keys():
        if annotation[id]['name'].split('/')[-1] == str(image_path.split('/')[-1]):
            anno = annotation[id]
            break
    filp = []
    test = Image.open(image_path)
    test = randomColor(test)
    test = np.asarray(test)
    orimage = test.copy()
    affrat = random.uniform(0.7, 1.25)
    height, width = orimage.shape[0], orimage.shape[1]
    center = (width / 2., height / 2.)
    halfl_w = min(width - center[0], (width - center[0]) / 1.25 * affrat)
    halfl_h = min(height - center[1], (height - center[1]) / 1.25 * affrat)
    y_min, y_max = round(center[1] - halfl_h), round(center[1] + halfl_h + 1)
    x_min, x_max = round(center[0] - halfl_w), round(center[0] + halfl_w + 1)
    image_cut = orimage[int(center[1] - halfl_h):int(center[1] + halfl_h + 1),
                int(center[0] - halfl_w):int(center[0] + halfl_w + 1)]
    annota = anno['annotation']
    if anno['type'] == 0:
        for i in range(len(annota)):
            y, x, w, h = annota[i]['y'], annota[i]['x'], annota[i]['w'], annota[i]['h']
            x = math.floor(x + w / 2)
            y = math.floor(y + h / 5)
            if x_min < x < x_max and y_min < y < y_max:
                x, y = round(float((x - x_min)) / float((x_max - x_min)) * w_new), round(
                    float((y - y_min)) / float((y_max - y_min)) * h_new)
                filp.append([x, y])
    if anno['type'] == 1:
        for i in range(len(annota)):
            y, x = annota[i]['y'], annota[i]['x']
            if x_min < x < x_max and y_min < y < y_max:
                x, y = round(float((x - x_min)) / float((x_max - x_min)) * w_new), round(
                    float((y - y_min)) / float((y_max - y_min)) * h_new)
                filp.append([x, y])
    image_cut = Image.fromarray(image_cut)
    image_cut = np.asarray(image_cut.resize((w_new, h_new), Image.ANTIALIAS))
    image_cut.setflags(write=1)
    ##print("image_cut", np.shape(image_cut))
    ##print("filp", filp)
    return image_cut, filp


def dataugment_filp(annotation, image_path, h_new, w_new):
    # ##print(len(annotation.keys()))
    for id in annotation.keys():
        if annotation[id]['name'].split('/')[-1] == str(image_path.split('/')[-1]):
            anno = annotation[id]
            break
    filp = []
    test = Image.open(image_path)
    test = randomColor(test)
    test = np.asarray(test)
    gotData = test.copy()
    orimage = gotData.transpose(0, 1, 2)
    height, width = orimage.shape[0], orimage.shape[1]
    newing = cv2.flip(test, 1)
    annota = anno['annotation']
    if anno['type'] == 0:
        for i in range(len(annota)):
            y, x, w, h = annota[i]['y'], annota[i]['x'], annota[i]['w'], annota[i]['h']
            filp.append([round((width - x - w / 2.0) / width * w_new), round((y + h / 5.0) / height * 1.0 * h_new)])
    if anno['type'] == 1:
        for i in range(len(annota)):
            y, x = annota[i]['y'], annota[i]['x']
            filp.append([round((width - x) * 1.0 / width * 1.0 * w_new), round(y * 1.0 / height * 1.0 * h_new)])
    newing = Image.fromarray(newing)
    newing = np.asarray(newing.resize((w_new, h_new), Image.ANTIALIAS))
    newing.setflags(write=1)
    return newing, filp


def bbox_convert_2_point(x, y, w, h):
    x = x + w * 1.0 / 2
    y = y + h * 1.0 / 5
    return round(x), round(y)


def create_density(gts, d_map_h, d_map_w):
    # res = np.zeros(shape=[d_map_h, d_map_w])
    ##print("gts", np.shape(gts))
    # pts = np.array(list(zip(np.nonzero(res)[1], np.nonzero(res)[0])))
    pts = gts
    ##print ("np.shape pts", np.shape(pts))
    neighbors = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', leaf_size=1200)
    neighbors.fit(pts.copy())
    distances, _ = neighbors.kneighbors()
    ##print("1111", distances.shape)
    map_shape = [int(d_map_h), int(d_map_w)]
    density = np.zeros(shape=map_shape, dtype=np.float32)
    sigmas = distances.sum(axis=1) * 0.075
    ##print("2222", sigmas.shape)
    for i in range(len(pts)):
        pt = pts[i]
        pt2d = np.zeros(shape=map_shape, dtype=np.float32)
        ##print("pt", pt)
        pt2d[int(pt[1]) - 1][int(pt[0]) - 1] = 1
        density += filters.gaussian_filter(pt2d, sigmas[i], mode='constant')
    return density


def density_map(image_path, annotation_ls, zuofang, h_new, w_new, img, filp, flag):
    if flag == 0:
        gt = []
        img = Image.open(image_path)
        img = randomColor(img)
        img = np.asarray(img)
        shape = img.shape
        img = Image.fromarray(img)
        img = np.asarray(img.resize((w_new, h_new), Image.ANTIALIAS))
        img.setflags(write=1)
        # img = np.resize(img, (h_new, w_new, 3))
        for id in annotation_ls.keys():
            if annotation_ls[id]['name'].split('/')[-1] == str(image_path.split('/')[-1]):
                anno = annotation_ls[id]
                break
        if anno['annotation'] != []:
            if len(anno['annotation'][0]) == 2:
                for i in range(len(anno['annotation'])):
                    x, y = anno['annotation'][i]['x'], anno['annotation'][i]['y']
                    gt.append([x * 1.0 / shape[1] * 1.0 * w_new, y * 1.0 / shape[0] * 1.0 * h_new])
            else:
                for i in range(len(anno['annotation'])):
                    x, y, w, h = anno['annotation'][i]['x'], anno['annotation'][i]['y'], anno['annotation'][i]['w'], \
                                 anno['annotation'][i]['h']
                    x, y = bbox_convert_2_point(x, y, w, h)
                    gt.append([x * 1.0 / shape[1] * 1.0 * w_new, y * 1.0 / shape[0] * 1.0 * h_new])
        d_map_h = math.floor(math.floor(float(img.shape[0]) / zuofang) / zuofang)
        d_map_w = math.floor(math.floor(float(img.shape[1]) / zuofang) / zuofang)
        den_map = create_density(np.array(gt) / zuofang / zuofang, d_map_h, d_map_w)
    else:
        d_map_h = math.floor(math.floor(float(img.shape[0]) / zuofang) / zuofang)
        d_map_w = math.floor(math.floor(float(img.shape[1]) / zuofang) / zuofang)
        den_map = create_density(np.array(filp) / zuofang / zuofang, d_map_h, d_map_w)
    return den_map, img


def image_process_gauss(image_path, annotation_ls, h_new, w_new, image, filp, flag):  # , image_path):
    # p = gaussian_kernel_2d_opencv(6)
    # z = np.zeros([2, 2], int)
    if flag == 0:
        image = Image.open(image_path)
        image = randomColor(image)
        image = np.asarray(image)
        image_size = np.shape(image)
        image = Image.fromarray(image)
        image = np.asarray(image.resize((w_new, h_new), Image.ANTIALIAS))
        image.setflags(write=1)
        col = image_size[0]
        raw = image_size[1]
        label_mat = np.zeros([int(h_new), int(w_new)], 'float32')
        ##print(image_size)
        for id in annotation_ls.keys():
            if annotation_ls[id]['name'].split('/')[-1] == str(image_path.split('/')[-1]):
                anno = annotation_ls[id]
                break
        if anno['annotation'] != []:
            if len(anno['annotation'][0]) == 4:
                for i in range(len(anno['annotation'])):
                    x, y, w, h = anno['annotation'][i]['x'], anno['annotation'][i]['y'], anno['annotation'][i]['w'], \
                                 anno['annotation'][i]['h']
                    x, y = bbox_convert_2_point(x, y, w, h)
                    label_mat[int(y * 1.0 / col * h_new)][int(x * 1.0 / raw * w_new)] = 1
                    # conv2_filter = tf.get_variable("weight", p)
                    # if cv2.waitKey(1000) == 1000:
                    #    break
            if len(anno['annotation'][0]) == 2:
                for i in range(len(anno['annotation'])):
                    x, y = anno['annotation'][i]['x'], anno['annotation'][i]['y']
                    label_mat[int(y * 1.0 / col * h_new)][int(x * 1.0 / raw * w_new)] = 1
            ##print("len(anno['annotation'])",len(anno['annotation']))
            label_mat = cv2.GaussianBlur(label_mat, (3, 3), 0)
            ##print("total222", int(np.sum(label_mat)))
            # label_mat_show = label_mat_show * 255
    else:
        image_size = np.shape(image)
        col = image_size[0]
        raw = image_size[1]
        label_mat = np.zeros([int(h_new), int(w_new)], 'float32')
        ##print(image_size)
        if filp != []:
            for i in range(len(filp)):
                x, y = filp[i][0], filp[i][1]
                label_mat[int(y * 1.0 / col * h_new)][int(x * 1.0 / raw * w_new)] = 1
            ###print("len(anno['annotation'])", len(anno['annotation']))
            label_mat = cv2.GaussianBlur(label_mat, (3, 3), 0)
            ##print("total222", int(np.sum(label_mat)))
    return label_mat, image


def convert_record_data_list(id_list, annotation_ls, arg_func):
    h_new = 540
    w_new = 960
    den_map_list = []
    total_num = []
    group_num = []
    src_image = []
    for i in id_list:
        image_path = root_path + '/image/' + annotation_ls[i]['name']
        image = 0
        flip = 0
        flag = 0
        if arg_func is not None:
            image, filp = arg_func(annotation_ls, image_path, h_new, w_new)
            if filp == []:
                continue
            if len(filp) == 1:
                den_map, img = image_process_gauss(image_path, annotation_ls, h_new, w_new, image, filp, 1)
            else:
                # image_process_gauss(annotation_ls)
                den_map, img = density_map(image_path, annotation_ls, 1.0, h_new, w_new, image, filp, 1)
            src_image.append(np.transpose(img, (2, 0, 1)))
            den_map_list.append(den_map)
            total_num.append(len(filp))
            gnum = np.clip(total_num[-1], a_min=0, a_max=99)
            gnum = gnum * 0.1
            gnum = np.floor(gnum).astype('int64')
            group_num.append(gnum)

        else:
            if annotation_ls[i]['num'] == 1:
                den_map, img = image_process_gauss(image_path, annotation_ls, h_new, w_new, image, flip, flag)
            else:
                den_map, img = density_map(image_path, annotation_ls, 1.0, h_new, w_new, image, flip, flag)
            src_image.append(np.transpose(img, (2, 0, 1)))
            den_map_list.append(den_map)
            total_num.append(annotation_ls[i]['num'])
            gnum = np.clip(total_num[-1], a_min=0, a_max=99)
            gnum = gnum * 0.1
            gnum = np.floor(gnum).astype('int64')
            group_num.append(gnum)
        print "train data append over" + str(i)
    return src_image, den_map_list, total_num, group_num


def randomColor(image):
    random_factor = random.randint(0, 31) / 10.
    color_image = ImageEnhance.Color(image).enhance(random_factor)
    random_factor = random.randint(10, 21) / 10.
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)
    random_factor = random.randint(10, 21) / 10.
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)
    random_factor = random.randint(0, 31) / 10.
    return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)



if __name__ == '__main__':
    current = time.time()
    root_path = '/home/aistudio/stage1/baidu_star_2018/'
    file_ls = ['/home/aistudio/stage1/baidu_star_2018/annotation/annotation_train_stage1.json',
           '/home/aistudio/stage2/baidu_star_2018/annotation/annotation_train_stage2.json']
    annotation_ls = json_read(file_ls)
    id_list = annotation_ls.keys()
    np.random.shuffle(id_list)
    train_list = id_list[:int(len(id_list) / 5) * 4]
    test_list = id_list[int(len(id_list) / 5) * 4:]
    print(len(train_list), len(test_list))
    img = fluid.layers.data(name="image", shape=[-1, 3, 540, 960])
    label = fluid.layers.data(name="label", shape=[-1, 1, 540, 960])
    num = fluid.layers.data(name="num", shape=[1], dtype='int64')
    gnum = fluid.layers.data(name="gnum", shape=[1], dtype='int64')
    feeder = fluid.DataFeeder(feed_list=[img, label, num, gnum], place=fluid.CPUPlace())
    for j in range(26):
        id_ls = []
        src_images, denmaps_list, total_nums, group_num = [], [], [], []
        is_test = False
        if j < 21:
            if j == 20:
                id_ls = train_list[j * 250:]
            else:
                id_ls = train_list[j * 250: (j + 1) * 250]
        else:
            is_test = True
            if j == 25:
                id_ls = test_list[(j-21)*250:]
            else:
                id_ls = test_list[(j-21)*250:(j-20)*250]
        if is_test:
            src_images, denmaps_list, total_nums, group_num = convert_record_data_list(id_ls, annotation_ls, None)


            def reader_creator1():
                def __impl__():
                    for ii in range(len(src_images)):
                        yield [
                            src_images[ii],
                            denmaps_list[ii],
                            total_nums[ii],
                            group_num[ii]
                        ]

                return __impl__


            reader1 = paddle.batch(reader_creator1(), batch_size=1)
            print "save test"
            fluid.recordio_writer.convert_reader_to_recordio_file(
                "/media/wyb/document1/pp/record_data/test_"+str(j-21)+".recordio", feeder=feeder, reader_creator=reader1)
        else:
            for jj in range(3):
                if jj == 0:
                    arg_func = None
                if jj == 1:
                    arg_func = dataugment_cut
                if jj == 2:
                    arg_func = dataugment_filp
                src_images, denmaps_list, total_nums, group_num = convert_record_data_list(id_ls, annotation_ls, arg_func)


                def reader_creator():
                    def __impl__():
                        for ii in range(len(src_images)):
                            yield [
                                src_images[ii],
                                denmaps_list[ii],
                                total_nums[ii],
                                group_num[ii]
                            ]

                    return __impl__


                reader = paddle.batch(reader_creator(), batch_size=1)
                print "save train" + str(j) + '/'+str(jj)
                fluid.recordio_writer.convert_reader_to_recordio_file(
                    "train_" + str(j) + str(jj) + ".recordio", feeder=feeder, reader_creator=reader)
