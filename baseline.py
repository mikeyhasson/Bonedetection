import os
from skimage.color import rgb2hsv, rgb2gray
from skimage.filters.rank import entropy
from skimage.morphology import *
from skimage.util import img_as_ubyte
from skimage.filters import threshold_otsu
import scipy.ndimage as ndi
from skimage import measure
import skimage.color as color
import skimage.io as io
from skimage.util import img_as_ubyte,random_noise

import cv2
import matplotlib.pyplot as plt
from skimage import io

import cv2
import numpy as np
from skimage import filters
from skimage.segmentation import flood
import sys

sys.path.append(os.getcwd())
from bonemarrow_label import BoneMarrowLabel

CONNECTION_SIZE=20
FRAME_SIZE = 6
BLOCK_SIZE = 10
MIN_BONE_AREA = 2000
MIN_FULL_BONE_AREA = 8000


def remove_small_cells(segmented_image,debug):
    cells_binary = np.logical_or(segmented_image == BoneMarrowLabel.OSTEOBLAST,
                                 segmented_image == BoneMarrowLabel.OSTEOCLAST,
                                 segmented_image == BoneMarrowLabel.OSTEOCYTE)
    if debug:
        io.imshow(cells_binary)
        io.show()
    cells_opening = binary_opening(cells_binary, disk(1))
    if debug:
        io.imshow(cells_opening)
        io.show()
    no_small_cells = remove_small_objects(cells_opening, min_size=50)
    if debug:
        io.imshow(no_small_cells)
        io.show()

    segmented_image[cells_binary * ~no_small_cells] = BoneMarrowLabel.BACKGROUND

    return segmented_image


def segment_image(image_data, print_progress=False,debug=False):
    """
        Given an image array returns the segmented image
    """
    segmented_image = segment_by_color(image_data)
    if print_progress:
        print("segmented image by color")
    segmented_image = remove_small_cells(segmented_image,debug)
    if print_progress:
        print("removed small cells")
    segmented_image = seperate_bone_and_other(image_data, segmented_image,debug)
    if print_progress:
        print("segmented other and bone")
    segmented_image = classify_cells(segmented_image)
    if print_progress:
        print("classified cells")
    return segmented_image

def remove_small_holes_bone (ar):
    ar_progress = np.copy(ar)
    for x in range(0, ar.shape[0],BLOCK_SIZE):
        for y in range(0, ar.shape[1],BLOCK_SIZE):
            if ar_progress[x, y]:
                flooded_image = flood(ar, (x, y))

                contours, _ = cv2.findContours(flooded_image.astype(int),
                                               cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_NONE)

                outer_contour = max(contours, key=cv2.contourArea)

                tmp = np.zeros_like(ar, dtype=np.uint8)
                cv2.drawContours(tmp, [outer_contour], 0, color=5, thickness=cv2.FILLED)
                inside_contour_binary = tmp == 5

                ar[inside_contour_binary] = 1
                ar_progress[inside_contour_binary] = 0

    return ar


def classify_cells(segmented_image):
    """
        Input: An ndarray of shape (w,h,3) (for each pixel its RGB values)
        Return: An ndarray of shape (w,h), for each pixel its class number:
            background, fats and bone are segmented as BACKGROUND
            osteoblasts are segmented as OSTEOBLAST
            osteoclasts are segmented as OSTEOCLAST
            osteocytes are segmented as OSTEOCYTES
    """

    bones_binary = np.copy(segmented_image == BoneMarrowLabel.BONE)
    bones_binary = remove_small_holes_bone(bones_binary)

    background_binary = segmented_image ==  BoneMarrowLabel.BACKGROUND
    segmented_image[bones_binary * background_binary] = BoneMarrowLabel.BONE

    bones_dilated_10 = binary_dilation (bones_binary,disk(10))
    bones_dilated_20 = binary_dilation (bones_binary,disk(20))

    segmented_image [~bones_dilated_20] = BoneMarrowLabel.BACKGROUND

    osteoclast = segmented_image == BoneMarrowLabel.OSTEOCLAST
    segmented_image[~bones_dilated_10 * ~osteoclast] = BoneMarrowLabel.BACKGROUND

    frame_of_bone = bones_dilated_10 * ~bones_binary

    osteoblasts = frame_of_bone * (segmented_image == BoneMarrowLabel.OSTEOBLAST)
    segmented_image[osteoblasts] = BoneMarrowLabel.OSTEOBLAST



    return segmented_image


def segment_by_color(image_data):
    """
        Input: An ndarray of shape (w,h,3) (for each pixel its RGB values)
        Return: An ndarray of shape (w,h), for each pixel its class number:
            background, fats and bone are segmented as BACKGROUND
            suspected osteoclasts are segmented as OSTEOCLAST
            suspected osteoblasts and osteocytes are segmented as OSTEOBLAST
    """
    # https://mattmaulion.medium.com/color-image-segmentation-image-processing-4a04eca25c0
    # Convert RGB to HSV
    img_hsv = rgb2hsv(image_data)

    # blue-green tones are osteoclasts or osteocytes
    lower_mask = (img_hsv[:, :, 0] > 0.25) * (img_hsv[:, :, 1] > 0.1)
    upper_mask = (img_hsv[:, :, 0] < 0.7) * (img_hsv[:, :, 1] > 0.1)

    blue = BoneMarrowLabel.OSTEOBLAST * lower_mask * upper_mask

    # red are osteo-blasts
    red = BoneMarrowLabel.OSTEOCLAST * ((img_hsv[:, :, 0] >= 0.7) + (img_hsv[:, :, 0] <= 0.1))

    return red + blue



def remove_empty_tissues(possible_bones):
    binary = np.zeros_like(possible_bones)
    for x in range(0, possible_bones.shape[0], BLOCK_SIZE):
        for y in range(0, possible_bones.shape[1], BLOCK_SIZE):
            if possible_bones[x, y]:
                bone_flag = True
                flooded_image = flood(possible_bones, (x, y))
                flooded_area = np.sum(flooded_image)

                boundary_flag = np.any(flooded_image[0]) or np.any(flooded_image[-1]) or \
                                np.any(flooded_image[:, 0]) or np.any(flooded_image[:, -1])

                if flooded_area < 300:
                    bone_flag = False
                elif flooded_area < 500:
                    # checking if it's on the edge
                    if not boundary_flag:
                        bone_flag = False

                contours, _ = cv2.findContours(flooded_image.astype(int),
                                               cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_NONE)
                outer_contour = max(contours, key=cv2.contourArea)
                outer_area = cv2.contourArea(outer_contour)  # outer bone area
                tmp = np.zeros_like(possible_bones, dtype=np.uint8)
                cv2.drawContours(tmp, [outer_contour], 0, color=5, thickness=cv2.FILLED)

                #

                inside_contour_binary = tmp == 5
                if bone_flag:
                    inside_contour = flooded_image[inside_contour_binary]
                    inside_contour_area = np.sum(inside_contour)

                    outer_area_image_ratio = outer_area / (possible_bones.shape[0] * possible_bones.shape[1])
                    if inside_contour_area / outer_area > 0.5 or outer_area_image_ratio > 0.2 \
                            or (boundary_flag and outer_area_image_ratio > 0.01):
                        binary[flooded_image] = 1
                        possible_bones[flooded_image] = 0
                        continue

                possible_bones[inside_contour_binary] = 0
    return binary

def expand_bone_frame(image_data, segmented_image):
    segmented_image_c = np.copy(segmented_image)
    x, y = 0, 0
    # expend right
    while y != image_data.shape[1]:
        if segmented_image_c[x, y] == BoneMarrowLabel.BONE and segmented_image_c[x + 1, y] == BoneMarrowLabel.BACKGROUND:
            for x1 in [min(x + i, image_data.shape[0] - 1) for i in range(0, FRAME_SIZE)]:
                if segmented_image_c[x1, y] == BoneMarrowLabel.BACKGROUND:
                    segmented_image[x1, y] = BoneMarrowLabel.BONE
            x += FRAME_SIZE - 1
        x += 1
        if x >= image_data.shape[0] - 1:
            x = 0
            y += 1

    # expend left
    x, y = image_data.shape[0] - 1, 0
    while y != image_data.shape[1]:
        if segmented_image_c[x, y] == BoneMarrowLabel.BONE and segmented_image_c[x - 1, y] == BoneMarrowLabel.BACKGROUND:
            for x1 in [max(x - i, 0) for i in range(0, FRAME_SIZE)]:
                if segmented_image_c[x1, y] == BoneMarrowLabel.BACKGROUND:
                    segmented_image[x1, y] = BoneMarrowLabel.BONE
            x -= FRAME_SIZE - 1
        x -= 1
        if x <= 0:
            x = image_data.shape[0] - 1
            y += 1

    # expend up
    x, y = 0, 0
    while x != image_data.shape[0]:
        if segmented_image_c[x, y] == BoneMarrowLabel.BONE and segmented_image_c[x, y + 1] == BoneMarrowLabel.BACKGROUND:
            for y1 in [min(y + i, image_data.shape[0] - 1) for i in range(0, FRAME_SIZE)]:
                if segmented_image_c[x, y1] == BoneMarrowLabel.BACKGROUND:
                    segmented_image[x, y1] = BoneMarrowLabel.BONE
            y += FRAME_SIZE - 1
        y += 1
        if y >= image_data.shape[0] - 1:
            y = 0
            x += 1

    # expend down
    x, y = 0, image_data.shape[1] - 1
    while x != image_data.shape[0]:
        if segmented_image_c[x, y] == BoneMarrowLabel.BONE and segmented_image_c[x, y - 1] == BoneMarrowLabel.BACKGROUND:
            for y1 in [max(y - i, 0) for i in range(0, FRAME_SIZE)]:
                if segmented_image_c[x, y1] == BoneMarrowLabel.BACKGROUND:
                    segmented_image[x, y1] = BoneMarrowLabel.BONE
            y -= FRAME_SIZE - 1
        y -= 1
        if y <= 0:
            y = image_data.shape[1] - 1
            x += 1

    return segmented_image


def remove_small_objects_inside_frame(image_data,min_size_frame,min_size):
    all = remove_small_objects(image_data,min_size)

    out = np.zeros_like(image_data)
    range_x = range(0, image_data.shape[0], BLOCK_SIZE)
    range_y = range(0, image_data.shape[1], BLOCK_SIZE)
    x_y_zip = list(zip(range_x,np.zeros_like(range_x))) \
                + list(zip(range_x,np.full_like(range_x,image_data.shape[1]-1))) \
                + list(zip(np.zeros_like(range_y),range_y)) \
                + list(zip(np.full_like(range_y,image_data.shape[0]-1),range_y))

    for x,y in x_y_zip:
            if not image_data[x,y]:
                continue
            flooded_image = flood(image_data, (x, y))
            if np.sum(flooded_image) >= min_size_frame:
                out = np.logical_or(out,flooded_image)

    out = np.logical_or(out,all)
    return out

def target_bones(image_data,debug):
    image_data = color.rgb2gray(image_data)
    img = np.copy(image_data)
    entropy_img = entropy(img_as_ubyte(img), disk(6)) # inspecting the noise

    img_noisy = random_noise(img, mode='gaussian')
    quite_parts = entropy_img < 1.7
    img [quite_parts] = img_noisy[quite_parts]
    entropy_img = entropy(img_as_ubyte(img), disk(6))  # inspecting the noise
    thresh = threshold_otsu(entropy_img)

    binary = ((entropy_img < thresh) * ~quite_parts)  # bone is of lower noise, but higher than 1.7
    if debug:
        io.imshow(binary)
        io.show()

    binary = binary_opening(binary,footprint=disk(5))
    if debug:
        io.imshow(binary)
        io.show()

    binary = remove_empty_tissues(binary)
    if debug:
        io.imshow(binary)
        io.show()


    binary = binary_closing(binary,
                            footprint=disk(3))
    if debug:
        io.imshow(binary)
        io.show()


    binary = remove_small_objects_inside_frame(binary, min_size=5000, min_size_frame=2000)  # removing said areas
    if debug:
        io.imshow(binary)
        io.show()

    binary = binary_dilation(binary, footprint=disk(6))  # removing said areas
    if debug:
        io.imshow(binary)
        io.show()

    return binary

def seperate_bone_and_other(image_data, segmented_image,debug):
    bone_mask = target_bones(image_data,debug)

    #connecting bones
    for x in range(bone_mask.shape[0]):
        for y in range(bone_mask.shape[1]):
            if bone_mask[x, y] and bone_mask[min(x + CONNECTION_SIZE, bone_mask.shape[0] - 1), y]:
                for x1 in [min(x + i, bone_mask.shape[0] - 1) for i in range(0, CONNECTION_SIZE)]:
                        bone_mask[x1, y] = 1
            if bone_mask[x, y] and bone_mask[x, min(y + CONNECTION_SIZE, bone_mask.shape[1] - 1)]:
                for y1 in [min(y + i, bone_mask.shape[1] - 1) for i in range(0, CONNECTION_SIZE)]:
                        bone_mask[x, y1] = 1

    io.show()
    # changing to bones
    for x in range(image_data.shape[0]):
        for y in range(image_data.shape[1]):
            if bone_mask[x, y]:
                if segmented_image[x, y] == BoneMarrowLabel.BACKGROUND:
                    segmented_image[x, y] = BoneMarrowLabel.BONE
                else:
                    segmented_image[x, y] = BoneMarrowLabel.OSTEOCYTE

    return segmented_image

def count(seg,interest):
    segmented = np.copy(seg == interest)

    perimiter = 0
    count = 0
    surface = 0
    for x in range(0, segmented.shape[0]):
        for y in range(0, segmented.shape[1]):
            if segmented[x, y]:
                flooded_image = flood(segmented, (x, y))
                #io.imshow(flooded_image)
                #io.show()

                contours, _ = cv2.findContours(flooded_image.astype(int),
                                               cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_NONE)

                outer_contour = max(contours, key=cv2.contourArea)

                surface += cv2.contourArea(outer_contour)  # outer bone area
                perimiter += cv2.arcLength(outer_contour, True)
                count += 1

                segmented[flooded_image] = 0


    return perimiter,count,surface

if __name__ == "__main__":
     for x in [str(i) for i in range(3,4)]:
        img = io.imread(
            f"/Users/mikeyhasson/PycharmProjects/BoneDetection/content/BCCD_Dataset/BCCD/JPEGImages/{x}.bmp")
        img_annot = io.imread(
            f"/Users/mikeyhasson/PycharmProjects/BoneDetection/content/BCCD_Dataset/BCCD/annots/{x}.bmp")

        seg = segment_image(img,debug=False)

        bone_seg = seg == BoneMarrowLabel.BONE
        osteoclast = seg == BoneMarrowLabel.OSTEOCLAST
        osteoblast = seg == BoneMarrowLabel.OSTEOBLAST
        osteocyte = seg == BoneMarrowLabel.OSTEOCYTE
        tissue_seg = seg == BoneMarrowLabel.BACKGROUND

        #print(count(seg,BoneMarrowLabel.OSTEOCYTE))


        seg = np.stack((osteoclast, bone_seg, osteoblast), axis=2)

        seg[osteocyte] = (214,0,255)

        io.imshow(seg.astype(np.uint8) * 100)
        io.show()
        #io.imshow(seg.astype(np.uint8) * 100)
        #io.show()

        fig = plt.figure(figsize=(14, 7))

        # Adds subplot on position 1
        ax1 = fig.add_subplot(121)
        ax1.set_title(f'{x}.bmp')
        # Adds subplot on position 2
        ax2 = fig.add_subplot(122)

        ax1.imshow(img_annot)
        ax2.imshow(seg.astype(np.uint8) * 100)
        plt.savefig(f'figures/fig{x}.png')
        plt.close()

