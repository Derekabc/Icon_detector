import os
import numpy as np
from os import walk
import cv2
from PIL import Image, ImageChops
import xml.etree.ElementTree as ET


def resize_image(input_dir, out_dir, width, height, padding=False):
    """
    This function is used to resize images while maintains the aspect ratio,
    :param input_dir: input directory
    :param out_dir: output directory
    :param width:  width to resize
    :param height: height to resize
    :param padding: padding or not
    :return: none
    """
    # get all the pictures in directory
    images = []
    ext = (".jpeg", ".jpg", ".png", "PNG")

    for (dirpath, dirnames, filenames) in walk(input_dir):
        for filename in filenames:
            if filename.endswith(ext):
                images.append(os.path.join(dirpath, filename))

    print("Working...")
    for image in images:
        # keep the transpancy layer or not
        # img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        img = cv2.imread(image)

        h, w = img.shape[:2]
        ratio = w / h

        if h > height or w > width:
            # shrinking image algorithm
            interp = cv2.INTER_AREA
        else:
            # stretching image algorithm
            interp = cv2.INTER_CUBIC

        w = width
        h = round(w / ratio)
        if h > height:
            h = height
        w = round(h * ratio)

        scaled_img = cv2.resize(img, (w, h), interpolation=interp)
        if padding:
            pad_bottom = abs(height - h)
            pad_right = abs(width - w)
            padded_img = cv2.copyMakeBorder(
                scaled_img, 0, pad_bottom, 0, pad_right, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
            cv2.imwrite(os.path.join(out_dir, os.path.basename(image)), padded_img)
        else:
            cv2.imwrite(os.path.join(out_dir, os.path.basename(image)), scaled_img)
    print("Completed!")


def plot_rectangle(input_dir, out_dir):
    """
    this function will plot a rectangle on the images.
    """
    # get all the pictures in directory
    images = []
    ext = (".jpeg", ".jpg", ".png")

    for (dirpath, dirnames, filenames) in walk(input_dir):
        for filename in filenames:
            if filename.endswith(ext):
                images.append(os.path.join(dirpath, filename))

    print("Working...")
    for image in images:
        img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        # represents the top left corner of rectangle
        start_point = (50, 50)
        # represents the bottom right corner of rectangle
        end_point = (220, 220)
        # Blue color in BGR
        color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 2
        # Draw a rectangle with blue line borders of thickness of 2 px
        image_new = cv2.rectangle(img, start_point, end_point, color, thickness)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(image)), image_new)
    print("Completed!")


def trim(input_dir, out_dir):
    """
    this function is used to remove the black padding at the right of images.
    """
    # get all the pictures in directory
    images = []
    ext = (".jpeg", ".jpg", ".png")

    for (dirpath, dirnames, filenames) in walk(input_dir):
        for filename in filenames:
            if filename.endswith(ext):
                images.append(os.path.join(dirpath, filename))

    print("Working...")
    for image in images:
        im = Image.open(image)
        bg = Image.new(im.mode, im.size, im.getpixel((im.size[0] - 5, im.size[1] - 5)))
        diff = ImageChops.difference(im, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        if bbox:
            scaled_img = im.crop(bbox)
        # scaled_img.show()
        # save a image using extension
        new_file_name = im.filename.split('/')[-1]
        scaled_img.save(out_dir + new_file_name)


def resize_image_xml(input_dir, out_dir, width, height, padding=False):
    """
    This function is used to resize images while maintains the aspect ratio, and modify the size information in the label file (*.xml)
    :param input_dir: input directory
    :param out_dir: output directory
    :param width:  width to resize
    :param height: height to resize
    :param padding: padding or not
    :return: none
    """
    # get all the pictures in directory
    images = []
    ext = (".jpeg", ".jpg", ".png", "PNG")

    for (dirpath, dirnames, filenames) in walk(input_dir):
        for filename in filenames:
            if filename.endswith(ext):
                images.append(os.path.join(dirpath, filename))

    print("Working...")
    for image in images:
        # keep the transpancy layer or not
        # img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        img = cv2.imread(image)

        h, w = img.shape[:2]
        ratio = w / h
        ratio_image = height / h
        if h > height or w > width:
            # shrinking image algorithm
            interp = cv2.INTER_AREA
        else:
            # stretching image algorithm
            interp = cv2.INTER_CUBIC

        w = width
        h = round(w / ratio)
        if h > height:
            h = height
        w = round(h * ratio)

        # here for ratio < hight/width,  changes of Height and Width of original image
        # is height / h, e.g 1280 / 2000.
        scaled_img = cv2.resize(img, (w, h), interpolation=interp)

        if padding:
            pad_bottom = abs(height - h)
            pad_right = abs(width - w)
            padded_img = cv2.copyMakeBorder(
                scaled_img, 0, pad_bottom, 0, pad_right, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
            cv2.imwrite(os.path.join(out_dir, os.path.basename(image)), padded_img)
        else:
            cv2.imwrite(os.path.join(out_dir, os.path.basename(image)), scaled_img)

        ######################################
        # Now we need to modify the corresponding width and height information of xml file.
        #####################################
        # step 1: find corresponding xml file.
        xml_file = image.split('\\')[-1][:-4] + '.xml'
        xml_file_path = input_dir + xml_file
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        new_file_name = out_dir + xml_file

        for child in root:
            if child.tag == 'object':
                child[4][0].text = str(round(int(child[4][0].text) * ratio_image))
                child[4][1].text = str(round(int(child[4][1].text) * ratio_image))
                child[4][2].text = str(round(int(child[4][2].text) * ratio_image))
                child[4][3].text = str(round(int(child[4][3].text) * ratio_image))

        tree.write(new_file_name)
    print("Completed!")


if __name__ == '__main__':
    # 1242 x 2208;  720 x 1280
    width = 720
    height = 1280

    #######################################
    # resize_image()
    #######################################
    # # location of the input dataset
    # input_dir = 'C:\\PycharmProjects\\data_1000 - Copy'
    # # location of the output dataset
    # out_dir = 'C:\\PycharmProjects\\data_1000_proccessed'
    # os.makedirs(out_dir, exist_ok=True)
    # padding or not, True, False
    # padding = True
    # function use to resize images
    # resize_image(input_dir, out_dir, width, height, padding)

    #######################################
    # plot_rectangle()
    #######################################
    # location of the input dataset
    # input_d#= 'C:\\PycharmProjects\\data_1000_proccessed'
    # function use to plot rectangle on images
    # plot_rectangle(input_dir, out_dir)

    #######################################
    # trim()
    #######################################
    # images = '/home/dong/PycharmProjects/Intern/Icon_Detector/TensorFlow/workspace_v1/training_demo/Inference_output/'
    # out_dir = '/home/dong/PycharmProjects/Intern/Icon_Detector/TensorFlow/workspace_v1/training_demo/Inference_output_trim/'
    # os.makedirs(out_dir, exist_ok=True)
    # trim(images, out_dir)

    #######################################
    # resize_image_xml()
    #######################################
    # location of the input dataset
    input_dir = 'C:\\PycharmProjects\\data_1000 - Copy\\'
    # location of the output dataset
    out_dir = 'C:\\PycharmProjects\\data_1000_proccessed\\'
    os.makedirs(out_dir, exist_ok=True)
    resize_image_xml(input_dir, out_dir, width, height, padding=True)

