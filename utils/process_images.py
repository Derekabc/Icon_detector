import os
import numpy as np
from os import walk
import cv2
from PIL import Image, ImageChops


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


if __name__ == '__main__':
    # 1242 x 2208;  720 x 1280
    width = 720
    height = 1280
    # location of the input dataset
    input_dir = '/home/dong/Downloads/data'
    # location of the output dataset
    out_dir = '/home/dong/Downloads/data_160'
    os.makedirs(out_dir, exist_ok=True)

    # padding or not, True, False
    padding = True

    # function use to resize images
    # resize_image(input_dir, out_dir, width, height, padding)

    # function use to plot rectangle on images
    # plot_rectangle(input_dir, out_dir)

    images = '/home/dong/PycharmProjects/Intern/Icon_Detector/TensorFlow/workspace_v1/training_demo/Inference_output/'
    out_dir = '/home/dong/PycharmProjects/Intern/Icon_Detector/TensorFlow/workspace_v1/training_demo/Inference_output_trim/'
    os.makedirs(out_dir, exist_ok=True)
    trim(images, out_dir)
