import os, glob

source_folder_train = 'C:\\PycharmProjects\\Icon_Detector_v2\\workspace\\images\\train\\'
source_folder_test = 'C:\\PycharmProjects\\Icon_Detector_v2\\workspace\\images\\test\\'

dest_train = 'C:\\Users\\Windows\\Downloads\\tensorflow-yolov3-master\\data\\image_160\ImageSets\\main\\train.txt'
dest_test = 'C:\\Users\\Windows\\Downloads\\tensorflow-yolov3-master\\data\\image_160\ImageSets\\main\\test.txt'


# replace the spaces in file name with underscores
for filename in os.listdir(source_folder_train):
    os.rename(os.path.join(source_folder_train, filename),
              os.path.join(source_folder_train, filename.replace(' ', '_')))

for filename in os.listdir(source_folder_test):
    os.rename(os.path.join(source_folder_test, filename), os.path.join(source_folder_test, filename.replace(' ', '_')))

file_list = glob.glob(source_folder_train + '*.png')
print("number of training images is", len(file_list))
train_file = open(dest_train, 'a')
for file_obj in file_list:
    file_path = os.path.join(source_folder_train, file_obj)
    file_name, file_extend = os.path.splitext(file_obj)
    # file_name: file name，file_extend: file extension
    train_file.write(file_name + '\n')
train_file.close()

file_list1 = glob.glob(source_folder_test + '*.png')
print("number of training images is", len(file_list1))
test_file = open(dest_test, 'a')
for file_obj in file_list1:
    file_path = os.path.join(source_folder_test, file_obj)
    file_name, file_extend = os.path.splitext(file_obj)
    test_file.write(file_name + '\n')
test_file.close()
