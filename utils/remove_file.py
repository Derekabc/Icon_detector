import os

dir_name = "C:\PycharmProjects\Icon_Detector_v1\data_160_yolo1\images/"
test = os.listdir(dir_name)

for item in test:
    if item.endswith(".xml"):
        os.remove(os.path.join(dir_name, item))