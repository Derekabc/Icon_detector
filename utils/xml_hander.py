import xml.etree.ElementTree as ET
import os, glob


def edit_xml(xml_name, folder_name, file_name, file_path, new_file_name):
    """we want to edit
       folder
       filename
       path
    """
    xml_file = ET.parse(xml_name)
    root = xml_file.getroot()
    folder_tag = root.findall('folder')
    # print(folder_tag[0].text)
    folder_tag[0].text = folder_name

    file_name_tag = root.findall('filename')
    file_name_tag[0].text = file_name

    file_path_tag = root.findall('path')
    file_path_tag[0].text = file_path

    xml_file.write(new_file_name)


if __name__ == '__main__':
    source_folder_train = 'C:\\PycharmProjects\\data_1000_proccessed\\'
    new_folder = 'C:\\PycharmProjects\\data_1000_proccessed\\'
    folder_name = new_folder.split('\\')[-2]  # 'data_160'

    file_list = glob.glob(source_folder_train + '*.xml')
    print("number of training images is", len(file_list))
    for file_obj in file_list:
        xml_name = os.path.join(source_folder_train, file_obj)
        # xml_name = r'C:\PycharmProjects\Icon_Detector_v1\data\Google_Pixel_2_Jun_26_2020_16_39_47.xml'

        file_name = xml_name.split('\\')[-1][:-4] + '.png'  # 'Google_Pixel_2_Jun_26_2020_16_39_21.png'
        file_path = new_folder + file_name
        new_file_name = new_folder + xml_name.split('\\')[-1]  # 'C:\\PycharmProjects\\data_160\\Google_Pixel_2_Jun_26_2020_16_39_21.xml'

        edit_xml(xml_name, folder_name, file_name, file_path, new_file_name)
