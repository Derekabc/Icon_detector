import os


source_folder = r'C:\PycharmProjects\data_1000'

for filename in os.listdir(source_folder):
    os.rename(os.path.join(source_folder, filename),
              os.path.join(source_folder, filename.replace(' ', '_')))