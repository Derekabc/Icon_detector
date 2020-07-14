import glob
import shutil

source = 'C:\\PycharmProjects\\data_160\\'
mydict = {
    'C:\\PycharmProjects\\data_1000': ['png']
}

for destination, extensions in mydict.items():
    for ext in extensions:
        for file in glob.glob(source + '*.' + ext):
            shutil.copy(file, destination)