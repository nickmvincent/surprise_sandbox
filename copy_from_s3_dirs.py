import os
import shutil


def recursive_search(directory):
    for root, dirs, files in os.walk(directory):
        print('root', root)
        print('dirs', dirs)
        print('files', files)
        for file in files:
            if file.endswith('.csv'):
                print(file)
                shutil.copyfile(root + '/' + file, 'results/' + file)
        for d in dirs:
            recursive_search(root + '/' + d)


if __name__ == '__main__':
    # put your directory here...
    recursive_search('C:/Users/Nick/Desktop/mys3/databoycott/round4')