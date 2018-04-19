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
                if 'uid_sets' in file:
                    if not os.path.isfile('standard_results/' + file):
                        shutil.copyfile(root + '/' + file, 'standard_results/' + file)
                else:
                    
                    shutil.copyfile(root + '/' + file, 'results/' + file)
                    
        for d in dirs:
            recursive_search(root + '/' + d)


if __name__ == '__main__':
    # put your directory here...
    