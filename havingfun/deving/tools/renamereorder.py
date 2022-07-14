import os
import sys

<<<<<<< HEAD
# mount the device
# sudo mount -t drvfs E: /mnt/g
# sudo umount /mnt/g
# rename images
os.chdir('/mnt/g/datasets/geolocating/20220601/renamed/17.1/imgs/')
print(os.getcwd())
=======
# # rename images
# os.chdir('/home/qiaolinhan/dev/giao/datasets/M300S/extended/imgs')
# print(os.getcwd())
# file_list = os.listdir()
>>>>>>> 46c2817fa7f5b56f3bae574dc3f4093af1cc8687

# # reorder: sort the files by name
# def last_11chars(x):
#     return(x[-4:])

<<<<<<< HEAD
    f_name = 'M300_' + '20220601_' + '17.1_' + str(count)
    print(f'======> new name {f_name} ')
    new_name = f'{f_name}{f_ext}'
    os.rename(f, new_name)
    print(f'======> renamed, please check')
=======
# print(sorted(file_list, key = last_11chars))  

# for count, f in enumerate(sorted(file_list, key = last_11chars)):
#     f_name, f_ext = os.path.splitext(f)
#     print(f'{f_name} + {f_ext}')

#     f_name = 'M300_' + '20220613_' + str(count)
#     print(f'======> new name {f_name} ')
#     new_name = f'{f_name}{f_ext}'
#     os.rename(f, new_name)
#     print(f'======> renamed, please check')
>>>>>>> 46c2817fa7f5b56f3bae574dc3f4093af1cc8687

# rename labels
os.chdir('/home/qiaolinhan/dev/giao/datasets/M300S/extended/41.2/labels')
print(os.getcwd())

file_list = os.listdir()

# reorder: sort the files by name
def last_11chars(x):
    return(x[-8:])
filelist_sorted = sorted(file_list, key = last_11chars)
print(filelist_sorted)  

for count, f in enumerate(filelist_sorted):
    f_name, f_ext = os.path.splitext(f)
    print(f'{f_name} + {f_ext}')

<<<<<<< HEAD
    f_name = 'label_' + 'M300_' + '20220601_' + '17.1_' + str(count)
=======
    f_name = 'label_' + 'M300_' + '20220613_' + '41.2_' + str(count)
>>>>>>> 46c2817fa7f5b56f3bae574dc3f4093af1cc8687
    print(f'======> new name {f_name} ')

    new_name = f'{f_name}{f_ext}'
    os.rename(f, new_name)
    print(f'======> renamed, please check')