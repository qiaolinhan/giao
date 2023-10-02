import os
import sys

########################
# # For dataset on Ejectable device
# # Mount the device
# sudo mount -t drvfs E: /mnt/g
# sudo umount /mnt/g
# # rename images
# os.chdir('/mnt/g/datasets/geolocating/20220601/renamed/17.1/imgs/')
# print(os.getcwd())
########################
# For dataet copied to pc
# load the datas from a specific folder
<<<<<<< HEAD
os.chdir('/home/qiao/dev/giao/datasets/data_google2022/mask')
=======
os.chdir('/home/qiao/dev/giao/data/videos/20230926/frames/')
>>>>>>> ca9ee36c2ae812077350eb11b68415607f23415d
file_list = os.listdir()
print("[INFO] Successfully read the folder:", os.getcwd())
print("[INFO] Sample name of the file in this folder:", file_list[0])

########################
# reorder: sort the files by name
def last_11chars(x):
<<<<<<< HEAD
    return(x[-9:])
=======
    return(x[-11:])
>>>>>>> ca9ee36c2ae812077350eb11b68415607f23415d
# Sort the files 
filelist_sorted = sorted(file_list, key = last_11chars)
# print("[INFO] The dataset is sorted\n", filelist_sorted)  

for count, f in enumerate(filelist_sorted):
    f_name, f_ext = os.path.splitext(f)
    print(f'[INFO] Original name: {f_name} + {f_ext}')
    
    # new name 
    # use '{:03}'.format(num) to zero lead the num
    f_name_n = "%04d"%count
    new_name = f'{f_name_n}{f_ext}'
    print(f'[INFO] New name {f_name_n} + {f_ext}')

    os.rename(f, new_name)
    print(f'[INFO] renamed, please check')

# # rename labels
# os.chdir('/home/qiaolinhan/dev/giao/datasets/M300S/extended/41.2/labels')
# print(os.getcwd())

# file_list = os.listdir()

# # reorder: sort the files by name
# def last_11chars(x):
#     return(x[-8:])
# filelist_sorted = sorted(file_list, key = last_11chars)
# print(filelist_sorted)  

# for count, f in enumerate(filelist_sorted):
#     f_name, f_ext = os.path.splitext(f)
#     print(f'{f_name} + {f_ext}')

#     f_name = 'label_' + 'M300_' + '20220601_' + '17.1_' + str(count)
#     print(f'======> new name {f_name} ')

#     new_name = f'{f_name}{f_ext}'
#     os.rename(f, new_name)
#     print(f'======> renamed, please check')
