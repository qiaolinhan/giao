import os
import sys

# # rename images
# os.chdir('/home/qiaolinhan/dev/giao/datasets/M300S/extended/imgs')
# print(os.getcwd())
# file_list = os.listdir()

# # reorder: sort the files by name
# def last_11chars(x):
#     return(x[-4:])

# print(sorted(file_list, key = last_11chars))  

# for count, f in enumerate(sorted(file_list, key = last_11chars)):
#     f_name, f_ext = os.path.splitext(f)
#     print(f'{f_name} + {f_ext}')

#     f_name = 'M300_' + '20220613_' + str(count)
#     print(f'======> new name {f_name} ')
#     new_name = f'{f_name}{f_ext}'
#     os.rename(f, new_name)
#     print(f'======> renamed, please check')

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

    f_name = 'label_' + 'M300_' + '20220613_' + '41.2_' + str(count)
    print(f'======> new name {f_name} ')

    new_name = f'{f_name}{f_ext}'
    os.rename(f, new_name)
    print(f'======> renamed, please check')