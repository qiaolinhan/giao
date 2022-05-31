import os
import sys

# rename images
os.chdir('/home/qiaolinhan/dev/datasets/M300S/m300dataset20220526/imgs')
print(os.getcwd())

for count, f in enumerate(os.listdir()):
    f_name, f_ext = os.path.splitext(f)
    print(f'{f_name} + {f_ext}')

    f_name = 'M300_' + str(count) + '_20220526'
    print(f'======> new name {f_name} ')
    new_name = f'{f_name}{f_ext}'
    os.rename(f, new_name)
    print(f'======> renamed, please check')

# rename labels
os.chdir('/home/qiaolinhan/dev/datasets/M300S/m300dataset20220526/labels')
print(os.getcwd())

for count, f in enumerate(os.listdir()):
    f_name, f_ext = os.path.splitext(f)
    print(f'{f_name} + {f_ext}')

    f_name = 'label_' + 'M300_' + str(count) + '_20220526'
    print(f'======> new name {f_name} ')
    new_name = f'{f_name}{f_ext}'
    os.rename(f, new_name)
    print(f'======> renamed, please check')