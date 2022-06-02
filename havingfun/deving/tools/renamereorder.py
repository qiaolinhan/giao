import os
import sys

# mount the device
# sudo mount -t drvfs E: /mnt/g
# sudo umount /mnt/g
# rename images
os.chdir('/mnt/g/datasets/geolocating/20220601/renamed/17.1/imgs/')
print(os.getcwd())

for count, f in enumerate(os.listdir()):
    f_name, f_ext = os.path.splitext(f)
    print(f'{f_name} + {f_ext}')

    f_name = 'M300_' + '20220601_' + '17.1_' + str(count)
    print(f'======> new name {f_name} ')
    new_name = f'{f_name}{f_ext}'
    os.rename(f, new_name)
    print(f'======> renamed, please check')

# rename labels
os.chdir('/mnt/g/datasets/geolocating/20220601/renamed/17.1/labels/')
print(os.getcwd())

for count, f in enumerate(os.listdir()):
    f_name, f_ext = os.path.splitext(f)
    print(f'{f_name} + {f_ext}')

    f_name = 'label_' + 'M300_' + '20220601_' + '17.1_' + str(count)
    print(f'======> new name {f_name} ')

    new_name = f'{f_name}{f_ext}'
    os.rename(f, new_name)
    print(f'======> renamed, please check')