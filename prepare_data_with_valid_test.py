import tensorlayer as tl
import numpy as np
import os, csv, random, gc, pickle
import nibabel as nib
from scipy.ndimage.interpolation import zoom
from scipy.misc import imresize 

###============================= SETTINGS ===================================###
DATA_SIZE = 'half' # (small, half or all)

save_dir = "data/train_dev_all/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

HGG_data_path = "data/Training"
survival_csv_path = "data/survival_data.csv"
###==========================================================================###

survival_id_list = []
survival_age_list =[]
survival_peroid_list = []

with open(survival_csv_path, 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for idx, content in enumerate(reader):
        survival_id_list.append(content[0])
        survival_age_list.append(float(content[1]))
        survival_peroid_list.append(float(content[2]))

print(len(survival_id_list)) #163

if DATA_SIZE == 'all':
    HGG_path_list = tl.files.load_folder_list(path=HGG_data_path)
elif DATA_SIZE == 'half':
    HGG_path_list = tl.files.load_folder_list(path=HGG_data_path)[0:13]# DEBUG WITH SMALL DATA
elif DATA_SIZE == 'small':
    HGG_path_list = tl.files.load_folder_list(path=HGG_data_path)[0:6] # DEBUG WITH SMALL DATA
   
else:
    exit("Unknow DATA_SIZE")
print(len(HGG_path_list)) #210 #75

HGG_name_list = [os.path.basename(p) for p in HGG_path_list]


survival_id_from_HGG = []
for i in survival_id_list:
        survival_id_from_HGG.append(i)
 

print(len(survival_id_from_HGG)) #163, 0


# use 42 from 210 (in 163 subset) and 15 from 75 as 0.8/0.2 train/dev split

# use 126/42/42 from 210 (in 163 subset) and 45/15/15 from 75 as 0.6/0.2/0.2 train/dev/test split
index_HGG = list(range(0, len(survival_id_from_HGG)))

# random.shuffle(index_HGG)
# random.shuffle(index_HGG)

if DATA_SIZE == 'all':
    dev_index_HGG = index_HGG[-7:-4]
    test_index_HGG = index_HGG[-4:]
    tr_index_HGG = index_HGG[:-7]
elif DATA_SIZE == 'half':
    dev_index_HGG = index_HGG[21:28]  # DEBUG WITH SMALL DATA
    test_index_HGG = index_HGG[25:28]
    tr_index_HGG = index_HGG[:20]
elif DATA_SIZE == 'small':
    dev_index_HGG = index_HGG[4:6]   # DEBUG WITH SMALL DATA
    # print(index_HGG, dev_index_HGG)
    # exit()
    test_index_HGG = index_HGG[5:6]
    tr_index_HGG = index_HGG[0:2]


survival_id_dev_HGG = [survival_id_from_HGG[i] for i in dev_index_HGG]
survival_id_test_HGG = [survival_id_from_HGG[i] for i in test_index_HGG]
survival_id_tr_HGG = [survival_id_from_HGG[i] for i in tr_index_HGG]


survival_age_dev = [survival_age_list[survival_id_list.index(i)] for i in survival_id_dev_HGG]
survival_age_test = [survival_age_list[survival_id_list.index(i)] for i in survival_id_test_HGG]
survival_age_tr = [survival_age_list[survival_id_list.index(i)] for i in survival_id_tr_HGG]

survival_period_dev = [survival_peroid_list[survival_id_list.index(i)] for i in survival_id_dev_HGG]
survival_period_test = [survival_peroid_list[survival_id_list.index(i)] for i in survival_id_test_HGG]
survival_period_tr = [survival_peroid_list[survival_id_list.index(i)] for i in survival_id_tr_HGG]


data_types = ['DWI','Flair', 'T1', 'T2']
data_types_mean_std_dict = {i: {'mean': 0.0, 'std': 1.0} for i in data_types}


# calculate mean and std for all data types

# preserving_ratio = 0.0
# preserving_ratio = 0.01 # 0.118 removed
# preserving_ratio = 0.05 # 0.213 removed
# preserving_ratio = 0.10 # 0.359 removed

#==================== LOAD ALL IMAGES' PATH AND COMPUTE MEAN/ STD
import matplotlib.pyplot as plt
for i in data_types:
    data_temp_list = []
    for j in HGG_name_list:
        img_path = os.path.join(HGG_data_path, j, j + '_' + i + '.nii')
	#print(img_path)
        img = nib.load(img_path).get_data()
        print(img.dtype)
        img1=np.zeros((240,240,153), np.float32)
        X=np.zeros((240,240), np.float32)
        for c in range(1, 153):
            X=img[:, :, c]
            X=imresize(X, (240, 240), mode='F')
            img1[:, :, c]=X;
    data_temp_list.append(img1)
          #print("a")

    data_temp_list = np.asarray(data_temp_list)
    print(data_temp_list.shape)
    m = np.mean(data_temp_list)
    s = np.std(data_temp_list)
    data_types_mean_std_dict[i]['mean'] = m
    data_types_mean_std_dict[i]['std'] = s
del data_temp_list
print(data_types_mean_std_dict)

with open(save_dir + 'mean_std_dict.pickle', 'wb') as f:
    pickle.dump(data_types_mean_std_dict, f, protocol=4)


##==================== GET NORMALIZE IMAGES

X_test_input = []
X_test_target = []


print("Preparing Testing Data")

for i in survival_id_test_HGG:
    all_3d_data = []
    for j in data_types:
        img_path = os.path.join(HGG_data_path, i, i + '_' + j + '.nii')
        img = nib.load(img_path).get_data()
        img1=np.zeros((240,240,153), np.float32)
        X=np.zeros((240,240), np.float32)
        for c in range(1, 153):
            X=img[:, :, c]
            X=imresize(X, (240, 240), mode='F')
            img1[:, :, c]=X;
        img1 = (img1 - data_types_mean_std_dict[j]['mean']) / data_types_mean_std_dict[j]['std']
        img1 = img1.astype(np.float32)
        all_3d_data.append(img1)

    seg_path = os.path.join(HGG_data_path, i, i + 'OT.nii')
    seg_img = nib.load(seg_path).get_data()
    seg_img1=np.zeros((240,240,153), np.float32)
    X=np.zeros((240,240), np.float32)
    for c in range(1, 153):
        X=seg_img[:, :, c]
        X=imresize(X, (240, 240), mode='F')
        seg_img1[:, :, c]=X;
    seg_img1 = np.transpose(seg_img1, (1, 0, 2))


    for j in range(all_3d_data[0].shape[2]):
        combined_array = np.stack((all_3d_data[0][:, :, j], all_3d_data[1][:, :, j], all_3d_data[2][:, :, j], all_3d_data[3][:, :, j]), axis=2)
        combined_array = np.transpose(combined_array, (1, 0, 2))#.tolist()
        combined_array.astype(np.float32)
        X_test_input.append(combined_array)

        seg_2d = seg_img1[:, :, j]

        seg_2d.astype(int)
        X_test_target.append(seg_2d)
    del all_3d_data
    gc.collect()
    print("finished {}".format(i))



X_test_input = np.asarray(X_test_input, dtype=np.float32)
X_test_target = np.asarray(X_test_target)#, dtype=np.float32)
print(X_test_input.shape)
print(X_test_target.shape)




