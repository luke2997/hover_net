import glob, sys, os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import shutil

if __name__ == '__main__':
  
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument('--dataroot', default='/lustre/home/acct-clsyzs/clsyzs/Luke/hover_net-master/data/CoNSeP/Train/540x540_80x80', help='path to dataset')
  args = parser.parse_args() 
  
  ratios = []
  files = []
  cnt = 0
  Valid_path = args.dataroot.replace('/Train/', '/Valid/')
  Test_path = args.dataroot.replace('/Train/', '/Test/')
  for arr_name in glob.glob(f'{args.dataroot}/*.npy'):
    
    print(arr_name)
    try:
    
      data = np.load(arr_name)
      
      img = data[...,:3] # RGB images
      ann = data[...,3:] # instance ID map
      
      sum1 = (ann[:, :, 1] == 1).sum()
      sum2 = (ann[:, :, 1] == 2).sum()
      sum3 = (ann[:, :, 1] == 3).sum()
      sum4 = (ann[:, :, 1] == 4).sum()
      sum_all = sum1 + sum2 + sum3 + sum4
      
      ratios.append([sum1 / sum_all, sum2 / sum_all, sum3 / sum_all, sum4 / sum_all])
      files.append(arr_name)
      cnt += 1
      
      if cnt > 100:
        break
    except:
      import traceback
      print(f' !!!! Failed to read input: {arr_name}')
      traceback.print_exc(file=sys.stdout)
      
  #hist_info = plt.hist(ratios, bins=10, density=True)  
  
  ratios_Valid = []
  for arr_name in glob.glob(f'{Valid_path}/*.npy'):
    
    print(arr_name)
    try:
    
      data = np.load(arr_name)
      
      img = data[...,:3] # RGB images
      ann = data[...,3:] # instance ID map
      
      
      sum1 = (ann[:, :, 1] == 1).sum()
      sum2 = (ann[:, :, 1] == 2).sum()
      sum3 = (ann[:, :, 1] == 3).sum()
      sum4 = (ann[:, :, 1] == 4).sum()
      sum_all = sum1 + sum2 + sum3 + sum4
      
      ratio = [sum1 / sum_all, sum2 / sum_all, sum3 / sum_all, sum4 / sum_all]
      ratios.append(ratio)
      ratios_Valid.append(ratio)
      files.append(arr_name)
      cnt += 1
      
      print(ratio)
      
    except:
      import traceback
      print(f' !!!! Failed to read input: {arr_name}')
      traceback.print_exc(file=sys.stdout)
  
  bins = [0.01, 0.1, 0.5]
  bin_vals = np.digitize(ratios, bins=bins)
  fake_cls = bin_vals[:, 0] * 4**3 + bin_vals[:, 1] * 4**2 + bin_vals[:, 2] * 4 + bin_vals[:, 1]
  
  bin_valsv = np.digitize(ratios_Valid, bins=bins)
  fake_clsv = bin_valsv[:, 0] * 4**3 + bin_valsv[:, 1] * 4**2 + bin_valsv[:, 2] * 4 + bin_valsv[:, 1]  
  
  
  plt.hist(fake_cls, bins=40, density=True) 
  plt.hist(fake_clsv, bins=40, density=True)   
  
  file_ids = np.arange(len(files))
  folds = 5
  fake_cls = np.asarray(fake_cls)
  skf = StratifiedKFold(n_splits=folds, random_state=2)
  skf.get_n_splits(file_ids, fake_cls)
  
  for Train_index, Test_index in skf.split(file_ids, fake_cls):
    break
  
  new_Train_path = args.dataroot.replace('/Train/', '/Train_strat/')
  if not os.path.exists(new_Train_path):
    os.makedirs(new_Train_path)
  for idx in Train_index:
    dst = files[idx].replace('/Train/', '/Train_strat/')
    dst = dst.replace('/Valid/', '/Train_strat/')
    shutil.copy(files[idx], dst, follow_symlinks=True)
  
  new_Valid_path = args.dataroot.replace('/Train/', '/Valid_strat/')
  if not os.path.exists(new_Valid_path):
    os.makedirs(new_Valid_path)  
  new_Valid_ratios = []
  for idx in Test_index:
    dst = files[idx].replace('/Train/', '/Valid_strat/')
    dst = dst.replace('/Valid/', '/Valid_strat/')
    shutil.copy(files[idx], dst, follow_symlinks=True)
    new_Valid_ratios.append(ratios[idx])
  
  bin_valsv2 = np.digitize(new_Valid_ratios, bins=bins)
  fake_clsv2 = bin_valsv2[:, 0] * 4**3 + bin_valsv2[:, 1] * 4**2 + bin_valsv2[:, 2] * 4 + bin_valsv2[:, 1]    
  
  hist_info3 = plt.hist(fake_clsv2, bins=40, density=True, rwidth=0.25)  
  plt.show()  
