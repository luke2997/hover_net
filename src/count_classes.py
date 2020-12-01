
import glob, sys
import argparse

import numpy as np
import cv2

import scipy.io as sio

if __name__ == '__main__':
  
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument('--dataroot', default='/lustre/home/acct-clsyzs/clsyzs/Luke/hover_net-master/data/CoNSeP/Train/Labels', help='path to dataset')
  args = parser.parse_args() 
  files = []
  files_counts = []
  cnt = 0
  for mat_name in glob.glob(f'{args.dataroot}/*.mat'):
    
    print(mat_name)
    try:
      
      ann = sio.loadmat(mat_name)
      ann_inst = ann['inst_map']
      ann_type = ann['type_map']
      
      # merge classes for CoNSeP (in paper we only utilise 3 nuclei classes and background)
      # If own dataset is used, then the below may need to be modified
      ann_type[(ann_type == 3) | (ann_type == 4)] = 3
      ann_type[(ann_type == 5) | (ann_type == 6) | (ann_type == 7)] = 4
      
      ann_type = ann_type.astype(np.int)
    
      class_counts = np.zeros((4), dtype=np.int)
      unique_ids =  np.unique(ann_inst)
      for id in unique_ids:
        if id == 0:
          continue
        mask = ann_inst == id
        class_no = ann_type[mask]
        cls_out = np.unique(class_no[:])
        assert(len(cls_out) == 1)
        class_counts[cls_out[0] - 1] += 1
      
      files.append(mat_name)
      files_counts.append(class_counts)
      cnt += 1
      
      #if cnt > 2:
      #  break
    except:
      import traceback
      print(f' !!!! Failed to read input: {mat_name}')
      traceback.print_exc(file=sys.stdout)
      
  
  files_counts = np.asarray(files_counts)
  print('Dataset statistics: ')
  print(f'Cls1, Cls2, Cls3, Cls4')
  print(f'{files_counts[:, 0].sum()}, {files_counts[:, 1].sum()}, {files_counts[:, 2].sum()}, {files_counts[:, 3].sum()}')
  
  
  
  
      
