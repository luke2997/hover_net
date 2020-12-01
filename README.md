# HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images

This code is based on the implementation of [Quoc Dang Vu](https://github.com/vqdang) and [Simon Graham](https://github.com/simongraham) available at [this link](https://github.com/vqdang/hover_net). This code uses new loss functions with various encoders implemented. <br />



## Dataset

Datasets can be found here [this link](https://warwick.ac.uk/fac/sci/dcs/research/tia/data/). <br />


The colour of the nuclear boundary denotes the type of nucleus. <br />
Blue: epithelial<br />
Red: inflammatory <br />
Green: spindle-shaped <br />
Cyan: miscellaneous

## Results

All comparative results on the CoNSeP, Kumar and CPM-17 datasets can be found [here](https://drive.google.com/drive/folders/1WTkleeaE6ne8qxuYzptv2bKwMdZVBpzr?usp=sharing). 

## Extra

The cell profiler pipeline that we used in our comparative experiments can be found [here](https://drive.google.com/file/d/1E5UII9fsYT2N2KBUNLS89OV9AstYDLlZ/view?usp=sharing).


# PanNuke Challenge

* Thanks to the code from Eugenia Martynova, Lili Mészáros, Denise Meerkerk, Linda Schmeitz, Enrico Schmitz and Luca Carotenuto.
* The evaluation is taken from [TIA-Lab/PanNuke-metrics](https://github.com/TIA-Lab/PanNuke-metrics).


## Steps to Reproduce Results
### Prerequisites
1. Install the requirements (it is probably easier to create a separate environment with conda):
```
pip install -r requirements.txt
```
2. Access PanNuke data and [pretrained weights](https://drive.google.com/file/d/187C9pGjlVmlqz-PlKW1K8AYfxDONrB0n/view).
3. Add the location of the weights in `src/opt/hover.py` and `src/opt/other.py`.

### Preprocessing
4. Transform the PanNuke data with `src/transform_pannuke.py` to the correct format of `(img_size,img_size,5)`, where the indices 0:3 are the RGB codes, 3 is the instance and 4 is the class type. The output directory (`out_dir`) has to be set manually.
5. An optional step is stain normalisation with `src/normalizeStaining.py`. This is added to `src/transform_pannuke.py`, but can be changed in the settings of the `transform` function. With the used stain normalisation implementation a few images in each fold are lost. Since their number is insignificant, we decided to ignore it, however, it should be handled properly. `src/remove_lost_mages.py` script removes the lost indices from the dataset (for `images.npy`, `types.npy` and `masks.npy`).

### Training
6. In `src/config.py` the model type and paths need to be set 
```
mode = 'hover'
self.model_type = 'np_hv'
```
* For the logging different paths need to be given, e.g. `self.log_path = '/hovernet_logs/'`.   


Our experimental setup follows the one from the original paper. The creators of the PanNuke dataset divided it into 3 splits (as described on the official dataset page https://warwick.ac.uk/fac/sci/dcs/research/tia/data/pannuke):   
1) Training: Fold 1; Validation: Fold 2; Testing: Fold 3  
2) Training: Fold 2; Validation: Fold 1; Testing: Fold 3  
3) Training: Fold 3; Validation: Fold 2; Testing: Fold 1  

* For split 1 the setup of the training and validation folders is the following:
```
self.train_dir = ['/fold1/']
self.valid_dir = ['/fold2/']
```
7. Training can be started with passing the GPU ids as arguments, e.g. with 0 and 1.
```
python train.py --gpu='0,1'
```
8. Overall, the config and training needs to be edited/started 6 times (3 folds times 2 models). The training can be done parallel, with creating duplicates of the config and the training files.

### Inference
9. For inference images in `.png` format should be used. To extract them we created `extract_png_pannuke.py` script, which can work with both original images and stain-normalized images.
10. Inference can be run by `python src/infer.py --gpu='gpu_id'`. Note that only one GPU is supported. As in the previous step, this needs to be done 6 times, using the right configuration. In `src/config.py` the path to the model checkpoint, the inference images, and the output directory needs to be given:
```
self.inf_model_path  = self.save_dir + '/model-19640.index'
self.inf_data_dir = '/inference/'
self.inf_output_dir = '/output/'
```
11. Final instance segmentation can be obtained by running `python src/process.py`.

### Evaluation
12. For the segmentation metrics `src/metrics/transform_for_metrics.py` needs to be run to convert the predictions back to the PanNuke format with 6 channels. The path to the resulting .mat files and the output .npy file is necessary. This will look something like the following:
```
path_to_folder = '/output/v3.0/micronet/_proc' 
output = 'masks.npy'
```
13. Run `src/metrics/remove_lostids_from_gt_fold3.py` and `src/metrics/remove_lostids_from_gt_fold1.py` to remove the lost files from the ground truth. 
14. Now, `python src/metrics/run.py --true_path=<n> --pred_path=<n> --save_path=<n>y` can be used to obtain the panoptic quality scores.
15. For the detection/classification metrics, run `python src/metrics/convert_to_centroids.py` to extract the centroids of the ground truth.
16. Now, `python src/compute_stats.py` can be run with ground truth being the extracted centroids and the predictions being post-processed predictions (.mat files).

### Ensemble
17. Ensemble learns weights it gives each model based on the subset of validation images balanced by tissue type. We used 10 images of each tissue type, which gave 190 for split 2 and 186 for splits 1 and 3, since for these splits one tissue type did not have 10 images. The extraction of the subset is implemented in `balanced_val_subset.py` script
18. To get the results for ensemble `ensemble_pipeline_split{x}.py` (x = {1,2,3}) is run. The input files are in `.mat` format for the HoVer-net and Micro-Net resulting images as well as the images obtained by `balanced_val_subset.py` as mentioned at point 17. The ground-truth is obtained from the `.npy` file of the masks in the original data. Make sure to use the right folders to extract the resulting images of Micro-net, HoVer-net and the ground truth. It outputs a `.csv` file of the Jaccard (sklearn) results per subset (due to memory issues not all data can be loaded at once) for softvote and hardvote. It also saves the resulting mask like images of the softvote.

### Visualisation
19. We compare prediction of HoverNet and Micronet model as well as ground truth by visualising it next to each other and the original images. For visualisation a random sample of the whole data set is selected, but also there is a possibility to select a sample by tissue type. Also training statistics is plotted for both models. The visualisation is contained in the notebook `src/vis/Visualisation.ipynb`. The notebook was run on Cartesius to access all the data and the output of cells is saved. Therefore it might be better to view it from Jupiter Lab/Jupiter Notebook, in GitHub it loads very slowly.
20. The overlayed annotation of ground truth was obtained with `annotate_ground_truth.py` script.

