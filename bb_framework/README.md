## USAGE of the tracker for a dataset containing multiple sequences
this framework runs on a set of sequences, with three cameras 159,160 and 161. with slight changes you could make it run on
less or more cameras as well. Here, camera 159 is the coordinates origin for both camera pairs 159161 and 159161. make sure 
to calibrate those camera pairs in the same order (or adjust code). The dataset should have following folder structure:


# dataset structure


<main_dir> / <day_dir> / <VX_WWWWWWWWWWWWWWW159_1> / <VX_WWWWWWWWWWWWWWW159_00000001.jpg>
<main_dir> / <day_dir> / <VX_WWWWWWWWWWWWWWW160_1> / <VX_WWWWWWWWWWWWWWW160_00000001.jpg>
<main_dir> / <day_dir> / <VX_WWWWWWWWWWWWWWW161_1> / <VX_WWWWWWWWWWWWWWW161_00000001.jpg>

where <main_dir> is an arbritary name of the main directory

where <day_dir> is an arbritary name of the day directory 

where <VX_WWWWWWWWWWWWWWW159_1> , <VX_WWWWWWWWWWWWWWW160_1> and <VX_WWWWWWWWWWWWWWW159_1> is the name of the sequence 
	where X is the index of the sequence
	where WWWWWWWWWWWWWWW is an arbritary string e.g. CR3000x2 1838-ST
	where 159,160 and 161 are the three cameras, 159 is the center of the 3D coordinates

where <VX_WWWWWWWWWWWWWWW159_00000001.jpg> , <VX_WWWWWWWWWWWWWWW160_00000001.jpg> and <VX_WWWWWWWWWWWWWWW161_00000001.jpg>
	are the frames with ID 00000001...0000000Y ... 0000000N



# COMMANDS
* with visualization of the video
python tracking_plus_reconstruction.py     --main_dir= <main_dir>     --nn_budget=100     --max_cosine_distance=99999.999 --batch_size=1 --display

* without visualization of the video
python tracking_plus_reconstruction.py     --main_dir=<main_dir>    --nn_budget=100     --max_cosine_distance=99999.999 --batch_size=1 --no-display
# ADDITIONAL FLAGS
* --detection_model=retina or --detection_model=rcnn, rcnn is default slightly more accurate but more computational expensive than
retina
* --conf_thresh=0.95 for rcnn, 0.5 for retina recommended
* --batch_size=8 for rcnn, 64 for retinanet possible on RTX 3090, default is batch_size=1 (slow)

# EXAMPLE FOR THIS MACHINE

* with visualization of the video
python tracking_plus_reconstruction.py     --main_dir=/media/linx123-rtx/Elements/test3     --nn_budget=100     --max_cosine_distance=99999.999 --batch_size=1 --display

* without visualization of the video
python tracking_plus_reconstruction.py     --main_dir=/media/linx123-rtx/Elements/test3     --nn_budget=100     --max_cosine_distance=99999.999 --batch_size=1 --no-display
