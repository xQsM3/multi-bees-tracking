
## description

py-motmetrics evaluates a MOT by using ground truth bboxes and identities and the predictions of the tracker, outputting
MOT benchmarks

https://github.com/cheind/py-motmetrics

## install 

pip install motmetrics

## for help

python -m motmetrics.apps.eval_motchallenge --help


## datastrcture

Compute metrics for trackers using MOTChallenge ground-truth data.

Files
-----
All file content, ground truth and test files, have to comply with the
format described in

Milan, Anton, et al.
"Mot16: A benchmark for multi-object tracking."
arXiv preprint arXiv:1603.00831 (2016).
https://motchallenge.net/

Structure
---------

Layout for ground truth data
    <GT_ROOT>/<SEQUENCE_1>/gt/gt.txt
    <GT_ROOT>/<SEQUENCE_2>/gt/gt.txt
    ...

Layout for test data
    <TEST_ROOT>/<SEQUENCE_1>.txt
    <TEST_ROOT>/<SEQUENCE_2>.txt
    ...

Sequences of ground truth and test will be matched according to the `<SEQUENCE_X>`
string.

positional arguments:
  groundtruths          Directory containing ground truth files.
  tests                 Directory containing tracker result files

optional arguments:
  -h, --help            show this help message and exit
  --loglevel LOGLEVEL   Log level
  --fmt FMT             Data format
  --solver SOLVER       LAP solver to use for matching between frames.
  --id_solver ID_SOLVER
                        LAP solver to use for ID metrics. Defaults to --solver.
  --exclude_id          Disable ID metrics





## usage

python -m motmetrics.apps.eval_motchallenge GT_ROOT_PATH TEST_ROOT_PATH


example: 

python -m motmetrics.apps.eval_motchallenge /home/linx123-rtx/multi-bees-tracking/py-motmetrics/bbtracker_with_labels_as_detection/gt_root /home/linx123-rtx/multi-bees-tracking/py-motmetrics/bbtracker_with_labels_as_detection/with_yolov5l/test_root
