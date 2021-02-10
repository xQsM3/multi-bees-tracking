The repository comes with code to train a model on the [ANTS] datasets, including ``bbox_train.zip`` and ``bbox_test.zip``.


## Training on ANTS

To train on ANTS, Unzip ``bbox_train.zip`` to the ANTS directory. The following description assumes they are stored in ``./ANTS/bbox_train``. Training can be started with the following command:
```
python train_ants.py \
    --dataset_dir=./ANTS \
    --loss_mode=cosine-softmax \
    --log_dir=./output/ants/ \
--run_id=cosine-softmax


python train_ants.py \
    --dataset_dir=./BUMBLEBEES \
    --loss_mode=cosine-softmax \
    --log_dir=./output/bees/ \
--run_id=cosine-softmax
``` 
Again, this will create a directory `./output/ants/cosine-softmax` where
TensorFlow checkpoints are stored and which can be monitored using
``tensorboard``:
```
tensorboard --logdir ./output/ants/cosine-softmax --port 7006
```

## Model export

To export your trained model for use with the
[ant tracking], run the following command:
```
python train_ants.py --mode=freeze --restore_path=PATH_TO_CHECKPOINT
```
This will create a ``ants.pb`` file which can be supplied to ant Tracking. 
