MODEL='UNet' 
BASE_PATH='/base_path/to/bony_labyrinth_dataset'
BATCH_SIZE=4
NUM_EPOCHS=100
DEVICE='cuda'
DATASET_TYPE='bony_labyrinth_dataset'
LR=0.001
TASK='segmentation'
EXP_DIR=${BASE_PATH}'/Seg_experiments/experiments/'${DATASET_TYPE}'/'${TASK}'/'${MODEL}'/lr_'${LR}
PREDS_DIR=${EXP_DIR}'/Predictions/'

TRAIN_PATH=${BASE_PATH}'/BLD/DS_uCT/train'
VALIDATION_PATH=${BASE_PATH}'/BLD/DS_uCT/val'

echo python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --preds-dir ${PREDS_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --lr ${LR}
python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --preds-dir ${PREDS_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --lr ${LR}

