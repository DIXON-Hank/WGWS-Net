python training_Setting1_Stage1.py --experiment_name debug \
    --debug True \
    --base_channel 18 --fix_sample 1000 --BATCH_SIZE 4 --Crop_patches 256 \
    --EPOCH 200 --T_period 100  --learning_rate 0.002 \
    --addition_loss VGG \
    --Aug_regular True --print_frequency 200