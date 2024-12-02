#/home/gagagk16/Rain/Derain/Dataset/RainDS/RainDS/RainDS_real/test_set/raindrop/
#/home/gagagk16/Rain/Derain/Dataset/RainDS/RainDS/RainDS_real/test_set/gt/
path=checkpoints/

python testing_model_Seting1.py --flag K1 \
    --base_channel 18 \
    --num_block 6 \
    --model_path $path \
    --eval_in_path_realRainDrop debug/data/input\
    --eval_gt_path_realRainDrop debug/data/gt\
    --save_path debug/