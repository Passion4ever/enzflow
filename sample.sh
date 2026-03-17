for steps in 10 20 50 100; do
    CUDA_VISIBLE_DEVICES=0 python scripts/sample.py \
        --ckpt checkpoints/run_name/step_xxx.pt \
        --batch_size 5 \
        --num_batches 4 \
        --length 100 200 300 500 800 \
        --steps "$steps"
done