export PYTHONPATH=./ \
&& python scripts/sample_images.py \
--model_path=checkpoints/aigc-ddpm-tiny_mnist_ddpm-2023-07-12-13-18-iteration-20000-model.pth \
--save_dir=eval_out -v
