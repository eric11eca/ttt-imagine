config='configs/007_hunyuanvideo/hunyuanvideo_t2v_diffuser.yaml'
prompt_file="inputs/t2v/hunyuanvideo/tyler_swift_video/labels.txt"
current_time=$(date +%Y%m%d%H%M%S)
savedir="results/t2v/$current_time-hunyuanvideo"
ckpt="/work/nlp/hzhao/ttt-imagine/checkpoints/hunyuanvideo/HunyuanVideo"

python3 scripts/inference_cogvideo.py \
--ckpt_path $ckpt \
--config $config \
--prompt_file $prompt_file \
--savedir $savedir \
--bs 1 --height 256 --width 256 \
--fps 16 \
--seed 6666 \