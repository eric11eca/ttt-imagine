export TOKENIZERS_PARALLELISM=false

# Base checkpoint directory (Wan2.1-T2V-14B)
CKPT="checkpoints/wan/Wan2.1-T2V-14B"

# E2E-TTT config (VideoTuna schema). Once the E2E-TTT flow exists, this will launch meta-training.
CONFIG="configs/008_wanvideo/wan2_1_t2v_14B_e2e_ttt_lora.yaml"

RESROOT="results/train"
EXPNAME="train_wanvideo_t2v_e2e_ttt_lora"
CURRENT_TIME=$(date +%Y%m%d%H%M%S)

python scripts/train_new.py -t \
  --ckpt "$CKPT" \
  --base "$CONFIG" \
  --logdir "$RESROOT" \
  --name "${EXPNAME}_${CURRENT_TIME}" \
  --devices 0, \
  --auto_resume

