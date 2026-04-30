# config.py — all constants and hyperparameters for the Pensieve re-implementation

# ---------------------------------------------------------------------------
# Video encoding — Envivio-Dash3 from the DASH reference client
# ---------------------------------------------------------------------------
BITRATE_LADDER_KBPS = [300, 750, 1200, 1850, 2850, 4300]   # kbps
NUM_BITRATES        = len(BITRATE_LADDER_KBPS)
CHUNK_DURATION_SEC  = 4.0       # seconds of video per chunk
NUM_CHUNKS          = 48        # total chunks in the test video
BUFFER_CAP_SEC      = 60.0      # maximum playback buffer

# ---------------------------------------------------------------------------
# State vector dimensions
# ---------------------------------------------------------------------------
HISTORY_LEN = 8    # k: number of past chunk measurements kept in state

# ---------------------------------------------------------------------------
# Reward coefficients  (QoE_lin from the paper, Table 1)
# ---------------------------------------------------------------------------
REBUFFER_PENALTY    = 4.3   # μ: penalty per second of rebuffering
SMOOTHNESS_PENALTY  = 1.0   # weight on |Δbitrate| in Mbps

# ---------------------------------------------------------------------------
# RL hyperparameters
# ---------------------------------------------------------------------------
GAMMA               = 0.99
ACTOR_LR            = 1e-4
CRITIC_LR           = 1e-3
ENTROPY_BETA_START  = 1.0
ENTROPY_BETA_END    = 0.1
ENTROPY_BETA_DECAY  = 1_200_000   # linear decay over this many training steps
NUM_TRAIN_ITERS     = 50_000
CHECKPOINT_EVERY    = 1_000
GRAD_CLIP_NORM      = 0.5
# PPO-specific
PPO_CLIP_EPS        = 0.2   # ε in the clipped surrogate objective
PPO_EPOCHS          = 4     # gradient epochs per collected rollout

# A3C-specific
A3C_NUM_WORKERS     = 8     # parallel worker processes
# ---------------------------------------------------------------------------
# Data splits
# ---------------------------------------------------------------------------
TRAIN_FRAC = 0.80
VAL_FRAC   = 0.10
TEST_FRAC  = 0.10

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
RESULTS_DIR     = "results"
CHECKPOINT_DIR  = "checkpoints"
