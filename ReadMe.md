# Pensieve: Neural Adaptive Video Streaming

**Citation:** Hongzi Mao, Ravi Netravali, Mohammad Alizadeh. "Neural Adaptive Video Streaming with Pensieve." SIGCOMM 2017, Los Angeles, CA.

---

## Plain-English Summary

**Problem:** When you stream video, your player constantly decides what quality (bitrate) to request for each 4-second chunk of video. Existing algorithms use hand-crafted rules — like "pick the bitrate that matches the average throughput from the last 5 chunks" — but these rules are rigid. They fail when the network behaves unexpectedly or when you want to optimize for a different quality goal (e.g., preferring no buffering pauses vs. preferring HD resolution). The core issue is that no fixed rule works well across all network conditions and viewer preferences.

**Conclusion:** Pensieve sidesteps hand-crafted rules entirely by training a neural network with reinforcement learning to make bitrate decisions. The network learns purely from experience — observing what happened after past choices — with no assumptions baked in. It outperforms the best existing algorithm by 12–25% across a wide range of network conditions and quality objectives, and comes within 0.2% of the theoretical best possible online algorithm.

---

## Implementation (A4 Course Project)

### Train
```bash
python train.py                          # 50k iters, 1000 synthetic traces
python train.py --iters 5000            # quick run
python train.py --traces path/to/dir    # use real FCC/HSDPA traces
```
Saves checkpoints to `checkpoints/` every 1000 iters; best val QoE → `checkpoints/best_model.pt`.

### Evaluate
```bash
python evaluate.py                       # uses best_model.pt + 1000 synthetic traces
python evaluate.py --model checkpoints/ckpt_010000.pt --traces path/to/dir
```
Writes to `results/`: `metrics_table.csv`, `qoe_cdf.png`, `qoe_components.png`, `rollout_trace_{1,2,3}.png`.

### Run unit tests
```bash
python env.py          # 6 env tests
python -m model.network  # network forward-pass smoke test
```

### Key files
| File | Role |
|---|---|
| `config.py` | All hyperparameters (bitrate ladder, QoE weights, LR, etc.) |
| `env.py` | `VideoStreamingEnv` — DASH simulator |
| `model/network.py` | `ActorCritic` CNN + `normalise_state` |
| `model/a2c.py` | `A2CTrainer`, `RolloutBuffer` |
| `traces/loader.py` | Load real traces or generate synthetic ones |
| `baselines/` | Fixed, ThroughputBased, BufferBased agents |

### Use real traces
Drop single-column (Mbps) or two-column (timestamp, kbps) `.txt` files into any directory and pass `--traces path/to/dir` to both scripts.

---

## Method

### Problem Setup (RL Framing)
- **Agent:** ABR controller
- **State** (inputs after each chunk download):
  - Last k=8 chunk throughput measurements
  - Last k=8 chunk download times
  - Sizes of next chunk at all available bitrates
  - Current playback buffer occupancy
  - Number of chunks remaining in video
  - Bitrate of last downloaded chunk
- **Action:** Bitrate for the next chunk
- **Reward:** QoE score for that chunk (see metrics below)

### QoE Metrics Evaluated
| Metric | Bitrate utility q(R) | Rebuffer penalty µ |
|--------|----------------------|--------------------|
| QoE_lin | R (linear) | 4.3 |
| QoE_log | log(R/R_min) | 2.66 |
| QoE_hd | Step function (high reward only for HD bitrates) | 8 |

All metrics also subtract a smoothness penalty for bitrate switches.

### Neural Network Architecture
- **Actor network** (outputs bitrate probability distribution):
  - Two separate 1D-CNN branches (128 filters, size 4, stride 1) for throughput history and next chunk sizes
  - Other inputs (buffer level, chunks remaining, last bitrate) concatenated into a 128-neuron hidden layer
  - Softmax output over available bitrates
- **Critic network:** Same structure, linear output (estimates value function)
- Single hidden layer performed best; deeper networks degraded performance in sweeps

### Training Algorithm: A3C (Asynchronous Advantage Actor-Critic)
- 16 parallel agents each running the chunk-level simulator with different network traces
- Agents send (state, action, reward) tuples to a central agent asynchronously
- Central agent updates actor/critic networks via policy gradient
- Entropy regularization encourages exploration (β decays from 1 → 0.1 over 10^5 iterations)
- Discount factor γ = 0.99 (effectively planning ~100 steps ahead)
- Actor learning rate: 10⁻⁴; Critic learning rate: 10⁻³
- ~50,000 iterations, ~300ms each with 16 parallel agents → **~4 hours total training**

### Simulation Environment
- Chunk-level simulator: assigns download time based on chunk bitrate and network trace throughput
- Tracks playback buffer, rebuffering events
- Simulates 100 hours of video downloads in ~10 minutes
- Requires TCP slow-start-restart to be disabled on server for accurate simulation

### Multi-Video Generalization
- Canonical input/output format supporting up to 13 bitrate levels
- Videos with fewer bitrates zero-pad unused input slots
- Output softmax masked to only the bitrates the current video supports (mask is independent of NN parameters, so backprop is unaffected)

### Deployment Architecture
- Pensieve runs on a standalone ABR server (Python BaseHTTPServer)
- Video clients include state observations in each chunk request
- Server feeds observations through actor NN and returns the bitrate decision
- Clients remain stateless; NN inference stays server-side (suits low-power clients)

---

## Evaluation Setup

### Network Traces
- **FCC broadband dataset:** 1M+ throughput traces → 1,000 traces of 320s sampled; 80/20 train/test split
- **Norway HSDPA dataset:** 30 minutes mobile data (bus, train) → 1,000 sliding-window traces
- Filtered to average throughput < 6 Mbps and minimum > 0.2 Mbps to exclude trivial cases

### Baselines
1. **Buffer-Based (BB):** Reservoir/cushion rule (5s reservoir, 10s cushion)
2. **Rate-Based (RB):** Harmonic mean of last 5 chunks; pick highest bitrate below prediction
3. **BOLA:** Lyapunov optimization on buffer occupancy alone
4. **MPC:** Optimizes QoE over 5-chunk horizon using throughput prediction
5. **robustMPC:** MPC with conservative throughput estimate (normalized by max past error)
6. **Offline Optimal:** Dynamic programming with perfect future throughput knowledge (unachievable upper bound)

### Test Video
"Envivio-Dash3": 193s, 48 chunks, bitrates {300, 750, 1200, 1850, 2850, 4300} kbps ({240, 360, 480, 720, 1080, 1440}p)

---

## Results

### Main QoE Comparison (vs. robustMPC, the best baseline)
| Network | QoE_lin | QoE_log | QoE_hd |
|---------|---------|---------|--------|
| FCC broadband | +15.5% | +18.9% | +24.6% |
| Norway HSDPA | similar gains | similar gains | similar gains |

- Pensieve matches or beats all baselines in every scenario tested
- Within **9.6%–14.3% of offline optimal** across all networks/metrics

### Optimality Check
- On a controlled Markov model where online optimal is computable: Pensieve achieves within **0.2%** of online optimal, which is ~9% below offline optimal

### QoE Breakdown
- Most of Pensieve's gains come from **reducing rebuffering** (10.6%–32.8% less across metrics)
- Pensieve balances bitrate utility and smoothness differently per QoE metric — it learns distinct strategies per objective automatically

### Generalization to New Networks (Real-World)
- Tested on Verizon LTE, public WiFi (coffee shop), Boston–Shanghai WAN
- Trained only on FCC + HSDPA traces — still outperformed BOLA and robustMPC on all three new networks

### Synthetic Training Data
- Pensieve trained on synthetic Markovian traces (no real data) achieves within **1.6%–10.8%** of Pensieve trained on real traces
- Beats robustMPC on both broadband and HSDPA test sets

### Multi-Video Generalization
- Single model trained on 1,000 synthetic videos achieves within **3.2%** of a model trained specifically on the test video

### vs. Tabular RL
- **46.3% higher QoE** than tabular Q-learning (prior RL-for-ABR approaches)
- Tabular methods must assume Markovian bandwidth (only 1 past measurement in state), which is insufficient for real network dynamics

### System Parameters (Sensitivity)
- Performance plateaus at **32+ CNN filters / hidden neurons**; 128 (default) adds stability
- Single hidden layer is best; more layers degrade without extended training
- **100ms RTT to ABR server:** only 3.5% QoE reduction — hidden by playback buffer
- Past chunk history: 8 chunks is the sweet spot; 16 adds only ~1% improvement
