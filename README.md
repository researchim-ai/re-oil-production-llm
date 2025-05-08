# re-oil-production-llm

![image](https://github.com/user-attachments/assets/6883733b-126d-4529-9cd6-850a5e500a20)

## LLM for Oil Production Optimization

This project explores the application of Large Language Models (LLMs) for optimizing oil production strategies. We use reinforcement learning techniques, specifically GRPO, to train language models to make optimal decisions for oil well control.

## Project Overview

Oil production optimization requires finding a balance between immediate extraction rates and long-term reservoir management. This project implements:

- A simulator environment for oil wells (both single and multi-well scenarios)
- A GRPO-based training framework for fine-tuning LLMs
- Prompt engineering techniques to guide LLM responses
- Evaluation methods to measure optimization performance

The trained models learn to control the choke opening (from 0 to 1) to maximize overall oil production over the entire simulation period.

## Features

- **Single and Multi-Well Simulations**: Simulate both standalone and interconnected well systems
- **GRPO Training**: Efficient policy optimization for LLMs
- **Prompt Engineering**: Structured input formats for consistent and effective LLM responses
- **WandB and TensorBoard Integration**: Comprehensive experiment tracking and visualization
- **Quantization Support**: 4-bit and 8-bit quantization for efficient model training
- **LoRA Fine-tuning**: Low-Rank Adaptation for parameter-efficient tuning

## Installation

1. Clone the repository:
```bash
git clone https://github.com/researchim-ai/re-oil-production-llm.git
cd re-oil-production-llm
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

To run training:

```bash
python -m grpo.main --forecast_days 1 --simulation_max_time 4 --kl_weight 0.0 --lr 1e-5 --clip_eps 0.3 --gamma 0.99 --total_steps 1000 --rollouts_per_step 8 --train_batch_size 8 --temperature 0.7 --wandb --use_discrete_actions
```

### Main Parameters

The script supports multiple parameters for customizing the training process:

```
--run_name               Custom run name for logging
--log_dir                Root directory for logs
--wandb                  Use WandB for logging
--wandb_project          WandB project name
--seed                   Random seed (default: 42)
--device_index           CUDA device index (default: 0)
--model_name             Model name or path (default: "Qwen/Qwen2.5-3B-Instruct")
--checkpoint_path        Path to save checkpoints
--checkpoint_interval    Save checkpoint every N global steps
--total_steps            Total number of optimization steps
--rollouts_per_step      Number of simulation episodes per global step
--multi_well             Use multi-well simulator instead of single well
--initial_pressure       Initial reservoir pressure (atm)
--productivity_index     Productivity index (m3/day/atm)
--total_volume           Total reservoir volume (m3)
--forecast_days          Number of days to forecast in each step
```

## Current State

The project currently provides fully functional implementations of both single-well and multi-well optimization scenarios. The single-well scenario serves as a simpler baseline and proof of concept, while the multi-well implementation accounts for more complex reservoir dynamics with interactions between wells.

## Pipeline

1. **Simulation Environment**: The system simulates oil well behavior, including pressure changes, flow rates, and cumulative production.
2. **LLM Interaction**: The language model receives the current state and history, then decides on the optimal choke opening.
3. **Reinforcement Learning**: The model is trained based on rewards from oil production outcomes.
4. **Evaluation**: The trained model's performance is evaluated on extended simulation periods.

## License

This project is provided as research software.

## Citation

If you use this project in your research, please cite:

```
@software{llm_oil_production,
  author = {researchim},
  title = {LLM for Oil Production Optimization},
  year = {2025},
  url = {https://github.com/yourusername/re-oil-production-llm}
}
```