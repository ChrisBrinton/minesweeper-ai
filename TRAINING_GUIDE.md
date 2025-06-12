# Minesweeper AI Training Guide

This document provides a comprehensive guide to using the AI training system.

## Overview

The training system uses Deep Q-Networks (DQN) to learn optimal Minesweeper strategies through reinforcement learning.

## Entry Points

### Primary Training Script: `train_ai.py`

This is the main entry point for all AI training activities:

```bash
python train_ai.py --help
```

### Training Modes

#### 1. Resume Mode (Default)
Continues training from existing Enhanced V2 checkpoints:
```bash
python train_ai.py --mode resume --eval-method lightweight
```

#### 2. New Training Mode
Starts fresh training from scratch:
```bash
python train_ai.py --mode new --difficulty beginner --episodes 10000
```

#### 3. Benchmark Mode
Tests different evaluation methods:
```bash
python train_ai.py --mode benchmark --episodes 100
```

## Evaluation Methods

### Sequential
- Single-threaded evaluation
- Lowest resource usage
- Good for debugging

```bash
python train_ai.py --eval-method sequential
```

### Lightweight (Recommended)
- Thread-based parallelism
- Balanced performance/resource usage
- Default method

```bash
python train_ai.py --eval-method lightweight --workers 8
```

### Optimized
- Process-based parallelism
- Maximum performance
- Higher resource usage

```bash
python train_ai.py --eval-method optimized --workers 4
```

## Configuration Options

### Common Parameters
- `--workers`: Number of evaluation workers (auto-detected if not specified)
- `--difficulty`: Game difficulty (beginner, intermediate, expert)
- `--episodes`: Number of training episodes
- `--target-win-rate`: Target win rate to achieve (default: 0.50)
- `--save-dir`: Directory to save models and results
- `--verbose`: Enable verbose output

### Examples

```bash
# Resume training with specific target
python train_ai.py --mode resume --target-win-rate 0.60

# New expert training with many episodes
python train_ai.py --mode new --difficulty expert --episodes 50000

# Benchmark with custom worker count
python train_ai.py --mode benchmark --workers 12 --episodes 200
```

## File Structure

### Core Training Files
- `train_ai.py` - Primary training entry point
- `evaluation.py` - Consolidated evaluation system
- `src/ai/trainer.py` - Core DQN trainer
- `src/ai/models.py` - Neural network models
- `src/ai/environment.py` - RL environment wrapper

### Legacy Files (Removed)
- `parallel_evaluation.py` - Consolidated into `evaluation.py`
- `parallel_evaluation_optimized.py` - Consolidated into `evaluation.py`
- `lightweight_evaluation.py` - Consolidated into `evaluation.py`

## Performance Tips

1. **Use lightweight evaluation** for most scenarios
2. **Start with beginner difficulty** to test setup
3. **Monitor CPU usage** during parallel evaluation
4. **Save intermediate checkpoints** for long training runs
5. **Use benchmark mode** to find optimal worker count

## Troubleshooting

### High CPU Usage
```bash
# Reduce workers
python train_ai.py --workers 2

# Use sequential evaluation
python train_ai.py --eval-method sequential
```

### Memory Issues
```bash
# Use lightweight evaluation
python train_ai.py --eval-method lightweight --workers 4
```

### Finding Optimal Settings
```bash
# Benchmark different configurations
python train_ai.py --mode benchmark --episodes 50
```

## Training Phases

The system uses a three-phase approach for optimal learning:

1. **Foundation Phase** (15k episodes)
   - Extended learning with stable parameters
   - High exploration (ε: 0.9 → 0.3)
   - Learning rate: 0.001

2. **Stabilization Phase** (15k episodes)
   - Gradual parameter adjustment
   - Reduced exploration (ε: 0.3 → 0.15)
   - Learning rate: 0.0008

3. **Mastery Phase** (15k episodes)
   - Fine-tuning with preserved knowledge
   - Minimal exploration (ε: 0.15 → 0.05)
   - Learning rate: 0.0005

## Output and Results

Training results are saved to directories with timestamps:
- `models_beginner_YYYYMMDD_HHMMSS/` - New training results
- `models_beginner_enhanced_v2_parallel_resume/` - Resume training results

Each directory contains:
- Model checkpoints (`.pth` files)
- Training metrics (JSON files)
- Performance plots (if enabled)
