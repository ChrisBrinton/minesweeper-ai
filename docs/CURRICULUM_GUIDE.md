# Minesweeper AI Curriculum Learning Guide

## Overview

This guide covers the curriculum learning system for training Minesweeper AI models progressively from simple to complex board configurations. The system automatically advances through difficulty stages based on performance targets.

## Quick Start

### Starting New Curriculum
```bash
# Start from beginning (micro 3x3 boards)
python train_curriculum.py --mode new

# Start from specific stage
python train_curriculum.py --mode new --start-stage beginner

# Limit training to specific number of stages
python train_curriculum.py --mode new --max-stages 3
```

### Resuming Training
```bash
# Resume from last checkpoint
python train_curriculum.py --mode resume

# Check current status
python train_curriculum.py --status
```

### Viewing Curriculum Stages
```bash
# List all available stages
python train_curriculum.py --list-stages
```

## Curriculum Stages

The curriculum consists of 7 progressive stages designed to gradually increase complexity:

| Stage | Board Size | Mines | Density | Target Win Rate | Episodes Range |
|-------|------------|-------|---------|-----------------|----------------|
| 1. Micro | 3×3 | 1 | 11.1% | 70.0% | 2,000 - 5,000 |
| 2. Tiny | 5×5 | 3 | 12.0% | 60.0% | 3,000 - 8,000 |
| 3. Small | 7×7 | 8 | 16.3% | 55.0% | 5,000 - 12,000 |
| 4. Mini Beginner | 8×8 | 9 | 14.1% | 50.0% | 7,000 - 15,000 |
| 5. Beginner | 9×9 | 10 | 12.3% | 45.0% | 10,000 - 25,000 |
| 6. Intermediate | 16×16 | 40 | 15.6% | 35.0% | 15,000 - 40,000 |
| 7. Expert | 16×30 | 99 | 20.6% | 25.0% | 25,000 - 60,000 |

## Features

### Automatic Progression
- **Performance-based advancement**: Only advances when target win rate is achieved
- **Knowledge transfer**: Each stage starts with weights from the previous stage
- **Adaptive training**: Training parameters adjust based on board complexity
- **Early stopping**: Prevents overtraining with patience-based stopping

### Robust Training Pipeline
- **Batch processing**: Trains in manageable batches (typically 1,000 episodes)
- **Checkpoint saving**: Regular model saves and progress tracking
- **Evaluation integration**: Built-in performance evaluation after each batch
- **Resume capability**: Can resume from any interruption point

### Monitoring and Control
- **Real-time progress**: Visual progress bars and detailed logging
- **Status checking**: View current stage, completion status, and statistics
- **Flexible control**: Start from any stage, limit training duration
- **Comprehensive logging**: All training metrics and decisions are logged

## Advanced Usage

### Custom Evaluation Settings
```bash
# Use sequential evaluation for more accuracy
python train_curriculum.py --mode new --eval-method sequential

# Adjust worker count
python train_curriculum.py --mode new --workers 16
```

### Custom Save Directory
```bash
# Use custom directory
python train_curriculum.py --mode new --save-dir my_curriculum_models
```

### Development and Testing
```bash
# Enable verbose output
python train_curriculum.py --mode new --verbose

# Test system functionality
python test_curriculum.py
```

## Implementation Details

### Knowledge Transfer
The curriculum system implements progressive knowledge transfer:

1. **Weight Initialization**: Each stage starts with the best model from the previous stage
2. **Transfer Learning**: Reduced learning rate and moderate exploration for transferred models
3. **Adaptive Parameters**: Training hyperparameters adjust based on board size and mine density

### Training Configuration
Training parameters are automatically adjusted per stage:

- **Small boards** (≤25 cells): Higher learning rate, smaller batch size, less memory
- **Medium boards** (25-64 cells): Balanced parameters
- **Large boards** (≥400 cells): Lower learning rate, larger batch size, more memory
- **High mine density** (>15%): More conservative exploration strategy

### Evaluation Strategy
Each stage uses appropriate evaluation methods:

- **Lightweight evaluation**: Fast approximate win rate estimation during training
- **Sequential evaluation**: Detailed evaluation for final stage assessment
- **Custom board evaluation**: Evaluates specifically on the target board configuration

## Troubleshooting

### Common Issues and Solutions

#### "DQNTrainer interface error"
**Fixed in current version.** The curriculum system now properly interfaces with DQNTrainer by:
- Creating trainers with correct `max_episodes` configuration
- Using proper batch-based training approach
- Handling knowledge transfer between batches correctly

#### "CUDA out of memory"
- Reduce batch size in training configuration
- Use fewer evaluation workers: `--workers 8`
- Consider using CPU-only training for very large boards

#### "Low win rates / No progress"
- Check if target win rates are too ambitious
- Verify knowledge transfer is working (check logs for "Loading knowledge from previous stage")
- Consider training longer on current stage before advancing

#### "Training interrupted"
- Use `--mode resume` to continue from last checkpoint
- Check `--status` to see current progress
- All progress is automatically saved and can be resumed

### Model Storage Structure
```
models/curriculum/YYYYMMDD_HHMMSS/
├── curriculum_progress.json          # Overall progress tracking
├── curriculum_summary.json           # Final training summary
├── stage_micro/                       # Individual stage directories
│   ├── best_stage_model.pth          # Best model for this stage
│   ├── dqn_final.pth                 # Final training checkpoint
│   └── training_metrics.json         # Training metrics
├── stage_tiny/
│   └── ...
└── ...
```

### Configuration Files
The curriculum system uses several configuration approaches:

1. **Stage Configuration**: Hardcoded in `CurriculumConfig.CURRICULUM_STAGES`
2. **Training Parameters**: Dynamically generated in `_get_trainer_config()`
3. **Progress Tracking**: Persisted in `curriculum_progress.json`

## Best Practices

### For Research/Experimentation
- Start with `--max-stages 2` to test the system quickly
- Use `--verbose` to understand the training process
- Monitor GPU memory usage during large board training

### For Production Training
- Use default settings for most reliable results
- Plan for 12-24 hours of training time for full curriculum
- Ensure adequate disk space (several GB for full curriculum)
- Use `--eval-method sequential` for final evaluation accuracy

### For Development
- Run `python test_curriculum.py` before major changes
- Test with `--start-stage micro --max-stages 1` for quick validation
- Use smaller episode ranges for faster iteration

## Performance Expectations

### Training Time Estimates
- **Micro stage**: 5-10 minutes
- **Tiny stage**: 10-20 minutes  
- **Small stage**: 20-40 minutes
- **Mini Beginner**: 30-60 minutes
- **Beginner**: 45-90 minutes
- **Intermediate**: 2-4 hours
- **Expert**: 4-8 hours

**Total curriculum**: 8-16 hours depending on hardware and target achievement

### Win Rate Progression
Expected progression through curriculum:
- Micro (3×3): 70%+ win rate typically achieved
- Tiny (5×5): 60%+ win rate with good spatial reasoning
- Small (7×7): 55%+ win rate with pattern recognition
- Beginner (9×9): 45%+ win rate with full game understanding
- Intermediate (16×16): 35%+ win rate with advanced strategy
- Expert (16×30): 25%+ win rate represents expert-level play

### Hardware Requirements
- **Minimum**: 8GB RAM, GTX 1060 or equivalent
- **Recommended**: 16GB RAM, RTX 3070 or equivalent  
- **Optimal**: 32GB RAM, RTX 4080 or equivalent

## API Reference

### CurriculumLearningTrainer
Main class for curriculum training:

```python
from src.ai.curriculum import CurriculumLearningTrainer

trainer = CurriculumLearningTrainer(
    save_dir="models/my_curriculum",
    start_stage="beginner", 
    evaluation_method="sequential"
)

# Run full curriculum
results = trainer.run_curriculum(max_stages=5)

# Check status
status = trainer.get_curriculum_status()
```

### CurriculumConfig
Static configuration for curriculum stages:

```python
from src.ai.curriculum import CurriculumConfig

# Get stage information
stage_info = CurriculumConfig.get_stage_config("beginner")
next_stage = CurriculumConfig.get_next_stage("beginner")
```

### CurriculumTracker
Progress tracking and persistence:

```python
from src.ai.curriculum import CurriculumTracker

tracker = CurriculumTracker("models/curriculum/20241201_120000")
current_stage = tracker.get_current_stage()
total_episodes = tracker.get_total_episodes()
```

## Integration with Existing Tools

### With Standard Training
```bash
# Train standard model on results
python train_ai.py --difficulty beginner --resume models/curriculum/20241201_120000/stage_beginner/
```

### With Evaluation System
```bash
# Evaluate curriculum model
python evaluation.py --model-path models/curriculum/20241201_120000/stage_expert/best_stage_model.pth --difficulty expert
```

### With Monitoring
```bash
# Monitor training progress
python monitoring/monitor_training.py --model-dir models/curriculum/20241201_120000/
```

---

## Summary

The curriculum learning system provides a robust, automated approach to training Minesweeper AI models progressively. It handles the complexity of managing multiple difficulty stages while providing comprehensive monitoring and control options.

**Key benefits:**
- ✅ Automated progression through difficulty levels
- ✅ Knowledge transfer between stages  
- ✅ Robust checkpoint and resume system
- ✅ Comprehensive evaluation and monitoring
- ✅ Flexible control and configuration options

For questions or issues, refer to the troubleshooting section or examine the detailed logs generated during training.
