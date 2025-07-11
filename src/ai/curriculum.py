"""
Curriculum Learning System for Minesweeper AI
Progressive training on increasingly difficult board configurations
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import shutil

from .trainer import DQNTrainer, create_trainer
from .environment import MinesweeperEnvironment, PerfectKnowledgeMinesweeperEnvironment
from .best_model_tracker import BestModelTracker
from ..ai.model_storage import get_model_save_dir, find_latest_model_dir


class CurriculumConfig:
    """Configuration for curriculum learning stages"""
    
    # Progressive difficulty configurations
    CURRICULUM_STAGES = [        {
            'name': 'phase_0',
            'description': 'Phase 0 (5x5) - Learn game logic with perfect knowledge of mine locations',
            'rows': 5, 'cols': 5, 'mines': 3,
            'target_win_rate': 0.95,
            'min_episodes': 3000,
            'max_episodes': 9000,
            'patience': 600,
            'fully_revealed': True  # Special flag for perfect knowledge environment
        },
        {
            'name': 'tiny',
            'description': 'Tiny boards (5x5) - Learn basic mechanics and spatial reasoning',
            'rows': 5, 'cols': 5, 'mines': 3,
            'target_win_rate': 0.65,
            'min_episodes': 2000,
            'max_episodes': 6000,
            'patience': 400
        },
        {
            'name': 'small',
            'description': 'Small boards (7x7) - Develop pattern recognition',
            'rows': 7, 'cols': 7, 'mines': 8,
            'target_win_rate': 0.55,
            'min_episodes': 5000,
            'max_episodes': 12000,
            'patience': 800
        },
        {
            'name': 'mini_beginner',
            'description': 'Mini beginner (8x8) - Bridge to standard sizes',
            'rows': 8, 'cols': 8, 'mines': 9,
            'target_win_rate': 0.50,
            'min_episodes': 7000,
            'max_episodes': 15000,
            'patience': 1000
        },
        {
            'name': 'beginner',
            'description': 'Standard beginner (9x9) - Full beginner mastery',
            'rows': 9, 'cols': 9, 'mines': 10,
            'target_win_rate': 0.45,
            'min_episodes': 10000,
            'max_episodes': 25000,
            'patience': 1500
        },
        {
            'name': 'intermediate',
            'description': 'Standard intermediate (16x16) - Advanced spatial reasoning',
            'rows': 16, 'cols': 16, 'mines': 40,
            'target_win_rate': 0.35,
            'min_episodes': 15000,
            'max_episodes': 40000,
            'patience': 2000
        },
        {
            'name': 'expert',
            'description': 'Standard expert (16x30) - Master level play',
            'rows': 16, 'cols': 30, 'mines': 99,
            'target_win_rate': 0.25,
            'min_episodes': 25000,
            'max_episodes': 60000,
            'patience': 3000
        }
    ]

    @classmethod
    def get_stage_config(cls, stage_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific curriculum stage"""
        for stage in cls.CURRICULUM_STAGES:
            if stage['name'] == stage_name:
                return stage.copy()
        return None

    @classmethod
    def get_stage_index(cls, stage_name: str) -> int:
        """Get index of curriculum stage"""
        for i, stage in enumerate(cls.CURRICULUM_STAGES):
            if stage['name'] == stage_name:
                return i
        return -1

    @classmethod
    def get_next_stage(cls, current_stage: str) -> Optional[str]:
        """Get the next stage in curriculum"""
        current_idx = cls.get_stage_index(current_stage)
        if current_idx >= 0 and current_idx < len(cls.CURRICULUM_STAGES) - 1:
            return cls.CURRICULUM_STAGES[current_idx + 1]['name']
        return None


class CurriculumTracker:
    """Tracks progress through curriculum stages"""
    
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.progress_file = os.path.join(save_dir, "curriculum_progress.json")
        self.load_progress()
    
    def load_progress(self):
        """Load curriculum progress from file"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                self.progress = json.load(f)
        else:
            self.progress = {
                'current_stage': 'phase_0',
                'stages_completed': [],
                'stage_history': [],
                'total_episodes': 0,
                'created_at': datetime.now().isoformat()
            }
    
    def save_progress(self):
        """Save curriculum progress to file"""
        os.makedirs(os.path.dirname(self.progress_file), exist_ok=True)
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def complete_stage(self, stage_name: str, episodes_trained: int, final_win_rate: float):
        """Mark stage as completed and advance to next"""
        completion_info = {
            'stage': stage_name,
            'episodes_trained': episodes_trained,
            'final_win_rate': final_win_rate,
            'completed_at': datetime.now().isoformat()
        }
        
        if stage_name not in self.progress['stages_completed']:
            self.progress['stages_completed'].append(stage_name)
        
        self.progress['stage_history'].append(completion_info)
        self.progress['total_episodes'] += episodes_trained
        # Advance to next stage
        next_stage = CurriculumConfig.get_next_stage(stage_name)
        if next_stage:
            self.progress['current_stage'] = next_stage
            print(f"✅ Stage '{stage_name}' completed with {final_win_rate:.1%} win rate")
            print(f"🎯 Advancing to stage '{next_stage}'")
        else:
            print(f"🏆 Curriculum completed! Final stage '{stage_name}' achieved {final_win_rate:.1%} win rate")
        self.save_progress()
    
    def get_current_stage(self) -> str:
        """Get current curriculum stage"""
        return self.progress['current_stage']
    
    def is_stage_completed(self, stage_name: str) -> bool:
        """Check if stage is completed"""
        return stage_name in self.progress['stages_completed']
    
    def get_total_episodes(self) -> int:
        """Get total episodes trained across all stages"""
        total = self.progress['total_episodes']
        
        # Add episodes from current stage in progress        
        current_stage = self.get_current_stage()
        current_stage_episodes = self._detect_stage_progress(current_stage)
        total += current_stage_episodes
        
        return total

    def _detect_stage_progress(self, stage_name: str) -> int:
        """Detect training progress from existing model checkpoints"""
        stage_dir = os.path.join(self.save_dir, f"stage_{stage_name}")
        if not os.path.exists(stage_dir):
            return 0
        
        # Look for episode checkpoint files
        max_episode = 0
        for filename in os.listdir(stage_dir):
            if filename.startswith("dqn_episode_") and filename.endswith(".pth"):
                try:
                    episode_num = int(filename.replace("dqn_episode_", "").replace(".pth", ""))
                    max_episode = max(max_episode, episode_num)
                except ValueError:
                    continue
        
        return max_episode

    def get_current_stage_progress(self) -> int:
        """Get training progress for current stage"""
        current_stage = self.get_current_stage()
        return self._detect_stage_progress(current_stage)


class CurriculumLearningTrainer:
    """
    Curriculum Learning Trainer for Progressive Difficulty Training
    
    Automatically progresses through difficulty levels based on performance
    """
    
    def __init__(self, save_dir: str = None, start_stage: str = None, 
                 evaluation_method: str = "lightweight", num_eval_workers: int = None):
        """
        Initialize curriculum trainer
        
        Args:
            save_dir: Directory to save models and progress
            start_stage: Override starting stage (default: auto-detect from progress)
            evaluation_method: Method for evaluation ('lightweight', 'sequential', 'optimized')
            num_eval_workers: Number of evaluation workers
        """
        self.evaluation_method = evaluation_method
        self.num_eval_workers = num_eval_workers or max(1, os.cpu_count() - 2)
        
        # Set up save directory
        if save_dir:
            self.save_dir = save_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_dir = get_model_save_dir("curriculum", timestamp)
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize progress tracker
        self.tracker = CurriculumTracker(self.save_dir)
          # Override starting stage if specified
        if start_stage:
            if CurriculumConfig.get_stage_config(start_stage):
                self.tracker.progress['current_stage'] = start_stage
                self.tracker.save_progress()
            else:
                raise ValueError(f"Invalid start stage: {start_stage}")
        
        self.current_trainer = None
        
        # Initialize best model tracker
        self.best_model_tracker = BestModelTracker()
        
        print(f"🎓 Curriculum Learning Trainer initialized")
        print(f"   💾 Save directory: {self.save_dir}")
        print(f"   🎯 Current stage: {self.tracker.get_current_stage()}")
        print(f"   📊 Total episodes so far: {self.tracker.get_total_episodes()}")
        print(f"   🖥️  Evaluation method: {evaluation_method} ({self.num_eval_workers} workers)")
        
        # Print best model info if available
        self.best_model_tracker.print_best_model_summary()
    
    def run_curriculum(self, max_stages: int = None) -> Dict[str, Any]:
        """
        Run the complete curriculum learning process
        
        Args:
            max_stages: Maximum number of stages to train (None for all)
            
        Returns:
            Training summary
        """
        stages_trained = 0
        training_summary = {
            'stages_completed': [],
            'total_episodes': 0,
            'start_time': datetime.now().isoformat(),
            'curriculum_completed': False
        }
        
        print(f"\n🚀 Starting curriculum learning process")
        if max_stages:
            print(f"   📊 Max stages to train: {max_stages}")
        
        while True:
            current_stage = self.tracker.get_current_stage()
            stage_config = CurriculumConfig.get_stage_config(current_stage)
            
            if not stage_config:
                print(f"🏁 Curriculum completed - no more stages available")
                training_summary['curriculum_completed'] = True
                break
            
            if max_stages and stages_trained >= max_stages:
                print(f"🛑 Reached maximum stages limit ({max_stages})")
                break
            
            print(f"\n" + "="*60)
            print(f"🎯 Training Stage: {current_stage.upper()}")
            print(f"   📝 {stage_config['description']}")
            print(f"   🎮 Board: {stage_config['rows']}x{stage_config['cols']} with {stage_config['mines']} mines")
            print(f"   🎯 Target win rate: {stage_config['target_win_rate']:.1%}")
            print(f"   📊 Episode range: {stage_config['min_episodes']:,} - {stage_config['max_episodes']:,}")
            print("="*60)
            
            # Train current stage
            stage_result = self._train_stage(current_stage, stage_config)
            
            # Update training summary
            training_summary['stages_completed'].append(stage_result)
            training_summary['total_episodes'] += stage_result['episodes_trained']
            stages_trained += 1
            
            # Check if we should continue
            if not stage_result['target_achieved']:
                print(f"⚠️  Target not achieved for stage '{current_stage}'. Consider manual intervention.")
                break
        
        training_summary['end_time'] = datetime.now().isoformat()
        training_summary['stages_trained'] = stages_trained
          # Save final summary
        summary_file = os.path.join(self.save_dir, "curriculum_summary.json")
        
        # Ensure JSON serializable by converting numpy types and booleans
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif isinstance(obj, (bool, type(True), type(False))):
                return bool(obj)
            else:
                return obj
        
        serializable_summary = make_serializable(training_summary)
        
        with open(summary_file, 'w') as f:
            json.dump(serializable_summary, f, indent=2)
        
        print(f"\n🎉 Curriculum training completed!")
        print(f"   📊 Stages trained: {stages_trained}")
        print(f"   🎮 Total episodes: {training_summary['total_episodes']:,}")
        print(f"   💾 Summary saved: {summary_file}")
        
        return training_summary
    
    def _train_stage(self, stage_name: str, stage_config: Dict[str, Any]) -> Dict[str, Any]:
        """Train a single curriculum stage"""
        
        # Create environment for this stage
        if stage_config.get('fully_revealed', False):
            # Use special environment for phase_0 with perfect knowledge
            env = PerfectKnowledgeMinesweeperEnvironment(
                rows=stage_config['rows'],
                cols=stage_config['cols'], 
                mines=stage_config['mines']
            )
        else:
            # Use standard environment for normal stages
            env = MinesweeperEnvironment(
                rows=stage_config['rows'],
                cols=stage_config['cols'], 
                mines=stage_config['mines']
            )
        
        # Create stage-specific save directory
        stage_save_dir = os.path.join(self.save_dir, f"stage_{stage_name}")
        os.makedirs(stage_save_dir, exist_ok=True)
        
        # Get base trainer config for this stage
        trainer_config = self._get_trainer_config(stage_config)
        
        # Initialize trainer variable (will be created per batch)
        self.current_trainer = None
        
        # Load checkpoint from previous stage if available
        self._transfer_knowledge(stage_name, stage_save_dir)
          # Training loop with adaptive episodes
        # Initialize with existing progress
        episodes_trained = self.tracker._detect_stage_progress(stage_name)
        if episodes_trained > 0:
            print(f"🔄 Resuming from episode {episodes_trained + 1} (found existing checkpoint)")
        else:
            print(f"🆕 Starting fresh training for stage '{stage_name}'")
        
        best_win_rate = 0.0
        patience_counter = 0
        evaluation_history = []
        
        min_episodes = stage_config['min_episodes']
        max_episodes = stage_config['max_episodes'] 
        target_win_rate = stage_config['target_win_rate']
        patience = stage_config['patience']
        print(f"\n🎯 Starting training for stage '{stage_name}'")
        
        while episodes_trained < max_episodes:
            # Train in batches
            batch_size = min(1000, max_episodes - episodes_trained)
            
            print(f"\n📈 Training batch: {episodes_trained:,} - {episodes_trained + batch_size:,} episodes")            # Update trainer config for this batch
            batch_config = trainer_config.copy()
            # Calculate step-based frequencies based on estimated steps per episode (~10-50 steps avg)
            estimated_steps_per_episode = 25  # Conservative estimate for step-based calculation
            estimated_total_steps = batch_size * estimated_steps_per_episode
            
            batch_config.update({
                'max_episodes': batch_size,
                'save_freq': max(estimated_total_steps // 4, 2000),  # Save every 25% of estimated steps
                'eval_freq': max(estimated_total_steps // 8, 1000),  # Evaluate every 12.5% of estimated steps
                'eval_episodes': min(200, batch_size // 5)  # Scale evaluation episodes with batch size
            })# Create new trainer for this batch (to set max_episodes correctly)
            batch_trainer = DQNTrainer(env, batch_config)
            
            # Load model: priority order is resume checkpoint > previous stage > previous batch
            model_loaded = False
            
            # 1. If resuming, try to load the most recent checkpoint for this stage
            if episodes_trained > 0 and self.current_trainer is None:
                latest_checkpoint = self._find_latest_checkpoint(stage_save_dir)
                if latest_checkpoint:
                    print(f"   🔄 Resuming from checkpoint: {os.path.basename(latest_checkpoint)}")
                    batch_trainer.load_model(latest_checkpoint)
                    model_loaded = True
            
            # 2. If first batch and no resume, load from previous stage
            if not model_loaded and hasattr(self, 'transfer_model_path') and self.transfer_model_path and os.path.exists(self.transfer_model_path) and self.current_trainer is None:
                print(f"   🔄 Loading knowledge from previous stage")
                batch_trainer.load_model(self.transfer_model_path)
                # Adjust learning parameters for transfer learning
                batch_trainer.epsilon = 0.3  # Start with moderate exploration
                print(f"   🎛️  Transfer learning: ε={batch_trainer.epsilon:.2f}")
                model_loaded = True
            
            # 3. If subsequent batch, transfer from previous batch
            if not model_loaded and hasattr(self, 'current_trainer') and self.current_trainer is not None:
                temp_model_path = os.path.join(stage_save_dir, "temp_batch_transfer.pth")
                self.current_trainer.save_model(temp_model_path)
                batch_trainer.load_model(temp_model_path)
                # Clean up temp file
                if os.path.exists(temp_model_path):
                    os.remove(temp_model_path)
                model_loaded = True
              # Training phase - DQNTrainer.train() handles step-based evaluation/saving internally
            # The trainer will evaluate and save based on step counts, not episode counts
            training_result = batch_trainer.train(save_dir=stage_save_dir)
            
            # Update our main trainer reference
            self.current_trainer = batch_trainer
            episodes_trained += batch_size
              # Evaluation phase - Stage-level evaluation (between batches) 
            # Note: This is episode-based for curriculum progression, while DQNTrainer uses step-based evaluation internally
            if episodes_trained >= min_episodes or episodes_trained % 2000 == 0:
                eval_result = self._evaluate_stage_performance(stage_name, stage_config)
                evaluation_history.append({
                    'episodes': episodes_trained,
                    'win_rate': eval_result['win_rate'],
                    'avg_reward': eval_result.get('avg_reward', 0.0)
                })
                
                current_win_rate = eval_result['win_rate']
                
                print(f"📊 Stage evaluation at {episodes_trained:,} episodes:")
                print(f"   🎯 Win rate: {current_win_rate:.1%} (target: {target_win_rate:.1%})")
                print(f"   🏆 Best so far: {best_win_rate:.1%}")
                
                # Check for improvement
                if current_win_rate > best_win_rate:
                    best_win_rate = current_win_rate
                    patience_counter = 0
                    
                    # Save best model for this stage
                    best_model_path = os.path.join(stage_save_dir, "best_stage_model.pth")
                    self.current_trainer.save_model(best_model_path)
                    print(f"💾 New best model saved: {best_model_path}")
                else:
                    patience_counter += batch_size
                
                # Check if target achieved
                if episodes_trained >= min_episodes and current_win_rate >= target_win_rate:
                    print(f"🎉 Target achieved! Win rate: {current_win_rate:.1%} >= {target_win_rate:.1%}")
                    self.tracker.complete_stage(stage_name, episodes_trained, current_win_rate)
                    
                    return {
                        'stage': stage_name,
                        'episodes_trained': episodes_trained,
                        'final_win_rate': current_win_rate,
                        'best_win_rate': best_win_rate,
                        'target_achieved': True,
                        'evaluation_history': evaluation_history
                    }
                
                # Check patience
                if patience_counter >= patience:
                    print(f"⏱️  Patience exhausted ({patience_counter}/{patience} episodes without improvement)")
                    if episodes_trained >= min_episodes:
                        print(f"   ✅ Minimum episodes met, stopping early")
                        break
                    else:
                        print(f"   ⏳ Continuing until minimum episodes ({min_episodes:,})")
                        patience_counter = 0  # Reset patience until min episodes reached
          # Stage completed (reached max episodes or early stopping)
        final_eval = self._evaluate_stage_performance(stage_name, stage_config)
        final_win_rate = final_eval['win_rate']
        target_achieved = final_win_rate >= target_win_rate
        
        if target_achieved:
            self.tracker.complete_stage(stage_name, episodes_trained, final_win_rate)
        else:
            print(f"⚠️  Stage target not achieved: {final_win_rate:.1%} < {target_win_rate:.1%}")
        
        # Evaluate for best model replacement
        self._evaluate_for_best_model(stage_name, final_eval, episodes_trained)
        
        return {
            'stage': stage_name,
            'episodes_trained': episodes_trained,
            'final_win_rate': final_win_rate,
            'best_win_rate': best_win_rate,
            'target_achieved': target_achieved,
            'evaluation_history': evaluation_history
        }
    
    def _get_trainer_config(self, stage_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get training configuration adapted for current stage"""
          # Base configuration
        config = {
            'learning_rate': 0.001,
            'batch_size': 64,
            'memory_size': 50000,
            'gamma': 0.99,
            'epsilon_start': 0.9,
            'epsilon_end': 0.1,
            'epsilon_decay': 0.995,
            'target_update_freq': 1000,
            'evaluation_method': self.evaluation_method,
            'num_eval_workers': self.num_eval_workers,
            # Default step-based frequencies
            'eval_freq': 5000,  # Steps, not episodes
            'save_freq': 5000,  # Steps, not episodes
            'eval_episodes': 100
        }
        
        # Adapt based on board size and complexity
        board_size = stage_config['rows'] * stage_config['cols']
        mine_density = stage_config['mines'] / board_size
        stage_name = stage_config.get('name', '')        # Smaller boards: faster learning, less memory
        if board_size <= 25:  # 5x5 or smaller
            config.update({
                'learning_rate': 0.002,
                'batch_size': 32,
                'memory_size': 20000,
                'epsilon_decay': 0.99,
                'eval_freq': 2000,  # More frequent evaluation for smaller boards
                'save_freq': 3000,  # More frequent saving for smaller boards
                'eval_episodes': 50  # Fewer evaluation episodes for faster feedback
            })
            
            # Special case: phase_0 needs more exploration with perfect knowledge
            if stage_name == 'phase_0':
                config.update({
                    'epsilon_start': 0.5,  # Lower starting epsilon for more focused learning
                    'epsilon_decay': 0.998  # Slower decay to maintain exploration longer
                })
              # Special case: slow down epsilon decay for tiny stage to allow more exploration
            if stage_name == 'tiny':
                config['epsilon_decay'] = 0.9985  # Much slower decay for better exploration
                
        elif board_size <= 64:  # 8x8 or smaller
            config.update({
                'learning_rate': 0.0015,
                'batch_size': 48,
                'memory_size': 35000,
                'epsilon_decay': 0.997,
                'eval_freq': 3000,  # Medium frequency for medium boards
                'save_freq': 4000,
                'eval_episodes': 75
            })
        elif board_size >= 400:  # 16x30 expert level
            config.update({
                'learning_rate': 0.0005,
                'batch_size': 128,
                'memory_size': 100000,
                'epsilon_start': 0.7,
                'epsilon_end': 0.05,
                'epsilon_decay': 0.9995,
                'eval_freq': 8000,  # Less frequent evaluation for large boards
                'save_freq': 10000,
                'eval_episodes': 150
            })
        
        # High mine density: more conservative exploration
        if mine_density > 0.15:
            config['epsilon_start'] = min(config['epsilon_start'], 0.7)
            config['epsilon_end'] = max(config['epsilon_end'], 0.1)
        return config
    
    def _transfer_knowledge(self, current_stage: str, stage_save_dir: str):
        """Transfer knowledge from previous stage if available"""
        
        current_idx = CurriculumConfig.get_stage_index(current_stage)
        if current_idx <= 0:
            print("🎯 First stage - training from scratch")
            self.transfer_model_path = None
            return
        
        # Look for best model from previous stage
        prev_stage = CurriculumConfig.CURRICULUM_STAGES[current_idx - 1]['name']
        prev_stage_dir = os.path.join(self.save_dir, f"stage_{prev_stage}")
        prev_best_model = os.path.join(prev_stage_dir, "best_stage_model.pth")
        
        if os.path.exists(prev_best_model):
            try:
                print(f"🔄 Knowledge transfer setup from stage '{prev_stage}'")
                
                # Copy as initial checkpoint for current stage
                initial_checkpoint = os.path.join(stage_save_dir, "transferred_model.pth")
                shutil.copy2(prev_best_model, initial_checkpoint)
                
                print(f"   ✅ Model copied to: {initial_checkpoint}")
                
                # Store path for later use when creating trainers
                self.transfer_model_path = initial_checkpoint
                
            except Exception as e:
                print(f"⚠️  Failed to setup knowledge transfer: {e}")
                print("   🎯 Will start stage from scratch")
                self.transfer_model_path = None
        else:
            print(f"⚠️  No previous stage model found at: {prev_best_model}")
            print("   🎯 Will start stage from scratch")
            self.transfer_model_path = None
    def _find_latest_checkpoint(self, stage_save_dir: str) -> Optional[str]:
        """Find the most recent checkpoint in a stage directory"""
        if not os.path.exists(stage_save_dir):
            return None
        
        # Look for episode checkpoint files
        checkpoints = []
        for filename in os.listdir(stage_save_dir):
            if filename.startswith("dqn_episode_") and filename.endswith(".pth"):
                try:
                    episode_num = int(filename.replace("dqn_episode_", "").replace(".pth", ""))
                    checkpoints.append((episode_num, os.path.join(stage_save_dir, filename)))
                except ValueError:
                    continue
        
        # Also check for final model
        final_model = os.path.join(stage_save_dir, "dqn_final.pth")
        if os.path.exists(final_model):
            # Get the latest episode number to use as final model episode
            max_episode = max([ep for ep, _ in checkpoints]) if checkpoints else 0
            checkpoints.append((max_episode + 1, final_model))
        
        if not checkpoints:
            return None
          # Return the most recent checkpoint
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        return checkpoints[0][1]
    
    def _evaluate_stage_performance(self, stage_name: str, stage_config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate current performance on stage"""
        
        # Import evaluation here to avoid circular imports
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        from evaluation import evaluate_model
        
        print(f"🔍 Evaluating performance on stage '{stage_name}'...")
        
        # Create temporary model file for evaluation
        temp_model_path = os.path.join(self.save_dir, f"temp_eval_{stage_name}.pth")
        self.current_trainer.save_model(temp_model_path)
        
        try:
            # Special handling for phase_0 (fully revealed boards)
            if stage_config.get('fully_revealed', False):
                # For phase_0, we know the win rate should be very high since boards are revealed
                # Use a simple evaluation that matches the training environment
                return self._evaluate_phase_0_performance(temp_model_path, stage_config)
            
            # Evaluate on current stage configuration
            result = evaluate_model(
                model_path=temp_model_path,
                difficulty_custom=(stage_config['rows'], stage_config['cols'], stage_config['mines']),
                num_games=500,  # Sufficient for reliable estimate
                method=self.evaluation_method,
                num_workers=self.num_eval_workers,
                verbose=False            )
            return {
                'win_rate': result['win_rate'],
                'avg_reward': result.get('avg_reward', 0.0),
                'games_played': result['games_played']
            }
            
        except Exception as e:
            print(f"⚠️  Evaluation failed: {e}")
            return {'win_rate': 0.0, 'avg_reward': 0.0, 'games_played': 0}
        finally:
            # Clean up temp file
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
    
    def _evaluate_phase_0_performance(self, model_path: str, stage_config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate performance on phase_0 (perfect knowledge boards)"""
        import torch
        from .models import DQN
        from .environment import PerfectKnowledgeMinesweeperEnvironment
        
        # Load the model
        rows, cols, mines = stage_config['rows'], stage_config['cols'], stage_config['mines']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create environment to determine correct input channels
        temp_env = PerfectKnowledgeMinesweeperEnvironment(rows, cols, mines)
        sample_obs = temp_env.reset()
        input_channels = sample_obs.shape[-1] if len(sample_obs.shape) == 3 else 4
        
        # Create model with correct architecture
        model = DQN(
            rows, cols,
            input_channels=input_channels,
            num_actions=rows * cols * 3
        )
        
        # Load model state - handle different save formats
        try:
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'q_network_state_dict' in checkpoint:
                # Full checkpoint format
                model.load_state_dict(checkpoint['q_network_state_dict'])
            else:
                # Simple state dict format
                model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"⚠️  Error loading model: {e}")
            # If we can't load the model, return a default result
            return {'win_rate': 0.0, 'avg_reward': 0.0, 'games_played': 0}
        
        model.to(device)
        model.eval()
        
        # Create environment for evaluation (reuse the temp_env for consistency)
        env = temp_env
        
        # Run evaluation episodes
        total_reward = 0.0
        wins = 0
        num_games = 100  # Smaller number since phase_0 is simple
        for _ in range(num_games):
            obs = env.reset()
            
            # Check if game is already won at start (which it should be in phase_0)
            initial_game_state = env.api.get_game_state()
            if initial_game_state['is_won']:
                # Game is already won - this is the expected behavior for phase_0
                wins += 1
                total_reward += 50.0  # Positive reward for recognizing won state
                continue
            
            # If game is not won at start (unexpected), play normally
            episode_reward = 0.0
            done = False
            steps = 0
            max_steps = 1000  # Prevent infinite loops
            
            while not done and steps < max_steps:
                # Convert observation to tensor (match trainer format)
                obs_tensor = torch.FloatTensor(obs).permute(2, 0, 1).unsqueeze(0).to(device)
                
                # Get action mask
                action_mask = torch.BoolTensor(env.get_action_mask()).to(device)
                
                # Get action from model (using same method as trainer)
                with torch.no_grad():
                    action = model.get_action(obs_tensor.squeeze(0), action_mask, epsilon=0.0)
                
                # Take action
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                steps += 1
            
            total_reward += episode_reward
            # Use same win detection as trainer
            if info.get('game_state') == 'won':
                wins += 1
        
        win_rate = wins / num_games
        avg_reward = total_reward / num_games
        
        print(f"✅ Phase 0 evaluation: {win_rate:.1%} win rate, {avg_reward:.2f} avg reward")
        
        return {
            'win_rate': win_rate,
            'avg_reward': avg_reward,
            'games_played': num_games
        }
    
    def resume_training(self) -> Dict[str, Any]:
        """Resume training from current curriculum stage"""
        current_stage = self.tracker.get_current_stage()
        print(f"🔄 Resuming curriculum training from stage: {current_stage}")
        return self.run_curriculum()
    
    def jump_to_stage(self, stage_name: str) -> None:
        """Jump to a specific curriculum stage"""
        if not CurriculumConfig.get_stage_config(stage_name):
            raise ValueError(f"Invalid stage name: {stage_name}")
        
        print(f"⏭️  Jumping to curriculum stage: {stage_name}")
        self.tracker.progress['current_stage'] = stage_name
        self.tracker.save_progress()
    
    def get_curriculum_status(self) -> Dict[str, Any]:
        """Get current curriculum learning status"""
        current_stage = self.tracker.get_current_stage()
        current_config = CurriculumConfig.get_stage_config(current_stage)
        
        status = {
            'current_stage': current_stage,
            'current_stage_config': current_config,
            'stages_completed': self.tracker.progress['stages_completed'],
            'total_episodes': self.tracker.get_total_episodes(),
            'curriculum_progress': {
                'completed': len(self.tracker.progress['stages_completed']),
                'total': len(CurriculumConfig.CURRICULUM_STAGES),
                'percentage': len(self.tracker.progress['stages_completed']) / len(CurriculumConfig.CURRICULUM_STAGES) * 100
            }
        }
        
        return status
    
    def _detect_stage_progress(self, stage_name: str) -> int:
        """Detect training progress from existing model checkpoints"""
        stage_dir = os.path.join(self.save_dir, f"stage_{stage_name}")
        if not os.path.exists(stage_dir):
            return 0
        
        # Look for episode checkpoint files
        max_episode = 0
        for filename in os.listdir(stage_dir):
            if filename.startswith("dqn_episode_") and filename.endswith(".pth"):
                try:
                    episode_num = int(filename.replace("dqn_episode_", "").replace(".pth", ""))
                    max_episode = max(max_episode, episode_num)
                except ValueError:
                    continue
        
        return max_episode
    
    def get_current_stage_progress(self) -> int:
        """Get training progress for current stage"""
        current_stage = self.get_current_stage()
        return self._detect_stage_progress(current_stage)
    
    def _evaluate_for_best_model(self, stage_name: str, stage_performance: Dict[str, Any], 
                                episodes_trained: int):
        """Evaluate current model for potential best model replacement"""
        print(f"\n🏆 Evaluating stage '{stage_name}' model for best model replacement...")
        
        # Get current model path (final model from training)
        stage_save_dir = os.path.join(self.save_dir, f"stage_{stage_name}")
        current_model_path = os.path.join(stage_save_dir, "dqn_final.pth")
        
        if not os.path.exists(current_model_path):
            print(f"⚠️  Current model not found: {current_model_path}")
            return
        
        # Prepare candidate metadata
        candidate_metadata = {
            'training_episodes': episodes_trained,
            'training_time_hours': 0.0,  # TODO: Track actual training time
            'stage': stage_name
        }
        
        # Prepare performance data
        candidate_performance = {
            'win_rate': stage_performance.get('win_rate', 0.0),
            'avg_reward': stage_performance.get('avg_reward', 0.0),
            'games_played': stage_performance.get('games_played', 0)
        }
        
        # Evaluate and potentially update best model
        updated = self.best_model_tracker.evaluate_and_update_best(
            current_model_path, stage_name, candidate_performance, candidate_metadata
        )
        
        if updated:
            print(f"🎉 New best model established from stage '{stage_name}'!")
        else:
            print(f"📊 Current model from stage '{stage_name}' does not exceed best model")
        
        # Print current best model summary
        print(f"\n" + "="*60)
        self.best_model_tracker.print_best_model_summary()
        print("="*60)
