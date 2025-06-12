#!/usr/bin/env python3
"""
Minesweeper AI Training - Primary Entry Point
Provides comprehensive training capabilities with multiple evaluation methods
"""

import os
import json
import torch
import numpy as np
import argparse
import traceback
from datetime import datetime, timedelta
from src.ai.trainer import DQNTrainer, create_trainer
from src.ai.environment import MinesweeperEnvironment
from src.ai.model_storage import get_model_save_dir, find_latest_model_dir, get_latest_checkpoint
from evaluation import (
    enable_optimized_parallel_evaluation,
    enable_lightweight_parallel_evaluation, 
    enable_sequential_evaluation_with_progress
)

class EnhancedBeginnerTrainerV2Resume:
    def __init__(self, difficulty="beginner", num_eval_workers=None, evaluation_method="lightweight", target_win_rate=0.50, save_dir=None):
        self.difficulty = difficulty.lower()
        self.target_win_rate = target_win_rate
        
        # Use new organized directory structure
        if save_dir:
            self.save_dir = save_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_dir = get_model_save_dir(self.difficulty, timestamp)
        
        # Legacy directories for backward compatibility
        self.legacy_dirs = [
            "models_beginner_enhanced_v2",
            "models_beginner_enhanced_v2_parallel_resume",
            f"models_{self.difficulty}_enhanced_v2",
            f"models_{self.difficulty}_enhanced_v2_parallel_resume"
        ]
        
        self.num_eval_workers = num_eval_workers or max(1, os.cpu_count() - 2)
        self.evaluation_method = evaluation_method
        
        print(f"🔄 Enhanced Trainer V2 - RESUME from existing progress")
        print(f"   🎮 Difficulty: {self.difficulty}")
        print(f"   💾 New results: {self.save_dir}")
        print(f"   🖥️  Evaluation method: {evaluation_method}")
        if evaluation_method != "sequential":
            print(f"   🖥️  Evaluation workers: {self.num_eval_workers}")
        print(f"   🎯 Target: {self.target_win_rate*100:.0f}% win rate")
        
        # Training phases configuration
        self.training_phases = [
            {
                'name': 'Foundation',
                'episodes': 15000,
                'description': 'Extended foundation learning with stable parameters',
                'learning_rate': 0.001,
                'epsilon_start': 0.9,
                'epsilon_end': 0.3,
                'batch_size': 64,
                'memory_size': 50000,
                'eval_frequency': 100,
                'eval_episodes': 100
            },
            {
                'name': 'Stabilization', 
                'episodes': 15000,
                'description': 'Gradual parameter adjustment with knowledge preservation',
                'learning_rate': 0.0008,
                'epsilon_start': 0.3,
                'epsilon_end': 0.15,
                'batch_size': 96,
                'memory_size': 75000,
                'eval_frequency': 100,
                'eval_episodes': 100
            },
            {
                'name': 'Mastery',
                'episodes': 15000,
                'description': 'Fine-tuning with preserved knowledge',
                'learning_rate': 0.0005,
                'epsilon_start': 0.15,
                'epsilon_end': 0.05,
                'batch_size': 128,
                'memory_size': 100000,            
                'eval_frequency': 100,
                'eval_episodes': 100
            }
        ]
        
        # Ensure save directory exists
        os.makedirs(self.save_dir, exist_ok=True)
        
    def find_latest_checkpoint(self):
        """Find the latest checkpoint from organized models and legacy directories"""
        
        latest_checkpoint_info = None
        
        # Phase priority: Mastery > Stabilization > Foundation
        phase_priority = {"Mastery": 3, "Stabilization": 2, "Foundation": 1}
          # Check organized model directories first
        try:
            organized_checkpoint = get_latest_checkpoint(self.difficulty)
            if organized_checkpoint:
                checkpoint_dir, checkpoint_file = organized_checkpoint
                # Try to determine phase from checkpoint metadata
                current_phase = 'Foundation'  # Default phase
                priority = phase_priority.get(current_phase, 1)
                
                try:
                    metadata_path = os.path.join(checkpoint_dir, 'training_metadata.json')
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            current_phase = metadata.get('current_phase', 'Foundation')
                            priority = phase_priority.get(current_phase, 1)
                except Exception as e:
                    print(f"⚠️  Could not read metadata from {checkpoint_dir}: {e}")
                
                latest_checkpoint_info = {
                    'path': os.path.join(checkpoint_dir, checkpoint_file),
                    'directory': checkpoint_dir,
                    'phase': current_phase,
                    'priority': priority,
                    'source': 'organized'
                }
        except Exception as e:
            print(f"⚠️  Error checking organized directories: {e}")
        
        # Check legacy directories for backward compatibility
        for legacy_dir in self.legacy_dirs:
            if not os.path.exists(legacy_dir):
                continue
                
            try:
                # Look for best model checkpoint
                best_checkpoint = os.path.join(legacy_dir, 'best_model_checkpoint.pth')
                if os.path.exists(best_checkpoint):
                    # Default to Foundation phase for legacy models
                    priority = phase_priority.get("Foundation", 1)
                    if not latest_checkpoint_info or priority >= latest_checkpoint_info['priority']:
                        latest_checkpoint_info = {
                            'path': best_checkpoint,
                            'directory': legacy_dir,
                            'phase': 'Foundation',
                            'priority': priority,
                            'source': 'legacy'
                        }
                        
                # Look for phase-specific checkpoints
                for phase_name, phase_priority_val in phase_priority.items():
                    phase_checkpoint = os.path.join(legacy_dir, f'{phase_name.lower()}_checkpoint.pth')
                    if os.path.exists(phase_checkpoint):
                        if not latest_checkpoint_info or phase_priority_val > latest_checkpoint_info['priority']:
                            latest_checkpoint_info = {
                                'path': phase_checkpoint,
                                'directory': legacy_dir,
                                'phase': phase_name,
                                'priority': phase_priority_val,
                                'source': 'legacy'
                            }
            except Exception as e:
                print(f"⚠️  Error checking legacy directory {legacy_dir}: {e}")
        return latest_checkpoint_info
    
    def load_training_state(self, checkpoint_info):
        """Load training state from checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_info['path'], map_location='cpu')
            
            # Determine starting phase based on checkpoint
            current_phase_name = checkpoint_info['phase']
            phase_index = next((i for i, p in enumerate(self.training_phases) if p['name'] == current_phase_name), 0)
            
            # Load training progress
            total_episodes = checkpoint.get('total_episodes', 0)
            phase_episodes = checkpoint.get('phase_episodes', 0)
            
            print(f"📂 Loading from {checkpoint_info['source']} checkpoint:")
            print(f"   📍 Phase: {current_phase_name}")
            print(f"   📊 Total episodes: {total_episodes:,}")
            print(f"   📊 Phase episodes: {phase_episodes:,}")
            return {
                'checkpoint': checkpoint,
                'phase_index': phase_index,
                'total_episodes': total_episodes,
                'phase_episodes': phase_episodes,
                'source_dir': checkpoint_info.get('directory', os.path.dirname(checkpoint_info['path']))
            }
        except Exception as e:
            print(f"❌ Error loading checkpoint {checkpoint_info['path']}: {e}")
            return None
    def run(self):
        """Main training execution"""
        # Find and load existing checkpoint
        checkpoint_info = self.find_latest_checkpoint()
        
        if checkpoint_info:
            training_state = self.load_training_state(checkpoint_info)
            if training_state:
                return self._resume_training(training_state)
        
        print("🆕 No checkpoint found - starting fresh training")
        return self._start_fresh_training()
    
    def _resume_training(self, training_state):
        """Resume training from checkpoint"""
        checkpoint = training_state['checkpoint']
        phase_index = training_state['phase_index']
        total_episodes = training_state['total_episodes']
        phase_episodes = training_state['phase_episodes']
        source_dir = training_state['source_dir']
          # Initialize trainer
        trainer = create_trainer(difficulty=self.difficulty)
        
        # Configure evaluation method
        if self.evaluation_method == "optimized":
            enable_optimized_parallel_evaluation(trainer, self.num_eval_workers)
        elif self.evaluation_method == "lightweight":
            enable_lightweight_parallel_evaluation(trainer, self.num_eval_workers)
        else:
            enable_sequential_evaluation_with_progress(trainer)
        
        # Load model state
        trainer.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        trainer.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Resume training from current phase
        for i in range(phase_index, len(self.training_phases)):
            phase = self.training_phases[i]
            
            # Calculate remaining episodes for current phase
            if i == phase_index:
                remaining_episodes = phase['episodes'] - phase_episodes
                current_phase_episodes = phase_episodes
            else:
                remaining_episodes = phase['episodes']
                current_phase_episodes = 0
            
            if remaining_episodes <= 0:
                print(f"✅ Phase {phase['name']} already complete, skipping")
                continue
            
            print(f"\n🚀 Resuming Phase {i+1}/{len(self.training_phases)}: {phase['name']}")
            print(f"   📝 {phase['description']}")
            print(f"   📊 Episodes remaining: {remaining_episodes:,}/{phase['episodes']:,}")
            
            # Update trainer parameters for this phase
            trainer.learning_rate = phase['learning_rate']
            trainer.epsilon = phase['epsilon_start'] if current_phase_episodes == 0 else trainer.epsilon
            trainer.epsilon_min = phase['epsilon_end']
            trainer.batch_size = phase['batch_size']
            trainer.memory.maxlen = phase['memory_size']
            
            # Run phase training
            success = self._run_phase_training(
                trainer, phase, remaining_episodes, 
                total_episodes, current_phase_episodes, i
            )
            
            if not success:
                print(f"❌ Training failed during {phase['name']} phase")
                return False
            total_episodes += remaining_episodes
        
        print(f"\n🎉 All training phases completed!")
        print(f"   📊 Total episodes: {total_episodes:,}")
        return True
    
    def _start_fresh_training(self):
        """Start training from scratch"""        # Initialize trainer
        trainer = create_trainer(difficulty=self.difficulty)
        
        # Configure evaluation method
        if self.evaluation_method == "optimized":
            enable_optimized_parallel_evaluation(trainer, self.num_eval_workers)
        elif self.evaluation_method == "lightweight":
            enable_lightweight_parallel_evaluation(trainer, self.num_eval_workers)
        else:
            enable_sequential_evaluation_with_progress(trainer)
        
        total_episodes = 0
        
        # Run all training phases
        for i, phase in enumerate(self.training_phases):
            print(f"\n🚀 Starting Phase {i+1}/{len(self.training_phases)}: {phase['name']}")
            print(f"   📝 {phase['description']}")
            print(f"   📊 Episodes: {phase['episodes']:,}")
            
            # Update trainer parameters for this phase
            trainer.learning_rate = phase['learning_rate']
            trainer.epsilon = phase['epsilon_start']
            trainer.epsilon_min = phase['epsilon_end']
            trainer.batch_size = phase['batch_size']
            trainer.memory.maxlen = phase['memory_size']
            
            # Run phase training
            success = self._run_phase_training(
                trainer, phase, phase['episodes'], 
                total_episodes, 0, i
            )
            
            if not success:
                print(f"❌ Training failed during {phase['name']} phase")
                return False
            
            total_episodes += phase['episodes']
        
        print(f"\n🎉 All training phases completed!")
        print(f"   📊 Total episodes: {total_episodes:,}")
        return True
    
    def _run_phase_training(self, trainer, phase, episodes_to_run, total_episodes_so_far, phase_episodes_so_far, phase_index):
        """Run training for a specific phase"""
        try:
            phase_name = phase['name']
            eval_frequency = phase['eval_frequency']
            eval_episodes = phase['eval_episodes']
            
            episode_count = phase_episodes_so_far
            best_win_rate = 0.0
            
            print(f"⏳ Training {phase_name} phase...")
            
            for episode in range(episodes_to_run):                # Train one episode
                trainer._train_episode()
                episode_count += 1
                
                # Evaluation checkpoint
                if episode_count % eval_frequency == 0:
                    print(f"\n📊 Evaluation at episode {episode_count} (Phase: {phase_name})")
                      # Run evaluation
                    win_rate, avg_score = trainer._evaluate(num_episodes=eval_episodes)
                    
                    print(f"   🏆 Win rate: {win_rate:.1%}")
                    print(f"   📈 Avg score: {avg_score:.1f}")
                    print(f"   🧠 Epsilon: {trainer.epsilon:.3f}")
                    
                    # Save checkpoint if improved
                    if win_rate > best_win_rate:
                        best_win_rate = win_rate
                        self._save_checkpoint(trainer, phase_name, episode_count, total_episodes_so_far + episode, win_rate, phase_index)
                        print(f"   💾 New best model saved! ({win_rate:.1%})")
                    
                    # Check if target reached
                    if win_rate >= self.target_win_rate:
                        print(f"🎯 Target win rate achieved: {win_rate:.1%} >= {self.target_win_rate:.1%}")
                        self._save_final_model(trainer, phase_name, episode_count, total_episodes_so_far + episode, win_rate)
                        return True
            
            # Save phase completion checkpoint
            self._save_checkpoint(trainer, phase_name, episode_count, total_episodes_so_far + episodes_to_run, best_win_rate, phase_index, is_phase_complete=True)
            print(f"✅ {phase_name} phase completed")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during {phase['name']} training: {e}")
            traceback.print_exc()
            return False
    
    def _save_checkpoint(self, trainer, phase_name, phase_episodes, total_episodes, win_rate, phase_index, is_phase_complete=False):
        """Save training checkpoint with organized structure"""
        try:
            # Create checkpoint data
            checkpoint = {
                'q_network_state_dict': trainer.q_network.state_dict(),
                'target_network_state_dict': trainer.target_network.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'total_episodes': total_episodes,
                'phase_episodes': phase_episodes,
                'current_phase': phase_name,
                'phase_index': phase_index,
                'win_rate': win_rate,
                'epsilon': trainer.epsilon,
                'timestamp': datetime.now().isoformat(),
                'is_phase_complete': is_phase_complete
            }
            
            # Save main checkpoint
            checkpoint_filename = f'{phase_name.lower()}_checkpoint.pth'
            checkpoint_path = os.path.join(self.save_dir, checkpoint_filename)
            torch.save(checkpoint, checkpoint_path)
            
            # Save as best model if this is the best so far
            best_path = os.path.join(self.save_dir, 'best_model_checkpoint.pth')
            if not os.path.exists(best_path) or win_rate > self._get_best_win_rate():
                torch.save(checkpoint, best_path)
            
            # Save training metadata
            metadata = {
                'difficulty': self.difficulty,
                'current_phase': phase_name,
                'phase_index': phase_index,
                'total_episodes': total_episodes,
                'phase_episodes': phase_episodes,
                'win_rate': win_rate,
                'target_win_rate': self.target_win_rate,
                'evaluation_method': self.evaluation_method,
                'timestamp': datetime.now().isoformat(),
                'save_directory': self.save_dir
            }
            
            metadata_path = os.path.join(self.save_dir, 'training_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
        except Exception as e:
            print(f"⚠️  Error saving checkpoint: {e}")
    
    def _save_final_model(self, trainer, phase_name, phase_episodes, total_episodes, win_rate):
        """Save final successful model"""
        try:
            final_model = {
                'q_network_state_dict': trainer.q_network.state_dict(),
                'target_network_state_dict': trainer.target_network.state_dict(),
                'total_episodes': total_episodes,
                'phase_episodes': phase_episodes,
                'final_phase': phase_name,
                'final_win_rate': win_rate,
                'target_achieved': True,
                'timestamp': datetime.now().isoformat()
            }
            
            final_path = os.path.join(self.save_dir, 'final_model.pth')
            torch.save(final_model, final_path)
            
            print(f"🏆 Final model saved: {final_path}")
            
        except Exception as e:
            print(f"⚠️  Error saving final model: {e}")
    
    def _get_best_win_rate(self):
        """Get the best win rate from existing checkpoints"""
        try:
            best_path = os.path.join(self.save_dir, 'best_model_checkpoint.pth')
            if os.path.exists(best_path):
                checkpoint = torch.load(best_path, map_location='cpu')
                return checkpoint.get('win_rate', 0.0)
        except:
            pass
        return 0.0


def create_new_training(args):
    """Create a new training session"""
    print("🆕 Starting new training session...")
    
    trainer = EnhancedBeginnerTrainerV2Resume(
        difficulty=args.difficulty,
        num_eval_workers=args.workers,
        evaluation_method=args.eval_method,
        target_win_rate=getattr(args, 'target_win_rate', 0.50)
    )
    
    return trainer.run()


def resume_training(args):
    """Resume existing training"""
    print("🔄 Resuming existing training...")
    
    # Allow user to specify a specific save directory to resume from
    save_dir = getattr(args, 'resume_dir', None)
    
    trainer = EnhancedBeginnerTrainerV2Resume(
        difficulty=args.difficulty,
        num_eval_workers=args.workers,
        evaluation_method=args.eval_method,
        target_win_rate=getattr(args, 'target_win_rate', 0.50),
        save_dir=save_dir
    )
    
    return trainer.run()


def run_benchmark(args):
    """Run evaluation method benchmark"""
    print("🏃 Running evaluation method benchmark...")
    
    # Create a temporary trainer for benchmarking
    trainer = create_trainer(difficulty=args.difficulty)
    
    # Train a bit first so we have a reasonable model to evaluate
    print("🔧 Training model briefly for benchmark...")
    trainer.train_episode_count = 0  # Reset episode count
    for _ in range(100):  # Quick training
        trainer._train_episode()
        trainer.train_episode_count += 1
    
    # Run benchmark
    from evaluation import benchmark_evaluation_methods
    results = benchmark_evaluation_methods(trainer, num_episodes=args.episodes or 50)
    
    return results


def main():
    """Main entry point with comprehensive CLI"""
    parser = argparse.ArgumentParser(
        description="Minesweeper AI Training System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_ai.py --mode resume --difficulty beginner
  python train_ai.py --mode new --difficulty intermediate --eval-method optimized
  python train_ai.py --mode benchmark --episodes 100 --workers 8
        """
    )
    
    # Core arguments
    parser.add_argument('--mode', 
                       choices=['resume', 'new', 'benchmark'],
                       default='resume',
                       help='Training mode (default: resume)')
    
    parser.add_argument('--difficulty',
                       choices=['beginner', 'intermediate', 'expert'],
                       default='beginner',
                       help='Game difficulty (default: beginner)')
    
    parser.add_argument('--eval-method',
                       choices=['sequential', 'lightweight', 'optimized'],
                       default='lightweight',
                       help='Evaluation method (default: lightweight)')
    
    parser.add_argument('--workers',
                       type=int,
                       default=None,
                       help='Number of evaluation workers (default: auto)')
    
    parser.add_argument('--episodes',
                       type=int,
                       help='Number of episodes for benchmark mode')
    
    parser.add_argument('--target-win-rate',
                       type=float,
                       default=0.50,
                       help='Target win rate (default: 0.50)')
    
    parser.add_argument('--resume-dir',
                       type=str,
                       help='Specific directory to resume training from')
    
    args = parser.parse_args()
    
    # Print configuration
    print("🤖 Minesweeper AI Training System")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Difficulty: {args.difficulty}")
    print(f"Evaluation method: {args.eval_method}")
    if args.workers:
        print(f"Workers: {args.workers}")
    if args.episodes:
        print(f"Episodes: {args.episodes}")
    print("=" * 50)
    
    try:
        # Execute based on mode
        if args.mode == "resume":
            success = resume_training(args)
        elif args.mode == "new":
            success = create_new_training(args)
        elif args.mode == "benchmark":
            success = run_benchmark(args)
        else:
            print(f"❌ Unknown mode: {args.mode}")
            return 1
        
        if success:
            print("\n✅ Training completed successfully!")
            return 0
        else:
            print("\n❌ Training failed!")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
