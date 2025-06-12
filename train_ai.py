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
from datetime import datetime, timedelta
from src.ai.trainer import DQNTrainer
from src.ai.environment import MinesweeperEnvironment
from evaluation import (
    enable_optimized_parallel_evaluation,
    enable_lightweight_parallel_evaluation, 
    enable_sequential_evaluation_with_progress
)

class EnhancedBeginnerTrainerV2Resume:
    def __init__(self, num_eval_workers=None, evaluation_method="lightweight", target_win_rate=0.50, save_dir=None):
        self.target_win_rate = target_win_rate
        self.original_save_dir = "models_beginner_enhanced_v2"  # Original V2 results
        self.save_dir = save_dir or "models_beginner_enhanced_v2_parallel_resume"  # New parallel results
        self.num_eval_workers = num_eval_workers or max(1, os.cpu_count() - 2)
        self.evaluation_method = evaluation_method
        
        print(f"🔄 Enhanced Trainer V2 - RESUME from existing progress")
        print(f"   📂 Original results: {self.original_save_dir}")
        print(f"   💾 New parallel results: {self.save_dir}")
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
            }        ]
        
        os.makedirs(self.save_dir, exist_ok=True)
        
    def find_latest_checkpoint(self):
        """Find the latest checkpoint from both resume and original Enhanced V2 training"""
        
        latest_checkpoint_info = None
        
        # Phase priority: Mastery > Stabilization > Foundation
        phase_priority = {"Mastery": 3, "Stabilization": 2, "Foundation": 1}
        
        # Helper function to check a directory for checkpoints
        def check_directory_for_checkpoints(base_dir, source_type):
            nonlocal latest_checkpoint_info
            
            if not os.path.exists(base_dir):
                return
            
            print(f"🔍 Checking {source_type}: {base_dir}")
            
            for item in os.listdir(base_dir):
                item_path = os.path.join(base_dir, item)
                if not os.path.isdir(item_path):
                    continue
                
                # Parse phase and timestamp from directory name
                phase_name = None
                dir_timestamp = None
                
                if item.startswith("Foundation_"):
                    phase_name = "Foundation"
                    if "resumed_" in item:
                        timestamp_part = item.split("resumed_")[-1]
                        try:
                            dir_timestamp = datetime.strptime(timestamp_part, "%Y%m%d_%H%M%S")
                        except ValueError:
                            pass
                    elif source_type == "original":
                        # Original foundation - use modification time as fallback
                        dir_timestamp = datetime.fromtimestamp(os.path.getmtime(item_path))
                
                elif item.startswith("Stabilization_"):
                    phase_name = "Stabilization"
                    if "resumed_" in item:
                        timestamp_part = item.split("resumed_")[-1]
                        try:
                            dir_timestamp = datetime.strptime(timestamp_part, "%Y%m%d_%H%M%S")
                        except ValueError:
                            pass
                    elif source_type == "original":
                        # Original stabilization - use modification time as fallback
                        dir_timestamp = datetime.fromtimestamp(os.path.getmtime(item_path))
                
                elif item.startswith("Mastery_"):
                    phase_name = "Mastery"
                    if "resumed_" in item:
                        timestamp_part = item.split("resumed_")[-1]
                        try:
                            dir_timestamp = datetime.strptime(timestamp_part, "%Y%m%d_%H%M%S")
                        except ValueError:
                            pass
                    elif source_type == "original":
                        # Original mastery - use modification time as fallback
                        dir_timestamp = datetime.fromtimestamp(os.path.getmtime(item_path))
                
                if not phase_name:
                    continue
                
                # Find the latest checkpoint in this directory
                checkpoint_info = self._find_checkpoint_in_directory(item_path, phase_name, source_type)
                if not checkpoint_info:
                    continue
                
                # Use directory timestamp if available, otherwise checkpoint timestamp
                effective_timestamp = dir_timestamp or checkpoint_info.get('timestamp')
                
                print(f"   📁 {phase_name} ({source_type}): Episode {checkpoint_info['episodes_completed']:,}")
                if effective_timestamp:
                    print(f"      ⏰ {effective_timestamp}")
                
                # Check if this is the latest based on phase priority and episode number
                is_better = False
                if latest_checkpoint_info is None:
                    is_better = True
                else:
                    current_phase_priority = phase_priority.get(phase_name, 0)
                    latest_phase_priority = phase_priority.get(latest_checkpoint_info['phase'], 0)
                    
                    if current_phase_priority > latest_phase_priority:
                        # Higher priority phase (e.g., Stabilization > Foundation)
                        is_better = True
                    elif current_phase_priority == latest_phase_priority:
                        # Same phase - compare episode numbers
                        if checkpoint_info['episodes_completed'] > latest_checkpoint_info['episodes_completed']:
                            is_better = True
                        elif (checkpoint_info['episodes_completed'] == latest_checkpoint_info['episodes_completed'] 
                              and effective_timestamp and latest_checkpoint_info.get('timestamp')
                              and effective_timestamp > latest_checkpoint_info['timestamp']):
                            # Same episode - use timestamp as tiebreaker
                            is_better = True
                
                if is_better:
                    latest_checkpoint_info = checkpoint_info
                    print(f"      🌟 LATEST so far")
        
        # Check parallel resume directory first (more recent)
        check_directory_for_checkpoints(self.save_dir, "parallel resume")
        
        # Check original directory
        check_directory_for_checkpoints(self.original_save_dir, "original")
        
        if latest_checkpoint_info:
            source = "parallel resume" if self.save_dir in latest_checkpoint_info['phase_dir'] else "original"
            print(f"\n✅ Using LATEST checkpoint from {source}")
            print(f"   📂 Phase: {latest_checkpoint_info['phase']}")
            print(f"   📊 Episode: {latest_checkpoint_info['episodes_completed']:,}")
            print(f"   💾 Model: {latest_checkpoint_info['checkpoint_path']}")
            if latest_checkpoint_info.get('timestamp'):
                print(f"   ⏰ Timestamp: {latest_checkpoint_info['timestamp']}")
        else:
            print("\n❌ No existing checkpoints found in either directory")
        
        return latest_checkpoint_info
    
    def _find_checkpoint_in_directory(self, phase_dir, phase_name, source_type):
        """Find the latest checkpoint file in a specific phase directory"""
        
        # Look for checkpoint files
        checkpoint_files = []
        if os.path.exists(phase_dir):
            for f in os.listdir(phase_dir):
                if f.startswith("dqn_episode_") and f.endswith(".pth"):
                    checkpoint_files.append(f)
        
        # If we have episode checkpoints, use the latest one
        if checkpoint_files:
            episodes = [int(f.split('_episode_')[1].split('.pth')[0]) for f in checkpoint_files]
            latest_episode = max(episodes)
            latest_checkpoint = os.path.join(phase_dir, f"dqn_episode_{latest_episode}.pth")
            
            return {
                'phase': phase_name,
                'checkpoint_path': latest_checkpoint,
                'metrics_path': os.path.join(phase_dir, "training_metrics.json"),
                'episodes_completed': latest_episode,
                'phase_dir': phase_dir,
                'timestamp': datetime.fromtimestamp(os.path.getmtime(latest_checkpoint))
            }
        
        # Check for final model (completed phase)
        final_model = os.path.join(phase_dir, "dqn_final.pth")
        if os.path.exists(final_model):
            # Determine episodes based on phase
            episodes_map = {
                'Foundation': 15000,
                'Stabilization': 15000,
                'Mastery': 15000
            }
            
            return {
                'phase': phase_name,
                'checkpoint_path': final_model,
                'metrics_path': os.path.join(phase_dir, "training_metrics.json"),
                'episodes_completed': episodes_map.get(phase_name, 15000),
                'phase_dir': phase_dir,
                'timestamp': datetime.fromtimestamp(os.path.getmtime(final_model))
            }
        
        return None
    
    def load_checkpoint_state(self, checkpoint_info):
        """Load the trainer state from checkpoint"""
        
        checkpoint_path = checkpoint_info['checkpoint_path']
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract training state
        training_state = {
            'episode': checkpoint.get('episode', 0),
            'epsilon': checkpoint.get('epsilon', 0.15),  # Default for stabilization
            'total_steps': checkpoint.get('total_steps', 0)
        }
        
        print(f"   📊 Episode: {training_state['episode']}")
        print(f"   🎲 Epsilon: {training_state['epsilon']:.3f}")
        print(f"   📈 Total steps: {training_state['total_steps']:,}")
        
        return checkpoint, training_state
    
    def train_remaining_phases(self, resume_info):
        """Continue training from where we left off"""
        
        phase_name = resume_info['phase']
        episodes_completed = resume_info['episodes_completed']
        
        print(f"🔄 Resuming from {phase_name} phase")
        print(f"   ✅ Episodes completed: {episodes_completed:,}")
        
        # Load checkpoint state
        checkpoint, training_state = self.load_checkpoint_state(resume_info)
        
        all_results = []
        best_overall_win_rate = 0.0
        best_model_path = None
        start_time = datetime.now()
        
        # Determine which phases to run
        phase_start_index = 0
        remaining_episodes = 0
        
        if phase_name == "Foundation":
            # Start from Stabilization
            phase_start_index = 1
            print("   ➡️  Starting Stabilization phase")
        elif phase_name == "Stabilization":
            # Continue Stabilization, then do Mastery
            phase_start_index = 1
            remaining_episodes = 15000 - episodes_completed  # How many left in Stabilization
            print(f"   ➡️  Continuing Stabilization ({remaining_episodes:,} episodes remaining)")
        else:
            print(f"❌ Unknown phase: {phase_name}")
            return None
        
        # Process remaining phases
        for i in range(phase_start_index, len(self.training_phases)):
            phase = self.training_phases[i].copy()
            
            # Adjust episodes for partially completed phase
            if i == phase_start_index and remaining_episodes > 0:
                phase['episodes'] = remaining_episodes
                print(f"   📝 Adjusted {phase['name']} to {remaining_episodes:,} remaining episodes")
            
            try:
                # Create trainer for this phase
                env = MinesweeperEnvironment(rows=9, cols=9, mines=10)
                
                trainer_config = {
                    'learning_rate': phase['learning_rate'],
                    'batch_size': phase['batch_size'],
                    'memory_size': phase['memory_size'],
                    'epsilon_start': training_state['epsilon'],  # Continue from current epsilon
                    'epsilon_end': phase['epsilon_end'],
                    'epsilon_decay': 0.995,
                    'target_update_freq': 1000,
                    'max_episodes': phase['episodes'],
                    'eval_freq': phase['eval_frequency'],
                    'eval_episodes': phase['eval_episodes'],                    'save_freq': 500
                }
                
                trainer = DQNTrainer(env, trainer_config)
                
                # Enable appropriate evaluation method
                if self.evaluation_method == "optimized":
                    trainer = enable_optimized_parallel_evaluation(trainer, num_workers=self.num_eval_workers)
                elif self.evaluation_method == "lightweight":
                    trainer = enable_lightweight_parallel_evaluation(trainer, num_threads=self.num_eval_workers)
                elif self.evaluation_method == "sequential":
                    trainer = enable_sequential_evaluation_with_progress(trainer)
                else:
                    print(f"⚠️ Unknown evaluation method: {self.evaluation_method}, using lightweight")
                    trainer = enable_lightweight_parallel_evaluation(trainer, num_threads=self.num_eval_workers)
                
                # Load the model weights
                trainer.q_network.load_state_dict(checkpoint['q_network_state_dict'])
                trainer.target_network.load_state_dict(checkpoint['target_network_state_dict'])
                trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # Set training state
                trainer.episode = training_state['episode']
                trainer.epsilon = training_state['epsilon']
                trainer.total_steps = training_state['total_steps']
                
                print(f"\n🚀 Starting {phase['name']} phase with parallel evaluation")
                print(f"   📋 {phase['description']}")
                print(f"   🎯 Episodes: {phase['episodes']:,}")
                print(f"   📚 Learning Rate: {phase['learning_rate']}")
                print(f"   🎲 Epsilon: {trainer.epsilon:.3f} → {phase['epsilon_end']}")
                print(f"   📦 Batch Size: {phase['batch_size']}")
                print(f"   🖥️  Eval Workers: {self.num_eval_workers}")
                
                # Create phase directory
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                phase_dir = os.path.join(self.save_dir, f"{phase['name']}_resumed_{timestamp}")
                os.makedirs(phase_dir, exist_ok=True)
                
                # Train
                phase_start_time = datetime.now()
                results = trainer.train(save_dir=phase_dir)
                phase_end_time = datetime.now()
                
                training_time = phase_end_time - phase_start_time
                print(f"   ⏱️  Phase training time: {training_time}")
                
                # Save model
                model_path = os.path.join(phase_dir, "model.pth")
                torch.save(trainer.q_network.state_dict(), model_path)
                
                # Results tracking
                phase_results = {
                    'phase_name': phase['name'],
                    'resumed_from_episode': training_state['episode'],
                    'episodes_trained': phase['episodes'],
                    'timestamp': timestamp,
                    'training_time': str(training_time),
                    'best_win_rate': float(max(results['eval_win_rates']) if results['eval_win_rates'] else 0.0),
                    'final_win_rate': float(results['eval_win_rates'][-1] if results['eval_win_rates'] else 0.0),
                    'best_eval_reward': float(max(results['eval_rewards']) if results['eval_rewards'] else 0.0),
                    'final_eval_reward': float(results['eval_rewards'][-1] if results['eval_rewards'] else 0.0),
                    'model_path': model_path,
                    'training_config': phase,
                    'parallel_eval_workers': self.num_eval_workers
                }
                
                all_results.append(phase_results)
                
                # Track best performance
                if phase_results['best_win_rate'] > best_overall_win_rate:
                    best_overall_win_rate = phase_results['best_win_rate']
                    best_model_path = model_path
                    print(f"   🌟 NEW BEST: {best_overall_win_rate:.3f} ({best_overall_win_rate*100:.1f}%)")
                
                print(f"   ✅ {phase['name']} completed!")
                print(f"   📊 Best win rate: {phase_results['best_win_rate']:.3f} ({phase_results['best_win_rate']*100:.1f}%)")
                print(f"   📊 Final win rate: {phase_results['final_win_rate']:.3f} ({phase_results['final_win_rate']*100:.1f}%)")
                
                # Check if target achieved
                if best_overall_win_rate >= self.target_win_rate:
                    print(f"\n🎉 TARGET ACHIEVED! {best_overall_win_rate*100:.1f}% >= {self.target_win_rate*100:.1f}%")
                    break
                
                # Update checkpoint for next phase
                checkpoint = {
                    'q_network_state_dict': trainer.q_network.state_dict(),
                    'target_network_state_dict': trainer.target_network.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'episode': trainer.episode,
                    'epsilon': trainer.epsilon,
                    'total_steps': trainer.total_steps
                }
                training_state = {
                    'episode': trainer.episode,
                    'epsilon': phase['epsilon_end'],  # Start next phase at this epsilon
                    'total_steps': trainer.total_steps
                }
                remaining_episodes = 0  # Reset for next phase
                
            except Exception as e:
                print(f"\n❌ Error in {phase['name']} phase: {e}")
                import traceback
                traceback.print_exc()
                continue
          # Save final results
        end_time = datetime.now()
        total_duration = end_time - start_time
        
        # Convert datetime objects to strings for JSON serialization
        def make_json_safe(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, timedelta):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: make_json_safe(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_safe(item) for item in obj]
            else:
                return obj
        
        final_results = {
            'version': 'v2_parallel_resume',
            'resumed_from': make_json_safe(resume_info),
            'training_completed': end_time.isoformat(),
            'training_duration': str(total_duration),
            'target_win_rate': float(self.target_win_rate),
            'best_win_rate': float(best_overall_win_rate),
            'best_model_path': str(best_model_path) if best_model_path else None,
            'target_achieved': bool(best_overall_win_rate >= self.target_win_rate),
            'phases_completed': int(len(all_results)),
            'eval_workers_used': self.num_eval_workers,
            'all_phase_results': make_json_safe(all_results)
        }
        
        results_path = os.path.join(self.save_dir, "resume_training_results.json")
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Print summary
        print(f"\n🎯 ENHANCED TRAINING V2 RESUME SUMMARY")
        print("=" * 75)
        print(f"⏱️  Total Duration: {total_duration}")
        print(f"🔄 Resumed from: {resume_info['phase']} Episode {resume_info['episodes_completed']:,}")
        print(f"🏆 Best Achieved: {best_overall_win_rate:.3f} ({best_overall_win_rate*100:.1f}%)")
        print(f"✅ Target Achieved: {'YES' if best_overall_win_rate >= self.target_win_rate else 'NO'}")
        print(f"📊 Additional Phases: {len(all_results)}")
        print(f"🖥️  Evaluation Workers: {self.num_eval_workers}")
        print(f"💾 Results: {results_path}")
        
        return final_results
    
    def run_resume_training(self):
        """Resume training from existing Enhanced V2 progress"""
        
        print("🔄 ENHANCED BEGINNER TRAINING V2 - RESUME MODE")
        print("=" * 75)
        print(f"Looking for existing Enhanced V2 progress...")
        
        # Find existing checkpoint
        resume_info = self.find_latest_checkpoint()
        
        if not resume_info:
            print("❌ No existing progress found. Please run the original Enhanced V2 training first.")
            return None
        
        print(f"✅ Found checkpoint in {resume_info['phase']} phase")
        print(f"   Episodes completed: {resume_info['episodes_completed']:,}")
        
        # Continue training with parallel evaluation
        results = self.train_remaining_phases(resume_info)
        
        if results and results['target_achieved']:
            print(f"\n🎉 SUCCESS! Achieved {results['best_win_rate']*100:.1f}% win rate!")
            print(f"🏆 Best model: {results['best_model_path']}")
        else:
            win_rate = results['best_win_rate'] if results else 0.0
            print(f"\n📈 Progress: {win_rate*100:.1f}% (target: {self.target_win_rate*100:.1f}%)")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Minesweeper AI Training - Primary Entry Point')
    
    # Training mode selection
    parser.add_argument('--mode', choices=['resume', 'new', 'benchmark'], default='resume',
                       help='Training mode: resume from checkpoint, start new training, or benchmark evaluation methods')
    
    # Evaluation settings
    parser.add_argument('--workers', type=int, default=None, 
                       help='Number of evaluation workers (default: auto-detected based on method)')
    parser.add_argument('--eval-method', choices=['optimized', 'lightweight', 'sequential'], 
                       default='lightweight', help='Evaluation method to use')
    
    # Training settings
    parser.add_argument('--difficulty', choices=['beginner', 'intermediate', 'expert'], 
                       default='beginner', help='Game difficulty level')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Number of episodes to train (default: depends on mode)')
    parser.add_argument('--target-win-rate', type=float, default=0.50,
                       help='Target win rate to achieve (default: 0.50)')
    
    # Output and debugging
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Directory to save models and results')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    print("🤖 MINESWEEPER AI TRAINING")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Difficulty: {args.difficulty}")
    print(f"Evaluation: {args.eval_method}")
    if args.eval_method != "sequential":
        print(f"Workers: {args.workers or 'auto'}")
    print()
    
    # Route to appropriate training mode
    if args.mode == 'resume':
        run_resume_training(args)
    elif args.mode == 'new':
        run_new_training(args)
    elif args.mode == 'benchmark':
        run_benchmark(args)


def run_resume_training(args):
    """Resume training from existing Enhanced V2 progress"""
    # Setup evaluation workers
    cpu_cores = os.cpu_count()
    
    if args.workers:
        eval_workers = args.workers
        print(f"🖥️  System: {cpu_cores} CPU cores detected")
        print(f"🚀 Using {eval_workers} workers (user specified)")
    else:
        # Default based on evaluation method
        if args.eval_method == "sequential":
            eval_workers = 1
        elif args.eval_method == "lightweight":
            eval_workers = min(8, max(2, cpu_cores // 2))  # Conservative for threading
        else:  # optimized
            eval_workers = max(1, cpu_cores - 2)
        
        print(f"🖥️  System: {cpu_cores} CPU cores detected")
        print(f"🚀 Using {eval_workers} workers (auto for {args.eval_method})")
    
    # Validate worker count
    if args.eval_method != "sequential":
        if eval_workers > cpu_cores:
            print(f"⚠️  Warning: {eval_workers} workers > {cpu_cores} CPU cores")
            print(f"   This may cause oversubscription and reduced performance")
        elif eval_workers > cpu_cores * 0.8:
            print(f"💡 Using {eval_workers}/{cpu_cores} cores (high utilization)")
    
    trainer = EnhancedBeginnerTrainerV2Resume(
        num_eval_workers=eval_workers,
        evaluation_method=args.eval_method,
        target_win_rate=args.target_win_rate,
        save_dir=args.save_dir
    )
    results = trainer.run_resume_training()
    
    if results:
        if results['target_achieved']:
            print(f"\n🎯 TARGET ACHIEVED with {args.eval_method} evaluation!")
        else:
            print(f"\n📊 Training resumed with {args.eval_method} evaluation")
        
        # Performance summary
        if 'all_phase_results' in results:
            total_episodes = sum(r.get('episodes_trained', 0) for r in results['all_phase_results'])
            print(f"📈 Additional episodes trained: {total_episodes:,}")
    else:
        print(f"\n❌ Resume training failed")


def run_new_training(args):
    """Start new training from scratch"""
    print("🆕 Starting new training from scratch...")
    
    # Create trainer
    from src.ai.trainer import create_trainer
    
    trainer = create_trainer(
        difficulty=args.difficulty,
        config={
            'max_episodes': args.episodes or 10000,
            'eval_episodes': 100,
            'eval_freq': 100
        }
    )
    
    # Setup evaluation method
    cpu_cores = os.cpu_count()
    if args.workers:
        eval_workers = args.workers
    else:
        if args.eval_method == "sequential":
            eval_workers = 1
        elif args.eval_method == "lightweight":
            eval_workers = min(8, max(2, cpu_cores // 2))
        else:  # optimized
            eval_workers = max(1, cpu_cores - 2)
    
    # Enable evaluation method
    if args.eval_method == "optimized":
        trainer = enable_optimized_parallel_evaluation(trainer, num_workers=eval_workers)
    elif args.eval_method == "lightweight":
        trainer = enable_lightweight_parallel_evaluation(trainer, num_threads=eval_workers)
    elif args.eval_method == "sequential":
        trainer = enable_sequential_evaluation_with_progress(trainer)
    
    # Set save directory
    save_dir = args.save_dir or f"models_{args.difficulty}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"💾 Saving to: {save_dir}")
    print(f"🎯 Target episodes: {trainer.config['max_episodes']:,}")
    print(f"📊 Evaluation method: {args.eval_method}")
    if args.eval_method != "sequential":
        print(f"🖥️  Workers: {eval_workers}")
    
    # Train
    results = trainer.train(save_dir=save_dir)
    
    # Report results
    if results['eval_win_rates']:
        best_win_rate = max(results['eval_win_rates'])
        final_win_rate = results['eval_win_rates'][-1]
        
        print(f"\n🏆 TRAINING COMPLETED")
        print(f"Best win rate: {best_win_rate:.3f} ({best_win_rate*100:.1f}%)")
        print(f"Final win rate: {final_win_rate:.3f} ({final_win_rate*100:.1f}%)")
        print(f"Target achieved: {'YES' if best_win_rate >= args.target_win_rate else 'NO'}")
    
    return results


def run_benchmark(args):
    """Benchmark different evaluation methods"""
    print("🏁 Benchmarking evaluation methods...")
    
    # Create a basic trainer for benchmarking
    from src.ai.trainer import create_trainer
    
    trainer = create_trainer(
        difficulty=args.difficulty,
        config={'eval_episodes': 50}  # Use fewer episodes for benchmarking
    )
    
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

if __name__ == "__main__":
    main()
