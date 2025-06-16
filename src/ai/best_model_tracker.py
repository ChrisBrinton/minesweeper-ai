"""
Best Model Tracker for Minesweeper AI
Tracks and maintains the best performing model across all training sessions
"""

import os
import json
import shutil
import torch
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict


@dataclass
class ModelMetadata:
    """Metadata for tracking model performance"""
    model_path: str
    timestamp: str
    highest_stage: str
    stage_performances: Dict[str, Dict[str, float]]  # stage_name -> {win_rate, avg_reward, games_played}
    training_episodes: int
    training_time_hours: float
    model_size_mb: float
    notes: str = ""


class BestModelTracker:
    """Tracks the best performing model across training sessions"""
    
    def __init__(self, best_models_dir: str = "models/best"):
        self.best_models_dir = best_models_dir
        self.metadata_file = os.path.join(best_models_dir, "best_model_metadata.json")
        self.model_file = os.path.join(best_models_dir, "best_model.pth")
        
        # Create directory if it doesn't exist
        os.makedirs(best_models_dir, exist_ok=True)
        
        # Load existing metadata
        self.current_best = self._load_metadata()
    
    def _load_metadata(self) -> Optional[ModelMetadata]:
        """Load existing best model metadata"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                return ModelMetadata(**data)
            except Exception as e:
                print(f"⚠️  Warning: Could not load best model metadata: {e}")
        return None
    
    def _save_metadata(self, metadata: ModelMetadata):
        """Save metadata to file"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(asdict(metadata), f, indent=2)
        except Exception as e:
            print(f"⚠️  Warning: Could not save best model metadata: {e}")
    
    def _get_model_size_mb(self, model_path: str) -> float:
        """Get model file size in MB"""
        try:
            size_bytes = os.path.getsize(model_path)
            return size_bytes / (1024 * 1024)
        except:
            return 0.0
    
    def _stage_priority(self, stage_name: str) -> int:
        """Get priority/order of curriculum stages"""
        stage_order = {
            'phase_0': 0,
            'tiny': 1,
            'small': 2,
            'medium': 3,
            'intermediate': 4,
            'large': 5,
            'expert': 6
        }
        return stage_order.get(stage_name, 999)
    
    def is_current_model_better(self, candidate_stage: str, candidate_performance: Dict[str, float], 
                               candidate_metadata: Dict[str, Any]) -> bool:
        """
        Determine if current model is better than the best model
        
        Args:
            candidate_stage: Current stage being evaluated
            candidate_performance: Performance dict with win_rate, avg_reward, games_played
            candidate_metadata: Additional metadata about the candidate model
            
        Returns:
            True if candidate should replace best model
        """
        if self.current_best is None:
            print("🆕 No existing best model - current model will become the best")
            return True
        
        # Get stage priorities
        candidate_priority = self._stage_priority(candidate_stage)
        best_priority = self._stage_priority(self.current_best.highest_stage)
        
        print(f"📊 Comparing models:")
        print(f"   Current: {candidate_stage} (priority {candidate_priority}) - Win rate: {candidate_performance['win_rate']:.1%}")
        print(f"   Best: {self.current_best.highest_stage} (priority {best_priority}) - Win rate: {self.current_best.stage_performances.get(self.current_best.highest_stage, {}).get('win_rate', 0):.1%}")
        
        # If best model has achieved higher stages, candidate needs to be significantly better
        if best_priority > candidate_priority:
            print("🏆 Best model has achieved higher stages - keeping best model")
            return False
        
        # If same stage, compare performance
        if best_priority == candidate_priority:
            best_performance = self.current_best.stage_performances.get(candidate_stage, {})
            best_win_rate = best_performance.get('win_rate', 0.0)
            candidate_win_rate = candidate_performance['win_rate']
            
            # Require at least 5% improvement to replace
            improvement_threshold = 0.05
            improvement = candidate_win_rate - best_win_rate
            
            if improvement > improvement_threshold:
                print(f"📈 Significant improvement: {improvement:.1%} > {improvement_threshold:.1%}")
                return True
            else:
                print(f"📉 Insufficient improvement: {improvement:.1%} <= {improvement_threshold:.1%}")
                return False
        
        # If candidate achieved higher stage, it's better
        if candidate_priority > best_priority:
            print("🚀 Current model achieved higher stage!")
            return True
        
        return False
    
    def evaluate_and_update_best(self, candidate_model_path: str, candidate_stage: str, 
                                candidate_performance: Dict[str, float], 
                                candidate_metadata: Dict[str, Any]) -> bool:
        """
        Evaluate candidate model and potentially update best model
        
        Args:
            candidate_model_path: Path to the candidate model
            candidate_stage: Stage the candidate was trained on
            candidate_performance: Initial performance metrics
            candidate_metadata: Additional metadata
            
        Returns:
            True if best model was updated
        """
        print(f"\n🔍 Evaluating candidate model for best model replacement...")
        print(f"   Candidate: {candidate_model_path}")
        print(f"   Stage: {candidate_stage}")
        print(f"   Performance: {candidate_performance}")
        
        # First check if it's potentially better
        if not self.is_current_model_better(candidate_stage, candidate_performance, candidate_metadata):
            return False
        
        # Perform thorough evaluation with 1000 games
        print(f"🎯 Performing thorough evaluation with 1000 games...")
        
        try:
            # Import evaluation here to avoid circular imports
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
            from evaluation import evaluate_model
            
            # Get stage configuration for evaluation
            from .curriculum import CurriculumConfig
            stage_config = CurriculumConfig.get_stage_config(candidate_stage)
            
            # Evaluate with 1000 games
            eval_result = evaluate_model(
                model_path=candidate_model_path,
                difficulty_custom=(stage_config['rows'], stage_config['cols'], stage_config['mines']),
                num_games=1000,
                method='optimized',
                verbose=False
            )
            
            thorough_performance = {
                'win_rate': eval_result['win_rate'],
                'avg_reward': eval_result.get('avg_reward', 0.0),
                'games_played': eval_result['games_played']
            }
            
            print(f"📊 Thorough evaluation results:")
            print(f"   Win rate: {thorough_performance['win_rate']:.1%}")
            print(f"   Avg reward: {thorough_performance['avg_reward']:.2f}")
            print(f"   Games played: {thorough_performance['games_played']}")
            
            # Final check with thorough performance
            if self.is_current_model_better(candidate_stage, thorough_performance, candidate_metadata):
                # Update best model
                return self._update_best_model(candidate_model_path, candidate_stage, 
                                             thorough_performance, candidate_metadata)
            else:
                print("❌ Model did not maintain performance in thorough evaluation")
                return False
                
        except Exception as e:
            print(f"⚠️  Error during thorough evaluation: {e}")
            return False
    
    def _update_best_model(self, model_path: str, stage: str, performance: Dict[str, float], 
                          metadata: Dict[str, Any]) -> bool:
        """Update the best model files and metadata"""
        try:
            print(f"🏆 Updating best model...")
            
            # Copy model file
            shutil.copy2(model_path, self.model_file)
            
            # Create new metadata
            new_metadata = ModelMetadata(
                model_path=self.model_file,
                timestamp=datetime.now().isoformat(),
                highest_stage=stage,
                stage_performances={stage: performance},
                training_episodes=metadata.get('training_episodes', 0),
                training_time_hours=metadata.get('training_time_hours', 0.0),
                model_size_mb=self._get_model_size_mb(self.model_file),
                notes=f"Updated from {model_path} on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            # If we have existing metadata, preserve other stage performances
            if self.current_best:
                for stage_name, stage_perf in self.current_best.stage_performances.items():
                    if stage_name not in new_metadata.stage_performances:
                        new_metadata.stage_performances[stage_name] = stage_perf
                
                # Update highest stage if current is higher
                if self._stage_priority(stage) > self._stage_priority(self.current_best.highest_stage):
                    new_metadata.highest_stage = stage
                else:
                    new_metadata.highest_stage = self.current_best.highest_stage
            
            # Save metadata
            self._save_metadata(new_metadata)
            self.current_best = new_metadata
            
            print(f"✅ Best model updated successfully!")
            print(f"   New best stage: {new_metadata.highest_stage}")
            print(f"   Win rate: {performance['win_rate']:.1%}")
            print(f"   Model size: {new_metadata.model_size_mb:.1f} MB")
            
            return True
            
        except Exception as e:
            print(f"❌ Error updating best model: {e}")
            return False
    
    def get_best_model_info(self) -> Optional[ModelMetadata]:
        """Get information about the current best model"""
        return self.current_best
    
    def print_best_model_summary(self):
        """Print a summary of the best model"""
        if self.current_best is None:
            print("📋 No best model found")
            return
        
        print(f"\n🏆 Best Model Summary:")
        print(f"   📁 Path: {self.current_best.model_path}")
        print(f"   📅 Updated: {self.current_best.timestamp}")
        print(f"   🎯 Highest Stage: {self.current_best.highest_stage}")
        print(f"   📊 Training Episodes: {self.current_best.training_episodes:,}")
        print(f"   ⏱️  Training Time: {self.current_best.training_time_hours:.1f} hours")
        print(f"   💾 Model Size: {self.current_best.model_size_mb:.1f} MB")
        
        print(f"   🎮 Stage Performances:")
        for stage, perf in self.current_best.stage_performances.items():
            print(f"      {stage}: {perf['win_rate']:.1%} win rate, {perf['avg_reward']:.1f} avg reward")
        
        if self.current_best.notes:
            print(f"   📝 Notes: {self.current_best.notes}")
