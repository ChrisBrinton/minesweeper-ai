"""
Tests for the main training entry point (train_ai.py)
Tests the EnhancedBeginnerTrainerV2Resume class and CLI functionality
"""

import pytest
import os
import json
import torch
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Import the training system
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_ai import (
    EnhancedBeginnerTrainerV2Resume,
    create_new_training,
    resume_training,
    run_benchmark,
    main
)
from src.ai.trainer import create_trainer


class TestEnhancedBeginnerTrainerV2Resume:
    """Test the main training class"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_trainer(self):
        """Create a mock trainer for testing"""
        trainer = Mock()
        trainer.q_network = Mock()
        trainer.target_network = Mock()        
        trainer.optimizer = Mock()
        trainer.epsilon = 0.5
        trainer.learning_rate = 0.001
        trainer.batch_size = 64
        trainer.memory = Mock()
        trainer.memory.maxlen = 50000
        trainer.config = {'target_update_freq': 100}  # Add config for target network updates
          # Mock state_dict methods
        trainer.q_network.state_dict.return_value = {"dummy": "q_network_state"}
        trainer.target_network.state_dict.return_value = {"dummy": "target_network_state"}
        trainer.target_network.load_state_dict = Mock()  # Add for target network updates
        trainer.optimizer.state_dict.return_value = {"dummy": "optimizer_state"}
        
        # Mock training methods
        trainer._train_episode.return_value = (-10.0, 50, False)
        trainer._evaluate.return_value = (-15.5, 0.0)  # avg_reward, win_rate
        
        return trainer
    
    def test_trainer_initialization(self, temp_dir):
        """Test basic trainer initialization"""
        trainer = EnhancedBeginnerTrainerV2Resume(
            difficulty="beginner",
            num_eval_workers=2,
            evaluation_method="sequential",
            target_win_rate=0.6,
            save_dir=temp_dir
        )
        
        assert trainer.difficulty == "beginner"
        assert trainer.num_eval_workers == 2
        assert trainer.evaluation_method == "sequential"
        assert trainer.target_win_rate == 0.6
        assert trainer.save_dir == temp_dir
        assert len(trainer.training_phases) == 3
        assert os.path.exists(temp_dir)
    
    def test_training_phases_configuration(self):
        """Test that training phases are properly configured"""
        trainer = EnhancedBeginnerTrainerV2Resume()
        
        phases = trainer.training_phases
        assert len(phases) == 3
        
        # Check phase names
        phase_names = [phase['name'] for phase in phases]
        assert phase_names == ['Foundation', 'Stabilization', 'Mastery']
          # Check phase parameters
        foundation = phases[0]
        assert foundation['episodes'] == 15000
        assert foundation['learning_rate'] == 0.001
        assert foundation['epsilon_start'] == 0.9
        assert foundation['epsilon_end'] == 0.1
        assert foundation['batch_size'] == 64
        assert foundation['memory_size'] == 50000
        assert foundation['eval_frequency'] == 100
        assert foundation['eval_episodes'] == 100
        assert foundation['expected_win_rate'] == 0.15
        assert foundation['min_win_rate'] == 0.05
    
    def test_legacy_directories_configuration(self):
        """Test that legacy directories are properly configured"""
        trainer = EnhancedBeginnerTrainerV2Resume(difficulty="intermediate")
        
        expected_legacy_dirs = [
            "models_beginner_enhanced_v2",
            "models_beginner_enhanced_v2_parallel_resume",
            "models_intermediate_enhanced_v2",
            "models_intermediate_enhanced_v2_parallel_resume"
        ]
        
        assert trainer.legacy_dirs == expected_legacy_dirs
    
    def test_auto_worker_count(self):
        """Test automatic worker count configuration"""
        trainer = EnhancedBeginnerTrainerV2Resume()
        expected_workers = max(1, os.cpu_count() - 2)
        assert trainer.num_eval_workers == expected_workers
    
    def test_find_latest_checkpoint_no_checkpoints(self, temp_dir):
        """Test checkpoint finding when no checkpoints exist"""
        trainer = EnhancedBeginnerTrainerV2Resume(save_dir=temp_dir)
        checkpoint_info = trainer.find_latest_checkpoint()
        assert checkpoint_info is None
    
    def test_find_latest_checkpoint_with_organized_checkpoint(self, temp_dir):
        """Test finding checkpoint from organized directory structure"""
        # Create organized checkpoint structure
        checkpoint_dir = os.path.join(temp_dir, "models", "beginner", "20250612_120000")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create checkpoint file
        checkpoint_file = os.path.join(checkpoint_dir, "foundation_checkpoint.pth")
        torch.save({"dummy": "checkpoint"}, checkpoint_file)
        
        # Create metadata file
        metadata = {
            "current_phase": "Foundation",
            "total_episodes": 1000,
            "phase_episodes": 500        }
        metadata_file = os.path.join(checkpoint_dir, "training_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        with patch('train_ai.get_latest_checkpoint') as mock_get_latest:
            mock_get_latest.return_value = (checkpoint_dir, "foundation_checkpoint.pth")
            
            trainer = EnhancedBeginnerTrainerV2Resume(save_dir=temp_dir)
            checkpoint_info = trainer.find_latest_checkpoint()
            
            assert checkpoint_info is not None
            assert checkpoint_info['phase'] == 'Foundation'
            assert checkpoint_info['source'] == 'organized'
            assert checkpoint_info['priority'] == 1  # Foundation priority
    
    def test_find_latest_checkpoint_with_legacy_checkpoint(self, temp_dir):
        """Test finding checkpoint from legacy directory"""
        # Create legacy directory
        legacy_dir = "models_beginner_enhanced_v2"
        os.makedirs(legacy_dir, exist_ok=True)
        
        # Create legacy checkpoint
        checkpoint_file = os.path.join(legacy_dir, "best_model_checkpoint.pth")
        torch.save({"dummy": "legacy_checkpoint"}, checkpoint_file)
        
        try:
            trainer = EnhancedBeginnerTrainerV2Resume(save_dir=temp_dir)
            checkpoint_info = trainer.find_latest_checkpoint()
            
            assert checkpoint_info is not None
            assert checkpoint_info['phase'] == 'Foundation'
            assert checkpoint_info['source'] == 'legacy'
            assert checkpoint_info['directory'] == legacy_dir
        finally:
            # Cleanup legacy directory
            shutil.rmtree(legacy_dir, ignore_errors=True)
    
    def test_load_training_state(self, temp_dir):
        """Test loading training state from checkpoint"""
        # Create checkpoint
        checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pth")
        checkpoint_data = {
            'q_network_state_dict': {"dummy": "q_state"},
            'target_network_state_dict': {"dummy": "target_state"},
            'optimizer_state_dict': {"dummy": "optimizer_state"},
            'total_episodes': 1500,
            'phase_episodes': 750,
            'epsilon': 0.4
        }
        torch.save(checkpoint_data, checkpoint_path)
        
        checkpoint_info = {
            'path': checkpoint_path,
            'phase': 'Stabilization',
            'source': 'test'
        }
        
        trainer = EnhancedBeginnerTrainerV2Resume(save_dir=temp_dir)
        training_state = trainer.load_training_state(checkpoint_info)
        
        assert training_state is not None
        assert training_state['phase_index'] == 1  # Stabilization is index 1
        assert training_state['total_episodes'] == 1500
        assert training_state['phase_episodes'] == 750
        assert training_state['checkpoint']['epsilon'] == 0.4
    
    def test_load_training_state_invalid_checkpoint(self, temp_dir):
        """Test loading training state with invalid checkpoint"""
        checkpoint_info = {
            'path': os.path.join(temp_dir, "nonexistent.pth"),
            'phase': 'Foundation',
            'source': 'test'
        }
        
        trainer = EnhancedBeginnerTrainerV2Resume(save_dir=temp_dir)
        training_state = trainer.load_training_state(checkpoint_info)
        
        assert training_state is None
    
    @patch('train_ai.create_trainer')
    @patch('train_ai.enable_sequential_evaluation_with_progress')
    def test_run_no_checkpoint(self, mock_eval, mock_create_trainer, temp_dir, mock_trainer):
        """Test running training when no checkpoint exists"""
        mock_create_trainer.return_value = mock_trainer
        
        trainer = EnhancedBeginnerTrainerV2Resume(
            save_dir=temp_dir,
            evaluation_method="sequential"
        )
        
        # Mock the phase training to return quickly
        with patch.object(trainer, '_run_phase_training', return_value=True):
            result = trainer.run()
        
        assert result is True
        mock_create_trainer.assert_called_once_with(difficulty='beginner')
        mock_eval.assert_called_once_with(mock_trainer)
    
    @patch('train_ai.create_trainer')
    @patch('train_ai.enable_lightweight_parallel_evaluation')
    def test_run_with_lightweight_evaluation(self, mock_eval, mock_create_trainer, temp_dir, mock_trainer):
        """Test running with lightweight evaluation method"""
        mock_create_trainer.return_value = mock_trainer
        
        trainer = EnhancedBeginnerTrainerV2Resume(
            save_dir=temp_dir,
            evaluation_method="lightweight",
            num_eval_workers=4
        )
        
        with patch.object(trainer, '_run_phase_training', return_value=True):
            result = trainer.run()
        
        assert result is True
        mock_eval.assert_called_once_with(mock_trainer, 4)
    
    @patch('train_ai.create_trainer')
    @patch('train_ai.enable_optimized_parallel_evaluation')
    def test_run_with_optimized_evaluation(self, mock_eval, mock_create_trainer, temp_dir, mock_trainer):
        """Test running with optimized evaluation method"""
        mock_create_trainer.return_value = mock_trainer
        
        trainer = EnhancedBeginnerTrainerV2Resume(
            save_dir=temp_dir,
            evaluation_method="optimized",
            num_eval_workers=2
        )
        
        with patch.object(trainer, '_run_phase_training', return_value=True):
            result = trainer.run()
        
        assert result is True
        mock_eval.assert_called_once_with(mock_trainer, 2)
    
    def test_save_checkpoint(self, temp_dir, mock_trainer):
        """Test saving training checkpoint"""
        trainer = EnhancedBeginnerTrainerV2Resume(save_dir=temp_dir)
        
        trainer._save_checkpoint(
            mock_trainer, 
            "Foundation", 
            phase_episodes=500, 
            total_episodes=500, 
            win_rate=0.15, 
            phase_index=0
        )
        
        # Check that checkpoint file was created
        checkpoint_path = os.path.join(temp_dir, "foundation_checkpoint.pth")
        assert os.path.exists(checkpoint_path)
        
        # Check that best model was created
        best_path = os.path.join(temp_dir, "best_model_checkpoint.pth")
        assert os.path.exists(best_path)
        
        # Check that metadata was created
        metadata_path = os.path.join(temp_dir, "training_metadata.json")
        assert os.path.exists(metadata_path)
        
        # Verify checkpoint contents
        checkpoint = torch.load(checkpoint_path)
        assert checkpoint['total_episodes'] == 500
        assert checkpoint['phase_episodes'] == 500
        assert checkpoint['current_phase'] == "Foundation"
        assert checkpoint['win_rate'] == 0.15
        assert checkpoint['phase_index'] == 0
        
        # Verify metadata contents
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        assert metadata['current_phase'] == "Foundation"
        assert metadata['total_episodes'] == 500
        assert metadata['win_rate'] == 0.15
    
    def test_save_final_model(self, temp_dir, mock_trainer):
        """Test saving final successful model"""
        trainer = EnhancedBeginnerTrainerV2Resume(save_dir=temp_dir)
        
        trainer._save_final_model(
            mock_trainer,
            "Mastery",
            phase_episodes=1000,
            total_episodes=45000,
            win_rate=0.65
        )
        
        final_path = os.path.join(temp_dir, "final_model.pth")
        assert os.path.exists(final_path)
        
        # Verify final model contents
        final_model = torch.load(final_path)
        assert final_model['total_episodes'] == 45000
        assert final_model['phase_episodes'] == 1000
        assert final_model['final_phase'] == "Mastery"
        assert final_model['final_win_rate'] == 0.65
        assert final_model['target_achieved'] is True
    
    def test_get_best_win_rate_no_checkpoint(self, temp_dir):
        """Test getting best win rate when no checkpoint exists"""
        trainer = EnhancedBeginnerTrainerV2Resume(save_dir=temp_dir)
        best_win_rate = trainer._get_best_win_rate()
        assert best_win_rate == 0.0
    
    def test_get_best_win_rate_with_checkpoint(self, temp_dir):
        """Test getting best win rate from existing checkpoint"""
        trainer = EnhancedBeginnerTrainerV2Resume(save_dir=temp_dir)
        
        # Create best model checkpoint
        best_path = os.path.join(temp_dir, "best_model_checkpoint.pth")
        checkpoint = {'win_rate': 0.42}
        torch.save(checkpoint, best_path)
        
        best_win_rate = trainer._get_best_win_rate()
        assert best_win_rate == 0.42
    
    def test_run_phase_training_success(self, temp_dir, mock_trainer):
        """Test successful phase training execution"""
        trainer = EnhancedBeginnerTrainerV2Resume(save_dir=temp_dir)
        phase = {
            'name': 'Foundation',
            'episodes': 10,  # Small number for testing
            'eval_frequency': 5,
            'eval_episodes': 10,
            'expected_win_rate': 0.15,
            'min_win_rate': 0.05
        }
        
        # Mock evaluation to return improving win rates
        mock_trainer._evaluate.side_effect = [(-20.0, 0.1), (-15.0, 0.2)]
        
        with patch.object(trainer, '_save_checkpoint') as mock_save:
            result = trainer._run_phase_training(
                mock_trainer, phase, episodes_to_run=10,
                total_episodes_so_far=0, phase_episodes_so_far=0, phase_index=0
            )
        
        assert result is True
        assert mock_trainer._train_episode.call_count == 10
        assert mock_trainer._evaluate.call_count == 2  # Every 5 episodes
        assert mock_save.call_count >= 2  # At least called for improvements
    
    def test_run_phase_training_target_reached(self, temp_dir, mock_trainer):
        """Test phase training when target win rate is reached"""
        trainer = EnhancedBeginnerTrainerV2Resume(
            save_dir=temp_dir,
            target_win_rate=0.3
        )
        phase = {
            'name': 'Foundation',
            'episodes': 20,
            'eval_frequency': 5,
            'eval_episodes': 10,
            'expected_win_rate': 0.15,
            'min_win_rate': 0.05
        }
        
        # Mock evaluation to reach target
        mock_trainer._evaluate.return_value = (-10.0, 0.35)  # Above target
        
        with patch.object(trainer, '_save_checkpoint'):
            with patch.object(trainer, '_save_final_model') as mock_save_final:
                result = trainer._run_phase_training(
                    mock_trainer, phase, episodes_to_run=20,
                    total_episodes_so_far=0, phase_episodes_so_far=0, phase_index=0
                )
        
        assert result is True
        mock_save_final.assert_called_once()
        # Should stop early when target is reached
        assert mock_trainer._train_episode.call_count == 5


class TestCLIFunctions:
    """Test the CLI helper functions"""
    
    @pytest.fixture
    def mock_args(self):
        """Create mock command line arguments"""
        args = Mock()
        args.difficulty = "beginner"
        args.workers = 4
        args.eval_method = "lightweight"
        args.target_win_rate = 0.5
        args.resume_dir = None
        args.episodes = 100
        return args
    
    @patch('train_ai.EnhancedBeginnerTrainerV2Resume')
    def test_create_new_training(self, mock_trainer_class, mock_args):
        """Test create_new_training function"""
        mock_trainer = Mock()
        mock_trainer.run.return_value = True
        mock_trainer_class.return_value = mock_trainer
        result = create_new_training(mock_args)
        
        assert result is True
        mock_trainer_class.assert_called_once_with(
            difficulty="beginner",
            num_eval_workers=4,
            evaluation_method="lightweight",
            target_win_rate=0.5,
            total_episodes=100
        )
        mock_trainer.run.assert_called_once()
    
    @patch('train_ai.EnhancedBeginnerTrainerV2Resume')
    def test_resume_training(self, mock_trainer_class, mock_args):
        """Test resume_training function"""
        mock_trainer = Mock()
        mock_trainer.run.return_value = True
        mock_trainer_class.return_value = mock_trainer
        result = resume_training(mock_args)
        
        assert result is True
        mock_trainer_class.assert_called_once_with(
            difficulty="beginner",
            num_eval_workers=4,
            evaluation_method="lightweight",
            target_win_rate=0.5,
            save_dir=None,
            total_episodes=100
        )
        mock_trainer.run.assert_called_once()
    
    @patch('train_ai.create_trainer')
    @patch('evaluation.benchmark_evaluation_methods')
    def test_run_benchmark(self, mock_benchmark, mock_create_trainer, mock_args):
        """Test run_benchmark function"""
        mock_trainer = Mock()
        mock_trainer.train_episode_count = 0
        mock_trainer._train_episode = Mock()
        mock_create_trainer.return_value = mock_trainer
        mock_benchmark.return_value = {"test": "results"}
        
        result = run_benchmark(mock_args)
        
        assert result == {"test": "results"}
        mock_create_trainer.assert_called_once_with(difficulty="beginner")
        assert mock_trainer._train_episode.call_count == 100  # Brief training
        mock_benchmark.assert_called_once_with(mock_trainer, num_episodes=100)
    
    def test_main_function_help(self):
        """Test main function argument parsing"""
        # Test that main function can parse arguments
        with patch('sys.argv', ['train_ai.py', '--help']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            # Help should exit with code 0
            assert exc_info.value.code == 0
    
    @patch('train_ai.resume_training')
    def test_main_function_resume_mode(self, mock_resume):
        """Test main function with resume mode"""
        mock_resume.return_value = True
        
        with patch('sys.argv', ['train_ai.py', '--mode', 'resume']):
            result = main()
        
        assert result == 0
        mock_resume.assert_called_once()
    
    @patch('train_ai.create_new_training')
    def test_main_function_new_mode(self, mock_new):
        """Test main function with new mode"""
        mock_new.return_value = True
        
        with patch('sys.argv', ['train_ai.py', '--mode', 'new']):
            result = main()
        
        assert result == 0
        mock_new.assert_called_once()
    
    @patch('train_ai.run_benchmark')
    def test_main_function_benchmark_mode(self, mock_benchmark):
        """Test main function with benchmark mode"""
        mock_benchmark.return_value = True
        
        with patch('sys.argv', ['train_ai.py', '--mode', 'benchmark']):
            result = main()
        
        assert result == 0
        mock_benchmark.assert_called_once()
    
    def test_main_function_unknown_mode(self):
        """Test main function with unknown mode"""
        # Patch argv to include invalid mode - this should be caught by argparse
        with patch('sys.argv', ['train_ai.py', '--mode', 'invalid']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            # Invalid argument should exit with code 2
            assert exc_info.value.code == 2
    
    @patch('train_ai.resume_training')
    def test_main_function_failure(self, mock_resume):
        """Test main function when training fails"""
        mock_resume.return_value = False
        
        with patch('sys.argv', ['train_ai.py', '--mode', 'resume']):
            result = main()
        
        assert result == 1
    
    @patch('train_ai.resume_training')
    def test_main_function_keyboard_interrupt(self, mock_resume):
        """Test main function with keyboard interrupt"""
        mock_resume.side_effect = KeyboardInterrupt()
        
        with patch('sys.argv', ['train_ai.py', '--mode', 'resume']):
            result = main()
        
        assert result == 1
    
    @patch('train_ai.resume_training')
    def test_main_function_unexpected_error(self, mock_resume):
        """Test main function with unexpected error"""
        mock_resume.side_effect = Exception("Test error")
        
        with patch('sys.argv', ['train_ai.py', '--mode', 'resume']):
            result = main()
        
        assert result == 1


class TestTrainingIntegration:
    """Integration tests for the complete training system"""
    
    @pytest.fixture
    def temp_model_dir(self):
        """Create a temporary model directory"""
        temp_dir = tempfile.mkdtemp()
        models_dir = os.path.join(temp_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        yield models_dir
        shutil.rmtree(temp_dir)
    
    @patch('train_ai.create_trainer')
    @patch('train_ai.enable_sequential_evaluation_with_progress')
    def test_complete_training_workflow(self, mock_eval, mock_create_trainer, temp_model_dir):
        """Test a complete training workflow end-to-end"""        # Setup mock trainer
        mock_trainer = Mock()
        mock_trainer.q_network.state_dict.return_value = {"test": "state"}
        mock_trainer.target_network.state_dict.return_value = {"test": "state"}
        mock_trainer.target_network.load_state_dict = Mock()  # Add for target network updates
        mock_trainer.optimizer.state_dict.return_value = {"test": "state"}
        mock_trainer.epsilon = 0.5
        mock_trainer.config = {'target_update_freq': 100}  # Add config for target network updates
        mock_trainer._train_episode.return_value = (-10.0, 50, False)
        mock_trainer._evaluate.return_value = (-15.0, 0.1)
        mock_create_trainer.return_value = mock_trainer
        
        # Create trainer with minimal episodes for testing
        trainer = EnhancedBeginnerTrainerV2Resume(
            save_dir=temp_model_dir,
            evaluation_method="sequential"
        )
          # Patch training phases to have fewer episodes
        trainer.training_phases = [
            {
                'name': 'Foundation',
                'episodes': 2,
                'description': 'Test phase',
                'learning_rate': 0.001,
                'epsilon_start': 0.9,
                'epsilon_end': 0.3,
                'batch_size': 64,
                'memory_size': 50000,
                'eval_frequency': 1,
                'eval_episodes': 1,
                'expected_win_rate': 0.15,
                'min_win_rate': 0.05
            }
        ]
        
        # Run training
        result = trainer.run()
        
        # Verify training completed
        assert result is True
        
        # Verify files were created
        assert os.path.exists(os.path.join(temp_model_dir, "foundation_checkpoint.pth"))
        assert os.path.exists(os.path.join(temp_model_dir, "training_metadata.json"))
        
        # Verify trainer methods were called
        mock_create_trainer.assert_called_once()
        mock_eval.assert_called_once()
        assert mock_trainer._train_episode.call_count == 2
        assert mock_trainer._evaluate.call_count == 2
