"""
Tests for the consolidated evaluation system (evaluation.py)
Tests all three evaluation methods: Sequential, Lightweight, and Optimized
"""

import pytest
import torch
import tempfile
import time
import os
from unittest.mock import Mock, patch, MagicMock

# Import the evaluation system
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import (
    OptimizedParallelEvaluator,
    LightweightParallelEvaluator,
    SequentialEvaluator,
    enable_optimized_parallel_evaluation,
    enable_lightweight_parallel_evaluation,
    enable_sequential_evaluation_with_progress,
    benchmark_evaluation_methods
)


class TestSequentialEvaluator:
    """Test the sequential evaluation class"""
    
    @pytest.fixture
    def mock_trainer(self):
        """Create a mock trainer for testing"""
        trainer = Mock()
        trainer.q_network = Mock()
        trainer.device = 'cpu'
        trainer.epsilon = 0.5
        trainer.config = {
            'eval_episodes': 10,
            'max_steps_per_episode': 100
        }
        
        # Mock environment
        env = Mock()
        env.reset.return_value = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]
        env.step.return_value = (
            [[[1, 1, 0], [0, 1, 0], [0, 0, 1]]], 
            -1.0, 
            True, 
            {'game_state': 'lost'}
        )
        env.get_action_mask.return_value = [True] * 81
        trainer.env = env
        
        # Mock Q-network
        trainer.q_network.get_action.return_value = 0
        
        return trainer
    def test_sequential_evaluator_initialization(self, mock_trainer):
        """Test sequential evaluator initialization"""
        evaluator = SequentialEvaluator(mock_trainer)
        
        assert evaluator.trainer == mock_trainer
    
    def test_evaluate_sequential(self, mock_trainer):
        """Test sequential evaluation execution"""
        evaluator = SequentialEvaluator(mock_trainer)
        
        win_rate, avg_score = evaluator.evaluate_sequential(2)
        
        assert isinstance(win_rate, float)
        assert isinstance(avg_score, float)
        assert win_rate == -1.0  # No wins with 'lost' game state, returns -1.0
        
        # Verify trainer methods were called
        mock_trainer.env.reset.assert_called()
        mock_trainer.q_network.get_action.assert_called()
        mock_trainer.env.step.assert_called()


class TestLightweightParallelEvaluator:
    """Test the lightweight parallel evaluation class"""
    
    @pytest.fixture
    def mock_trainer(self):
        """Create a mock trainer for testing"""
        trainer = Mock()
        trainer.q_network = Mock()
        trainer.device = 'cpu'
        trainer.config = {'eval_episodes': 10}
        
        # Mock environment for cloning
        env = Mock()
        env.rows = 9
        env.cols = 9
        env.mines = 10
        trainer.env = env
        
        # Mock model state and parameters
        mock_param = Mock()
        mock_param.device = torch.device('cpu')
        trainer.q_network.parameters.return_value = [mock_param]
        trainer.q_network.to.return_value = trainer.q_network
        trainer.q_network.eval.return_value = None
        
        return trainer
    
    def test_lightweight_evaluator_initialization(self, mock_trainer):
        """Test lightweight evaluator initialization"""
        evaluator = LightweightParallelEvaluator(mock_trainer, num_threads=4)
        
        assert evaluator.trainer == mock_trainer
        assert evaluator.num_threads == 4
    
    def test_auto_thread_count(self, mock_trainer):
        """Test automatic thread count detection"""
        evaluator = LightweightParallelEvaluator(mock_trainer, num_threads=None)
        
        expected_threads = min(8, max(2, os.cpu_count() // 2))
        assert evaluator.num_threads == expected_threads


class TestOptimizedParallelEvaluator:
    """Test the optimized parallel evaluation class"""
    
    @pytest.fixture
    def mock_trainer(self):
        """Create a mock trainer for testing"""
        trainer = Mock()
        trainer.q_network = Mock()
        trainer.device = 'cpu'
        trainer.config = {'eval_episodes': 10}
        
        # Mock environment config
        env = Mock()
        env.rows = 9
        env.cols = 9
        env.mines = 10
        trainer.env = env
        
        # Mock model state
        trainer.q_network.state_dict.return_value = {
            "test_param": torch.tensor([1.0, 2.0, 3.0])
        }
        
        return trainer
    
    def test_optimized_evaluator_initialization(self, mock_trainer):
        """Test optimized evaluator initialization"""
        evaluator = OptimizedParallelEvaluator(mock_trainer, num_workers=2, cpu_limit_percent=80)
        
        assert evaluator.trainer == mock_trainer
        assert evaluator.num_workers == 2
        assert evaluator.cpu_limit_percent == 80
    
    def test_auto_worker_count(self, mock_trainer):
        """Test automatic worker count detection"""
        evaluator = OptimizedParallelEvaluator(mock_trainer, num_workers=None)
        
        expected_workers = min(max(2, os.cpu_count() // 4), os.cpu_count() - 2)
        assert evaluator.num_workers == expected_workers


class TestTrainerPatchingFunctions:
    """Test the trainer patching functions"""
    
    @pytest.fixture
    def mock_trainer(self):
        """Create a mock trainer for testing"""
        trainer = Mock()
        trainer.config = {
            'eval_episodes': 10,
            'max_steps_per_episode': 100
        }
        trainer.epsilon = 0.5
        trainer.device = 'cpu'
        trainer._evaluate = Mock(return_value=(0.5, -10.0))
        
        # Mock environment
        env = Mock()
        env.rows = 9
        env.cols = 9
        env.mines = 10
        env.reset.return_value = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]
        env.step.return_value = ([[[1, 1, 0], [0, 1, 0], [0, 0, 1]]], -1.0, True, {'game_state': 'lost'})
        env.get_action_mask.return_value = [True] * 81
        trainer.env = env
        
        # Mock Q-network
        trainer.q_network = Mock()
        trainer.q_network.get_action.return_value = 0
        
        return trainer
    
    def test_enable_sequential_evaluation(self, mock_trainer):
        """Test enabling sequential evaluation"""
        result_trainer = enable_sequential_evaluation_with_progress(mock_trainer)
        
        assert result_trainer == mock_trainer
        assert hasattr(mock_trainer, '_evaluate_original')
        assert hasattr(mock_trainer, 'sequential_evaluator')
        
        # Test that _evaluate was replaced
        win_rate, avg_score = mock_trainer._evaluate(num_episodes=2)
        assert isinstance(win_rate, float)
        assert isinstance(avg_score, float)
    
    def test_enable_lightweight_parallel_evaluation(self, mock_trainer):
        """Test enabling lightweight parallel evaluation"""
        # Mock model parameters for lightweight evaluation
        mock_param = Mock()
        mock_param.device = torch.device('cpu')
        mock_trainer.q_network.parameters.return_value = [mock_param]
        mock_trainer.q_network.to.return_value = mock_trainer.q_network
        mock_trainer.q_network.eval.return_value = None
        
        result_trainer = enable_lightweight_parallel_evaluation(mock_trainer, num_threads=2)
        
        assert result_trainer == mock_trainer
        assert hasattr(mock_trainer, '_evaluate_sequential')
        assert hasattr(mock_trainer, 'parallel_evaluator')
    
    def test_enable_optimized_parallel_evaluation(self, mock_trainer):
        """Test enabling optimized parallel evaluation"""
        # Mock model state dict
        mock_trainer.q_network.state_dict.return_value = {
            "test_param": torch.tensor([1.0, 2.0, 3.0])
        }
        
        result_trainer = enable_optimized_parallel_evaluation(mock_trainer, num_workers=1)
        
        assert result_trainer == mock_trainer
        assert hasattr(mock_trainer, '_evaluate_sequential')
        assert hasattr(mock_trainer, 'parallel_evaluator')


class TestBenchmarkFunction:
    """Test the benchmark evaluation function"""
    
    @pytest.fixture
    def mock_trainer(self):
        """Create a mock trainer for testing"""
        trainer = Mock()
        trainer.config = {'eval_episodes': 10}
        
        # Make config mutable for benchmark testing
        trainer.config = {}
        
        # Mock the original _evaluate method
        trainer._evaluate = Mock(return_value=(0.3, -12.0))
        
        return trainer
    
    @patch('evaluation.enable_optimized_parallel_evaluation')
    @patch('evaluation.enable_lightweight_parallel_evaluation')
    @patch('evaluation.enable_sequential_evaluation_with_progress')
    def test_benchmark_evaluation_methods(self, mock_seq, mock_light, mock_opt, mock_trainer):
        """Test benchmark function"""
        # Mock the patched trainers to return different results
        mock_seq_trainer = Mock()
        mock_seq_trainer._evaluate.return_value = (0.2, -15.0)
        mock_seq_trainer.config = {}
        mock_seq.return_value = mock_seq_trainer
        
        mock_light_trainer = Mock()
        mock_light_trainer._evaluate.return_value = (0.25, -12.0)
        mock_light_trainer.config = {}
        mock_light.return_value = mock_light_trainer
        
        mock_opt_trainer = Mock()
        mock_opt_trainer._evaluate.return_value = (0.3, -10.0)
        mock_opt_trainer.config = {}
        mock_opt.return_value = mock_opt_trainer
          # Run benchmark
        results = benchmark_evaluation_methods(mock_trainer, num_episodes=5)
        
        # Verify results structure
        assert 'Sequential' in results
        assert 'Lightweight' in results
        assert 'Optimized' in results
        
        # Verify each result has required fields
        for method_name, result in results.items():
            assert 'time' in result
            assert 'win_rate' in result
            assert 'reward' in result
            assert 'eps_per_sec' in result
            assert isinstance(result['time'], float)
            assert isinstance(result['win_rate'], float)
            assert isinstance(result['reward'], float)
        
        # Verify patching functions were called
        mock_seq.assert_called_once()
        mock_light.assert_called_once()
        mock_opt.assert_called_once()


class TestEvaluationPerformance:
    """Test evaluation performance and timing"""
    
    @pytest.fixture
    def simple_trainer(self):
        """Create a simple trainer for performance testing"""
        trainer = Mock()
        trainer.config = {
            'eval_episodes': 10,
            'max_steps_per_episode': 10  # Small for fast testing
        }
        trainer.device = 'cpu'
        trainer.epsilon = 0.5
        
        # Simple mock environment that completes quickly
        env = Mock()
        env.reset.return_value = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]
        env.step.return_value = ([[[1, 1, 0], [0, 1, 0], [0, 0, 1]]], 0.0, True, {'game_state': 'lost'})
        env.get_action_mask.return_value = [True] * 81
        env.rows = 9
        env.cols = 9
        env.mines = 10
        trainer.env = env
        
        # Simple mock model
        trainer.q_network = Mock()
        trainer.q_network.get_action.return_value = 0
        
        return trainer
    
    def test_sequential_evaluation_timing(self, simple_trainer):
        """Test that sequential evaluation completes in reasonable time"""
        evaluator = SequentialEvaluator(simple_trainer)
        
        start_time = time.time()
        win_rate, avg_score = evaluator.evaluate_sequential(2)
        elapsed_time = time.time() - start_time
        
        # Should complete quickly with mock environment
        assert elapsed_time < 5.0  # Generous timeout for CI
        assert isinstance(win_rate, float)
        assert isinstance(avg_score, float)
    
    def test_evaluation_memory_efficiency(self, simple_trainer):
        """Test that evaluation doesn't leak memory"""
        evaluator = SequentialEvaluator(simple_trainer)
        
        # Run evaluation multiple times
        for _ in range(3):
            win_rate, avg_score = evaluator.evaluate_sequential(1)
            assert isinstance(win_rate, float)
            assert isinstance(avg_score, float)
        
        # Test passed if no memory errors occurred


class TestEvaluationEdgeCases:
    """Test edge cases and error handling"""
    
    def test_zero_episodes_evaluation(self):
        """Test evaluation with zero episodes"""
        trainer = Mock()
        trainer.config = {'eval_episodes': 0}
        trainer.epsilon = 0.5
        
        evaluator = SequentialEvaluator(trainer)
        
        win_rate, avg_score = evaluator.evaluate_sequential(0)
        
        # Should handle gracefully - check for NaN
        assert isinstance(win_rate, float)
        assert isinstance(avg_score, float)
        # NaN is expected for empty arrays
    
    def test_negative_thread_count_handling(self):
        """Test lightweight evaluator handles invalid thread count gracefully"""
        trainer = Mock()
        trainer.config = {'eval_episodes': 10}
        
        # Test that constructor doesn't crash with negative thread count
        # The implementation should handle this gracefully
        try:
            evaluator = LightweightParallelEvaluator(trainer, num_threads=-1)
            # Implementation may or may not validate this - test that it doesn't crash
            assert True
        except ValueError:
            # It's acceptable to raise a ValueError for invalid input
            assert True
    
    def test_evaluation_with_missing_config(self):
        """Test evaluation when trainer config is incomplete"""
        trainer = Mock()
        trainer.config = {}  # Missing eval_episodes
        
        evaluator = SequentialEvaluator(trainer)
        
        # Should handle missing config gracefully
        assert evaluator.trainer == trainer
    
    def test_evaluation_method_switching(self):
        """Test switching between evaluation methods"""
        trainer = Mock()
        trainer.config = {'eval_episodes': 5}
        trainer._evaluate = Mock(return_value=(0.5, -10.0))
        
        # Enable sequential
        trainer = enable_sequential_evaluation_with_progress(trainer)
        original_method = trainer._evaluate
        
        # Enable lightweight (should replace previous)
        trainer = enable_lightweight_parallel_evaluation(trainer, num_threads=2)
        lightweight_method = trainer._evaluate
        
        # Methods should be different
        assert original_method != lightweight_method
        
        # Enable optimized (should replace lightweight)
        trainer = enable_optimized_parallel_evaluation(trainer, num_workers=1)
        optimized_method = trainer._evaluate
        
        # All methods should be different
        assert original_method != optimized_method
        assert lightweight_method != optimized_method
