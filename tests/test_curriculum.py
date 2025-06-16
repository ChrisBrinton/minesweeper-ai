"""
Test suite for Curriculum Learning System
"""

import pytest
import os
import tempfile
from src.ai.curriculum import CurriculumConfig
from src.ai.environment import MinesweeperEnvironment, PerfectKnowledgeMinesweeperEnvironment


class TestCurriculumConfig:
    """Test curriculum configuration and stage definitions"""
    
    def test_all_stages_defined(self):
        """Test that all curriculum stages are properly defined"""
        expected_stages = ['phase_0', 'tiny', 'small', 'mini_beginner', 'beginner', 'intermediate', 'expert']
        
        for stage_name in expected_stages:
            config = CurriculumConfig.get_stage_config(stage_name)
            assert config is not None, f"Stage '{stage_name}' should be defined"
            assert 'name' in config, f"Stage '{stage_name}' should have name"
            assert 'rows' in config, f"Stage '{stage_name}' should have rows"
            assert 'cols' in config, f"Stage '{stage_name}' should have cols"
            assert 'mines' in config, f"Stage '{stage_name}' should have mines"
            assert 'target_win_rate' in config, f"Stage '{stage_name}' should have target_win_rate"
    
    def test_stage_progression(self):
        """Test that stages progress in difficulty"""
        stages = ['tiny', 'small', 'mini_beginner', 'beginner', 'intermediate', 'expert']
        
        prev_difficulty = 0
        for stage_name in stages:
            config = CurriculumConfig.get_stage_config(stage_name)
            # Simple difficulty metric: rows * cols * mines
            difficulty = config['rows'] * config['cols'] * config['mines']
            
            if stage_name != 'tiny':  # First stage to compare
                assert difficulty >= prev_difficulty, f"Stage '{stage_name}' should be harder than previous"
            prev_difficulty = difficulty
    
    def test_phase_0_special_config(self):
        """Test that phase_0 has special configuration"""
        config = CurriculumConfig.get_stage_config('phase_0')
        
        assert config['name'] == 'phase_0'
        assert config.get('fully_revealed', False), "Phase 0 should be marked as fully_revealed"
        assert config['target_win_rate'] >= 0.9, "Phase 0 should have high target win rate"
        assert 'perfect knowledge' in config['description'].lower(), "Description should mention perfect knowledge"
    
    def test_get_stage_index(self):
        """Test stage index retrieval"""
        assert CurriculumConfig.get_stage_index('phase_0') == 0
        assert CurriculumConfig.get_stage_index('tiny') == 1
        assert CurriculumConfig.get_stage_index('expert') == len(CurriculumConfig.CURRICULUM_STAGES) - 1
        assert CurriculumConfig.get_stage_index('nonexistent') == -1
    
    def test_stage_parameters_valid(self):
        """Test that all stage parameters are valid"""
        for stage in CurriculumConfig.CURRICULUM_STAGES:
            # Check basic validity
            assert stage['rows'] > 0, f"Stage '{stage['name']}' rows must be positive"
            assert stage['cols'] > 0, f"Stage '{stage['name']}' cols must be positive"
            assert stage['mines'] > 0, f"Stage '{stage['name']}' mines must be positive"
            assert stage['mines'] < stage['rows'] * stage['cols'], f"Stage '{stage['name']}' mines must be less than total cells"
            
            # Check training parameters
            assert 0 < stage['target_win_rate'] <= 1.0, f"Stage '{stage['name']}' target_win_rate must be between 0 and 1"
            assert stage['min_episodes'] > 0, f"Stage '{stage['name']}' min_episodes must be positive"
            assert stage['max_episodes'] >= stage['min_episodes'], f"Stage '{stage['name']}' max_episodes must be >= min_episodes"


class TestEnvironmentFactory:
    """Test environment creation based on curriculum config"""
    
    def test_standard_environment_creation(self):
        """Test creating standard environments from config"""
        config = CurriculumConfig.get_stage_config('tiny')
        
        env = MinesweeperEnvironment(
            rows=config['rows'],
            cols=config['cols'],
            mines=config['mines']
        )
        
        assert env.rows == config['rows']
        assert env.cols == config['cols']
        assert env.mines == config['mines']
        assert env.reset().shape == (config['rows'], config['cols'], 3)
    
    def test_perfect_knowledge_environment_creation(self):
        """Test creating perfect knowledge environment from config"""
        config = CurriculumConfig.get_stage_config('phase_0')
        
        if config.get('fully_revealed', False):
            env = PerfectKnowledgeMinesweeperEnvironment(
                rows=config['rows'],
                cols=config['cols'],
                mines=config['mines']
            )
            
            assert env.rows == config['rows']
            assert env.cols == config['cols']
            assert env.mines == config['mines']
            assert env.reset().shape == (config['rows'], config['cols'], 4)
    
    def test_environment_factory_function(self):
        """Test a factory function for creating environments"""
        def create_environment_from_config(stage_name):
            """Factory function to create environment from stage config"""
            config = CurriculumConfig.get_stage_config(stage_name)
            if config is None:
                return None
            
            if config.get('fully_revealed', False):
                return PerfectKnowledgeMinesweeperEnvironment(
                    rows=config['rows'],
                    cols=config['cols'],
                    mines=config['mines']
                )
            else:
                return MinesweeperEnvironment(
                    rows=config['rows'],
                    cols=config['cols'],
                    mines=config['mines']
                )
        
        # Test phase_0 creates perfect knowledge environment
        phase_0_env = create_environment_from_config('phase_0')
        assert isinstance(phase_0_env, PerfectKnowledgeMinesweeperEnvironment)
        
        # Test other stages create standard environment
        tiny_env = create_environment_from_config('tiny')
        assert isinstance(tiny_env, MinesweeperEnvironment)
        assert not isinstance(tiny_env, PerfectKnowledgeMinesweeperEnvironment)


class TestRewardConfiguration:
    """Test reward system configuration"""
    
    def test_standard_reward_config(self):
        """Test that standard environments have expected reward config"""
        env = MinesweeperEnvironment(5, 5, 3)
        
        # Check key reward values
        assert 'win' in env.reward_config
        assert 'lose' in env.reward_config
        assert 'reveal_safe' in env.reward_config
        assert 'flag_correct' in env.reward_config
        assert 'flag_incorrect' in env.reward_config
        assert 'unflag_penalty' in env.reward_config
        
        # Check that unflag penalty exists (anti-exploit measure)
        assert env.reward_config['unflag_penalty'] < 0, "Unflag penalty should be negative"
    
    def test_perfect_knowledge_reward_config(self):
        """Test that perfect knowledge environment has enhanced rewards"""
        env = PerfectKnowledgeMinesweeperEnvironment(5, 5, 3)
        
        # Should have enhanced flag rewards for learning
        assert env.reward_config['flag_correct'] > 5.0, "Perfect knowledge env should have enhanced flag correct reward"
        assert env.reward_config['flag_incorrect'] < -10.0, "Perfect knowledge env should have enhanced flag incorrect penalty"
    
    def test_reward_consistency(self):
        """Test that reward configurations are consistent"""
        env = MinesweeperEnvironment(5, 5, 3)
        
        # Win should be positive, lose should be negative
        assert env.reward_config['win'] > 0, "Win reward should be positive"
        assert env.reward_config['lose'] < 0, "Lose penalty should be negative"
        
        # Flag correct should be positive, incorrect negative
        assert env.reward_config['flag_correct'] > 0, "Flag correct reward should be positive"
        assert env.reward_config['flag_incorrect'] < 0, "Flag incorrect penalty should be negative"
        
        # Step penalty should be small and negative
        assert -1.0 < env.reward_config['step_penalty'] < 0, "Step penalty should be small and negative"
