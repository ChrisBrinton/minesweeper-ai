"""
Test suite for Phase 0 Perfect Knowledge Environment
"""

import pytest
import numpy as np
import torch
from src.ai.environment import PerfectKnowledgeMinesweeperEnvironment, MinesweeperEnvironment
from src.ai.curriculum import CurriculumConfig
from src.ai.trainer import DQNTrainer


class TestPerfectKnowledgeEnvironment:
    """Test the perfect knowledge environment for Phase 0"""
    
    def test_environment_creation(self):
        """Test that the perfect knowledge environment can be created"""
        env = PerfectKnowledgeMinesweeperEnvironment(rows=5, cols=5, mines=3)
        assert env.rows == 5
        assert env.cols == 5
        assert env.mines == 3
    
    def test_observation_shape(self):
        """Test that observations have the correct shape (4 channels)"""
        env = PerfectKnowledgeMinesweeperEnvironment(rows=5, cols=5, mines=3)
        obs = env.reset()
        
        # Should have 4 channels: visible, adjacent, flags, mine_locations
        assert obs.shape == (5, 5, 4), f"Expected shape (5, 5, 4), got {obs.shape}"
    
    def test_initial_state(self):
        """Test that the environment starts with correct initial state"""
        env = PerfectKnowledgeMinesweeperEnvironment(rows=5, cols=5, mines=3)
        obs = env.reset()
        
        # Initially no cells should be revealed
        visible_channel = obs[:, :, 0]
        revealed_count = np.sum(visible_channel > 0)
        assert revealed_count == 0, f"Expected 0 initially revealed cells, got {revealed_count}"
        
        # Initially no mines should be visible (mines placed on first move)
        mine_channel = obs[:, :, 3]
        mine_count = np.sum(mine_channel)
        assert mine_count == 0, f"Expected 0 initially visible mines, got {mine_count}"
    
    def test_perfect_knowledge_after_first_move(self):
        """Test that perfect knowledge is provided after first move"""
        env = PerfectKnowledgeMinesweeperEnvironment(rows=5, cols=5, mines=3)
        obs = env.reset()
        
        # Make first move
        action = 1 * env.cols * 3 + 1 * 3 + 0  # Reveal cell (1,1)
        obs, reward, done, info = env.step(action)
        
        # After first move, mines should be visible
        mine_channel = obs[:, :, 3]
        mine_count = np.sum(mine_channel)
        assert mine_count == 3, f"Expected 3 visible mines after first move, got {mine_count}"
        
        # Check info contains perfect knowledge flag
        assert info.get('has_perfect_info', False), "Info should indicate perfect knowledge available"
        assert info.get('phase') == 'perfect_knowledge', "Info should indicate perfect knowledge phase"
    
    def test_comparison_with_standard_environment(self):
        """Test differences with standard environment"""
        std_env = MinesweeperEnvironment(rows=5, cols=5, mines=3)
        pk_env = PerfectKnowledgeMinesweeperEnvironment(rows=5, cols=5, mines=3)
        
        std_obs = std_env.reset()
        pk_obs = pk_env.reset()
        
        # Standard environment has 3 channels
        assert std_obs.shape == (5, 5, 3), f"Standard env should have 3 channels, got {std_obs.shape}"
        
        # Perfect knowledge environment has 4 channels
        assert pk_obs.shape == (5, 5, 4), f"Perfect knowledge env should have 4 channels, got {pk_obs.shape}"
    
    def test_game_mechanics_work(self):
        """Test that game mechanics work normally"""
        env = PerfectKnowledgeMinesweeperEnvironment(rows=5, cols=5, mines=3)
        obs = env.reset()
        
        # Make first move
        action = 1 * env.cols * 3 + 1 * 3 + 0  # Reveal cell (1,1)
        obs, reward, done, info = env.step(action)
        
        # Should get some reward and process the action
        assert reward != 0, "Should receive some reward for making a move"
        assert info.get('action_success', False), "Action should be successful"
        assert info.get('steps_taken', 0) > 0, "Steps should be tracked"
        
        # Test flagging if mines are visible
        if np.sum(obs[:, :, 3]) > 0:
            mine_positions = np.where(obs[:, :, 3] == 1.0)
            if len(mine_positions[0]) > 0:
                mine_row, mine_col = mine_positions[0][0], mine_positions[1][0]
                flag_action = mine_row * env.cols * 3 + mine_col * 3 + 1
                
                obs, reward, done, info = env.step(flag_action)
                
                # Check that flag was placed
                flag_placed = obs[mine_row, mine_col, 2] > 0
                assert flag_placed, "Flag should be placed on mine"


class TestCurriculumIntegration:
    """Test integration with curriculum system"""
    
    def test_phase_0_config(self):
        """Test that phase_0 configuration is correct"""
        config = CurriculumConfig.get_stage_config('phase_0')
        
        assert config is not None, "Phase 0 config should exist"
        assert config['name'] == 'phase_0', "Config name should be 'phase_0'"
        assert config.get('fully_revealed', False), "Phase 0 should be marked as fully_revealed"
        assert config['rows'] == 5, "Phase 0 should use 5x5 board"
        assert config['cols'] == 5, "Phase 0 should use 5x5 board"
        assert config['mines'] == 3, "Phase 0 should use 3 mines"
    
    def test_trainer_auto_detection(self):
        """Test that trainer auto-detects 4-channel input"""
        env = PerfectKnowledgeMinesweeperEnvironment(rows=5, cols=5, mines=3)
        
        config = {
            'max_episodes': 1,
            'max_steps_per_episode': 10,
            'learning_rate': 0.001,
            'epsilon_start': 1.0,
            'epsilon_min': 0.1,
            'epsilon_decay': 0.99,
            'batch_size': 32,
            'target_update_freq': 5,
            'memory_size': 1000
        }
        
        trainer = DQNTrainer(env, config)
        
        # Test that model can handle 4-channel input
        obs = env.reset()
        obs_tensor = torch.FloatTensor(obs).permute(2, 0, 1).unsqueeze(0).to(trainer.device)
        
        with torch.no_grad():
            q_values = trainer.q_network(obs_tensor)
            
        expected_actions = env.rows * env.cols * 3
        assert q_values.shape == (1, expected_actions), f"Expected output shape (1, {expected_actions}), got {q_values.shape}"
        assert obs_tensor.shape == (1, 4, 5, 5), f"Expected input shape (1, 4, 5, 5), got {obs_tensor.shape}"


class TestEnvironmentRewards:
    """Test reward mechanics in perfect knowledge environment"""
    
    def test_reward_structure(self):
        """Test that rewards are properly configured"""
        env = PerfectKnowledgeMinesweeperEnvironment(rows=5, cols=5, mines=3)
        
        # Check that special reward config is applied
        assert env.reward_config['flag_correct'] == 10.0, "Flag correct reward should be enhanced"
        assert env.reward_config['flag_incorrect'] == -15.0, "Flag incorrect penalty should be enhanced"
        assert env.reward_config['win'] == 150.0, "Win reward should be standard"
    
    def test_step_rewards(self):
        """Test that step rewards work correctly"""
        env = PerfectKnowledgeMinesweeperEnvironment(rows=5, cols=5, mines=3)
        obs = env.reset()
        
        # Make a move
        action = 1 * env.cols * 3 + 1 * 3 + 0  # Reveal cell (1,1)
        obs, reward, done, info = env.step(action)
        
        # Should receive some reward
        assert 'reward_breakdown' in info, "Should provide reward breakdown"
        assert reward != 0, "Should receive non-zero reward for action"


# Test helper functions
def test_environment_factory():
    """Test that we can create environments through curriculum config"""
    stage_config = CurriculumConfig.get_stage_config('phase_0')
    
    if stage_config.get('fully_revealed', False):
        env = PerfectKnowledgeMinesweeperEnvironment(
            rows=stage_config['rows'],
            cols=stage_config['cols'], 
            mines=stage_config['mines']
        )
        
        assert isinstance(env, PerfectKnowledgeMinesweeperEnvironment)
        assert env.rows == stage_config['rows']
        assert env.cols == stage_config['cols']
        assert env.mines == stage_config['mines']
