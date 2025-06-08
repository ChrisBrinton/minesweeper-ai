"""
Demo script for Minesweeper AI Training Framework
Shows how to use the API, environment, and training components
"""

import sys
import os
import numpy as np
import torch

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ai import MinesweeperAPI, MinesweeperEnvironment, DQNTrainer, create_trainer, create_model, count_parameters
from ai.game_api import Action


def demo_api():
    """Demonstrate the Minesweeper API"""
    print("=== Minesweeper API Demo ===")
    
    # Create API instance
    api = MinesweeperAPI(rows=9, cols=9, mines=10)
    
    print(f"Created {api.rows}x{api.cols} board with {api.mines} mines")
    
    # Get initial state
    state = api.get_game_state()
    print(f"Initial game state: {state['game_state']}")
    print(f"Board size: {state['board_size']}")
    print(f"Total mines: {state['total_mines']}")
    
    # Take some actions
    print("\nTaking actions...")
    
    # Reveal center cell (first click is always safe)
    result = api.take_action(4, 4, Action.REVEAL)
    print(f"Revealed (4,4): success={result['success']}")
    
    # Try to flag a cell
    result = api.take_action(0, 0, Action.FLAG)
    print(f"Flagged (0,0): success={result['success']}")
    
    # Try to unflag
    result = api.take_action(0, 0, Action.UNFLAG)
    print(f"Unflagged (0,0): success={result['success']}")
    
    # Get updated state
    state = api.get_game_state()
    print(f"Game state after actions: {state['game_state']}")
    print(f"Cells revealed: {state['cells_revealed']}")
    print(f"Flags used: {state['flags_used']}")
    
    # Get board as numpy array
    board_array = api.get_board_array()
    print(f"Board array shape: {board_array.shape}")
    
    # Get valid actions
    valid_actions = api.get_valid_actions()
    print(f"Number of valid actions: {len(valid_actions)}")
    if valid_actions:
        print(f"First few valid actions: {valid_actions[:5]}")
    
    print()


def demo_environment():
    """Demonstrate the Minesweeper Environment"""
    print("=== Minesweeper Environment Demo ===")
    
    # Create environment
    env = MinesweeperEnvironment(rows=9, cols=9, mines=10)
    
    print(f"Created environment: {env.rows}x{env.cols} with {env.mines} mines")
    print(f"Action space size: {env.action_space_size}")
    print(f"Observation space shape: {env.observation_space_shape}")
    
    # Reset environment
    observation = env.reset()
    print(f"Initial observation shape: {observation.shape}")
    
    # Take random actions
    print("\nTaking random actions...")
    total_reward = 0
    steps = 0
    
    for step in range(10):
        # Get valid actions and sample one
        action = env.sample_action()
        
        # Take step
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        
        print(f"Step {step+1}: action={action}, reward={reward:.2f}, done={done}")
        print(f"  Game state: {info['game_state']}")
        print(f"  Cells revealed: {info['cells_revealed']}")
        
        if done:
            print(f"Game finished! Final reward: {total_reward:.2f}")
            break
    
    # Render final state
    print("\nFinal board state:")
    env.render()
    print()


def demo_model():
    """Demonstrate the neural network models"""
    print("=== Neural Network Models Demo ===")
    
    # Create DQN model
    model = create_model('dqn', board_height=9, board_width=9)
    
    print(f"Created DQN model")
    print(f"Number of parameters: {count_parameters(model):,}")
    
    # Create sample input
    batch_size = 4
    channels = 3
    height, width = 9, 9
    
    sample_input = torch.randn(batch_size, channels, height, width)
    
    # Forward pass
    with torch.no_grad():
        q_values = model(sample_input)
    
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {q_values.shape}")
    print(f"Sample Q-values: {q_values[0, :10].numpy()}")
    
    # Test action selection
    action_mask = torch.ones(model.num_actions, dtype=bool)
    action = model.get_action(sample_input[0], action_mask, epsilon=0.1)
    print(f"Selected action: {action}")
    
    print()


def demo_simple_training():
    """Demonstrate simple training loop"""
    print("=== Simple Training Demo ===")
    
    # Create trainer for beginner difficulty
    trainer = create_trainer('beginner', config={
        'max_episodes': 100,
        'eval_freq': 25,
        'save_freq': 50,
        'batch_size': 16,
        'epsilon_decay': 0.99
    })
    
    print("Created DQN trainer for beginner difficulty")
    print(f"Network has {count_parameters(trainer.q_network):,} parameters")
    
    # Run short training
    print("Starting training for 100 episodes...")
    
    try:
        metrics = trainer.train(save_dir="demo_models")
        
        print("Training completed!")
        print(f"Final average reward: {np.mean(metrics['training_rewards'][-10:]):.2f}")
        print(f"Final win rate: {np.mean(metrics['training_wins'][-10:]):.3f}")
        
        # Plot metrics if matplotlib is available
        try:
            trainer.plot_training_metrics("demo_training_plots.png")
        except Exception as e:
            print(f"Could not plot metrics: {e}")
            
    except KeyboardInterrupt:
        print("Training interrupted by user")
    
    print()


def demo_interactive_game():
    """Interactive game demo using trained model"""
    print("=== Interactive Game Demo ===")
    
    # Create environment
    env = MinesweeperEnvironment(rows=9, cols=9, mines=10)
    
    # Try to load a trained model
    model_path = "demo_models/dqn_final.pth"
    
    if os.path.exists(model_path):
        print("Loading trained model...")
        trainer = create_trainer('beginner')
        trainer.load_model(model_path)
        model = trainer.q_network
        model.eval()
        
        print("Playing game with trained model...")
        
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < 100:
            # Render current state
            print(f"\nStep {steps + 1}:")
            env.render()
            
            # Get action from model
            state_tensor = torch.FloatTensor(state).permute(2, 0, 1)
            action_mask = torch.BoolTensor(env.get_action_mask())
            
            with torch.no_grad():
                action = model.get_action(state_tensor, action_mask, epsilon=0.0)
            
            # Take step
            state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            row, col, action_type = env._decode_action(action)
            action_names = ['REVEAL', 'FLAG', 'UNFLAG']
            print(f"Action: {action_names[action_type]} at ({row}, {col})")
            print(f"Reward: {reward:.2f}, Total: {total_reward:.2f}")
            
            if done:
                print(f"\nGame finished!")
                print(f"Result: {info['game_state']}")
                print(f"Final reward: {total_reward:.2f}")
                env.render()
                break
            
            # Pause for readability
            input("Press Enter to continue...")
    else:
        print("No trained model found. Run demo_simple_training() first.")
    
    print()


def main():
    """Run all demos"""
    print("Minesweeper AI Training Framework Demo")
    print("=" * 50)
    
    try:
        demo_api()
        demo_environment() 
        demo_model()
        
        # Ask user about training demo
        response = input("Run training demo? This may take a few minutes (y/n): ")
        if response.lower() in ['y', 'yes']:
            demo_simple_training()
            
            # Ask about interactive demo
            response = input("Run interactive game demo? (y/n): ")
            if response.lower() in ['y', 'yes']:
                demo_interactive_game()
    
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Error in demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
