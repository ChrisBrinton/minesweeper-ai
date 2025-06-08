#!/usr/bin/env python3
"""
GPU Performance Benchmark and Model Testing for Minesweeper AI
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt

from src.ai import MinesweeperEnvironment, DQNTrainer, DQN

class GPUBenchmark:
    """Comprehensive GPU benchmarking for Minesweeper AI training"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}
        
    def run_full_benchmark(self):
        """Run complete GPU benchmark suite"""
        print("🚀 Starting Comprehensive GPU Benchmark")
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print("=" * 60)
        
        # 1. Basic GPU Operations Benchmark
        self.benchmark_basic_operations()
        
        # 2. Neural Network Operations Benchmark
        self.benchmark_neural_network_ops()
        
        # 3. Training Speed Benchmark
        self.benchmark_training_speed()
        
        # 4. Memory Usage Benchmark
        self.benchmark_memory_usage()
        
        # 5. Model Performance Test
        self.test_trained_model()
        
        # Save and display results
        self.save_benchmark_results()
        self.display_summary()
        
    def benchmark_basic_operations(self):
        """Benchmark basic tensor operations"""
        print("🔧 Benchmarking Basic GPU Operations...")
        
        results = {}
        
        # Matrix multiplication benchmark
        sizes = [512, 1024, 2048]
        for size in sizes:
            times = []
            for _ in range(5):
                a = torch.randn(size, size).to(self.device)
                b = torch.randn(size, size).to(self.device)
                
                start_time = time.time()
                c = torch.mm(a, b)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            results[f'matmul_{size}x{size}'] = avg_time
            print(f"   Matrix Multiplication {size}x{size}: {avg_time:.4f}s")
        
        self.results['basic_operations'] = results
        
    def benchmark_neural_network_ops(self):
        """Benchmark neural network operations"""
        print("🧠 Benchmarking Neural Network Operations...")
        
        results = {}
          # Create a sample network
        env = MinesweeperEnvironment(9, 9, 10)
        model = DQN(env.rows, env.cols, input_channels=3, num_actions=env.action_space_size).to(self.device)
        
        # Forward pass benchmark
        batch_sizes = [16, 32, 64, 128]
        for batch_size in batch_sizes:
            times = []
            for _ in range(10):
                x = torch.randn(batch_size, 3, 9, 9).to(self.device)
                
                start_time = time.time()
                with torch.no_grad():
                    output = model(x)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            results[f'forward_pass_batch_{batch_size}'] = avg_time
            print(f"   Forward Pass (batch {batch_size}): {avg_time:.4f}s")
        
        self.results['neural_network_ops'] = results
        
    def benchmark_training_speed(self):
        """Benchmark training speed for different configurations"""
        print("⚡ Benchmarking Training Speed...")
        
        results = {}
        
        # Test different batch sizes
        batch_sizes = [16, 32, 64, 128]
        
        for batch_size in batch_sizes:
            print(f"   Testing batch size {batch_size}...")
            
            env = MinesweeperEnvironment(9, 9, 10)
            config = {
                'max_episodes': 50,
                'batch_size': batch_size,
                'memory_size': 1000,
                'min_memory_size': 100,
                'eval_freq': 999  # Disable evaluation for speed
            }
            
            trainer = DQNTrainer(env, config)
            
            start_time = time.time()
            trainer.train(save_dir="temp_benchmark")
            end_time = time.time()
            
            total_time = end_time - start_time
            episodes_per_second = 50 / total_time
            
            results[f'batch_size_{batch_size}'] = {
                'total_time': total_time,
                'episodes_per_second': episodes_per_second
            }
            
            print(f"      Time: {total_time:.2f}s, Episodes/sec: {episodes_per_second:.2f}")
        
        self.results['training_speed'] = results
        
        # Clean up temp files
        import shutil
        if os.path.exists("temp_benchmark"):
            shutil.rmtree("temp_benchmark")
    
    def benchmark_memory_usage(self):
        """Benchmark memory usage patterns"""
        print("💾 Benchmarking Memory Usage...")
        
        if not torch.cuda.is_available():
            print("   Skipping memory benchmark (CUDA not available)")
            return
        
        results = {}
          # Test memory usage with different model sizes
        configurations = [
            ('small', 9, 9, 10),
            ('medium', 16, 16, 40),
            ('large', 16, 30, 99)
        ]
        
        for config_name, rows, cols, mines in configurations:
            torch.cuda.empty_cache()  # Clear cache
            
            env = MinesweeperEnvironment(rows, cols, mines)
            model = DQN(env.rows, env.cols, input_channels=3, num_actions=env.action_space_size).to(self.device)
            
            # Measure memory after model creation
            memory_model = torch.cuda.memory_allocated() / 1024**2  # MB
            
            # Create a batch of data
            batch_size = 64
            dummy_input = torch.randn(batch_size, 3, rows, cols).to(self.device)
            
            # Forward pass
            output = model(dummy_input)
            memory_forward = torch.cuda.memory_allocated() / 1024**2  # MB
            
            results[config_name] = {
                'model_memory_mb': memory_model,
                'forward_memory_mb': memory_forward,
                'total_memory_mb': memory_forward
            }
            
            print(f"   {config_name.capitalize()} ({rows}x{cols}): Model={memory_model:.1f}MB, Forward={memory_forward:.1f}MB")
        
        self.results['memory_usage'] = results
    
    def test_trained_model(self):
        """Test the performance of a trained model"""
        print("🎯 Testing Trained Model Performance...")
        
        # Find the latest trained model
        models_dir = "models_cuda"
        if not os.path.exists(models_dir):
            print("   No trained models found")
            return
        
        # Get the most recent model directory
        model_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
        if not model_dirs:
            print("   No trained models found")
            return
        
        latest_dir = os.path.join(models_dir, sorted(model_dirs)[-1])
        model_path = os.path.join(latest_dir, "dqn_final.pth")
        
        if not os.path.exists(model_path):
            print(f"   Model file not found: {model_path}")
            return
        
        # Load and test the model
        env = MinesweeperEnvironment(9, 9, 10)
        trainer = DQNTrainer(env)
        
        try:
            trainer.load_model(model_path)
            print(f"   Loaded model from: {model_path}")
            
            # Run evaluation
            avg_reward, win_rate = trainer._evaluate()
            
            # Test inference speed
            state = env.reset()
            state_tensor = torch.FloatTensor(state).permute(2, 0, 1).to(self.device)
            action_mask = torch.BoolTensor(env.get_action_mask()).to(self.device)
            
            inference_times = []
            for _ in range(100):
                start_time = time.time()
                with torch.no_grad():
                    action = trainer.q_network.get_action(state_tensor, action_mask, 0.0)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                inference_times.append(end_time - start_time)
            
            avg_inference_time = np.mean(inference_times)
            
            results = {
                'avg_reward': avg_reward,
                'win_rate': win_rate,
                'avg_inference_time_ms': avg_inference_time * 1000,
                'inference_fps': 1.0 / avg_inference_time
            }
            
            self.results['trained_model'] = results
            
            print(f"   Average Reward: {avg_reward:.2f}")
            print(f"   Win Rate: {win_rate:.3f}")
            print(f"   Inference Time: {avg_inference_time*1000:.2f}ms")
            print(f"   Inference FPS: {1.0/avg_inference_time:.1f}")
            
        except Exception as e:
            print(f"   Error testing model: {e}")
    
    def save_benchmark_results(self):
        """Save benchmark results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"gpu_benchmark_{timestamp}.json"
        
        # Add system info
        system_info = {
            'timestamp': timestamp,
            'device': str(self.device),
            'pytorch_version': torch.__version__
        }
        
        if torch.cuda.is_available():
            system_info.update({
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3,
                'cuda_version': torch.version.cuda
            })
        
        final_results = {
            'system_info': system_info,
            'benchmark_results': self.results
        }
        
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n📊 Benchmark results saved to: {results_file}")
    
    def display_summary(self):
        """Display benchmark summary"""
        print("\n" + "="*60)
        print("🏆 GPU BENCHMARK SUMMARY")
        print("="*60)
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("Device: CPU")
        
        print(f"PyTorch: {torch.__version__}")
        
        if 'training_speed' in self.results:
            print(f"\n📈 Training Performance:")
            best_batch = max(self.results['training_speed'].items(), 
                           key=lambda x: x[1]['episodes_per_second'])
            print(f"   Best Performance: Batch size {best_batch[0].split('_')[-1]} - {best_batch[1]['episodes_per_second']:.2f} episodes/sec")
        
        if 'trained_model' in self.results:
            print(f"\n🎯 Trained Model Performance:")
            model_results = self.results['trained_model']
            print(f"   Win Rate: {model_results['win_rate']:.3f}")
            print(f"   Average Reward: {model_results['avg_reward']:.2f}")
            print(f"   Inference Speed: {model_results['inference_fps']:.1f} FPS")
        
        print("\n✅ Benchmark completed successfully!")

def run_quick_gpu_test():
    """Run a quick GPU functionality test"""
    print("🔍 Quick GPU Test")
    print("-" * 30)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Test basic operation
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        z = torch.mm(x, y)
        print(f"✅ Basic tensor operations working")
          # Test neural network
        from src.ai import MinesweeperEnvironment, DQN
        env = MinesweeperEnvironment(9, 9, 10)
        model = DQN(env.rows, env.cols, input_channels=3, num_actions=env.action_space_size).to(device)
        dummy_input = torch.randn(1, 3, 9, 9).to(device)
        output = model(dummy_input)
        print(f"✅ Neural network operations working")
        
        print(f"✅ GPU setup is fully functional!")
    else:
        print("❌ CUDA not available")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU Performance Benchmark")
    parser.add_argument("--quick", action="store_true", help="Run quick GPU test only")
    parser.add_argument("--full", action="store_true", help="Run full benchmark suite")
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_gpu_test()
    elif args.full:
        benchmark = GPUBenchmark()
        benchmark.run_full_benchmark()
    else:
        print("Usage: python gpu_benchmark.py [--quick|--full]")
        print("  --quick: Quick GPU functionality test")
        print("  --full:  Complete benchmark suite")
