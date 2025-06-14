"""
Model Storage Utilities
Handles organized storage of AI models in models/<difficulty>/<date>/ structure
"""

import os
from datetime import datetime
from typing import Optional


def get_model_save_dir(difficulty: str, timestamp: Optional[str] = None, base_dir: str = "models") -> str:
    """
    Generate a model save directory following the organized structure.
    
    Args:
        difficulty: Game difficulty (beginner, intermediate, expert)
        timestamp: Optional timestamp string (default: current time)
        base_dir: Base models directory (default: "models")
        
    Returns:
        Full path to the model save directory
        
    Example:
        get_model_save_dir("beginner") -> "models/beginner/20241212_143022"
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    difficulty = difficulty.lower()
    if difficulty not in ["beginner", "intermediate", "expert"]:
        raise ValueError(f"Invalid difficulty: {difficulty}. Must be beginner, intermediate, or expert.")
    
    save_dir = os.path.join(base_dir, difficulty, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    return save_dir


def find_latest_model_dir(difficulty: str, base_dir: str = "models") -> Optional[str]:
    """
    Find the latest model directory for a given difficulty.
    
    Args:
        difficulty: Game difficulty (beginner, intermediate, expert)
        base_dir: Base models directory (default: "models")
        
    Returns:
        Path to the latest model directory or None if not found
    """
    difficulty = difficulty.lower()
    difficulty_dir = os.path.join(base_dir, difficulty)
    
    if not os.path.exists(difficulty_dir):
        return None
    
    # Get all timestamp directories
    timestamp_dirs = []
    for item in os.listdir(difficulty_dir):
        item_path = os.path.join(difficulty_dir, item)
        if os.path.isdir(item_path):
            # Validate timestamp format
            try:
                datetime.strptime(item, "%Y%m%d_%H%M%S")
                timestamp_dirs.append(item)
            except ValueError:
                continue
    
    if not timestamp_dirs:
        return None
    
    # Return the latest one (lexicographically sorted works for YYYYMMDD_HHMMSS)
    latest_timestamp = max(timestamp_dirs)
    return os.path.join(difficulty_dir, latest_timestamp)


def list_model_checkpoints(model_dir: str) -> list[str]:
    """
    List all model checkpoint files in a directory.
    
    Args:
        model_dir: Directory containing model files
        
    Returns:
        List of checkpoint file paths, sorted by episode number
    """
    if not os.path.exists(model_dir):
        return []
    
    checkpoints = []
    for file in os.listdir(model_dir):
        # Legacy format: dqn_episode_*.pth and dqn_final.pth
        if file.startswith("dqn_episode_") and file.endswith(".pth"):
            checkpoints.append(os.path.join(model_dir, file))
        elif file == "dqn_final.pth":
            checkpoints.append(os.path.join(model_dir, file))        # Enhanced format: foundation_checkpoint.pth, stabilization_checkpoint.pth, mastery_checkpoint.pth, etc.
        elif file.endswith("_checkpoint.pth"):
            checkpoints.append(os.path.join(model_dir, file))
        # Periodic checkpoints: foundation_checkpoint_ep1000.pth, etc.
        elif "_checkpoint_ep" in file and file.endswith(".pth"):
            checkpoints.append(os.path.join(model_dir, file))
        elif file == "best_model_checkpoint.pth":
            checkpoints.append(os.path.join(model_dir, file))
        elif file == "final_model.pth":
            checkpoints.append(os.path.join(model_dir, file))
    
    # Sort by phase priority and episode number (final comes last)
    def sort_key(path):
        filename = os.path.basename(path)
          # Enhanced format priority: Foundation < Stabilization < Mastery < Best/Final
        if filename == "foundation_checkpoint.pth":
            return 100
        elif filename == "stabilization_checkpoint.pth":
            return 200
        elif filename == "mastery_checkpoint.pth":
            return 300
        elif filename == "best_model_checkpoint.pth":
            return 900
        elif filename == "final_model.pth":
            return 1000
        
        # Periodic checkpoints: foundation_checkpoint_ep1000.pth, etc.
        elif "_checkpoint_ep" in filename:
            try:
                # Extract phase and episode number
                if filename.startswith("foundation_checkpoint_ep"):
                    base_priority = 100
                    episode = int(filename.replace("foundation_checkpoint_ep", "").replace(".pth", ""))
                elif filename.startswith("stabilization_checkpoint_ep"):
                    base_priority = 200
                    episode = int(filename.replace("stabilization_checkpoint_ep", "").replace(".pth", ""))
                elif filename.startswith("mastery_checkpoint_ep"):
                    base_priority = 300
                    episode = int(filename.replace("mastery_checkpoint_ep", "").replace(".pth", ""))
                else:
                    base_priority = 150  # Default for unknown phase
                    episode = 0
                
                # Within a phase, higher episode numbers have higher priority
                return base_priority + (episode / 100000)  # Small increment for episode progression
            except (ValueError, IndexError):
                return 150
        
        # Legacy format
        elif filename == "dqn_final.pth":
            return float('inf')  # Final model comes last
        
        # Legacy episode-based checkpoints
        try:
            episode = int(filename.split('_episode_')[1].split('.pth')[0])
            return episode
        except (IndexError, ValueError):
            return 0
    
    checkpoints.sort(key=sort_key)
    return checkpoints


def get_latest_checkpoint(difficulty: str, base_dir: str = "models") -> Optional[tuple[str, str]]:
    """
    Get the latest checkpoint file for a given difficulty.
    
    Args:
        difficulty: Game difficulty (beginner, intermediate, expert)
        base_dir: Base models directory (default: "models")
        
    Returns:
        Tuple of (checkpoint_directory, checkpoint_filename) or None if not found
    """
    latest_dir = find_latest_model_dir(difficulty, base_dir)
    if not latest_dir:
        return None
    
    checkpoints = list_model_checkpoints(latest_dir)
    if not checkpoints:
        return None
    
    latest_checkpoint = checkpoints[-1]
    checkpoint_filename = os.path.basename(latest_checkpoint)
    return (latest_dir, checkpoint_filename)


def get_legacy_model_directories() -> dict[str, list[str]]:
    """
    Find legacy model directories (pre-refactor structure) for migration.
    
    Returns:
        Dictionary mapping legacy directory names to their full paths
    """
    legacy_patterns = [
        "models_beginner_enhanced_v2",
        "models_beginner_enhanced_v2_parallel_resume",
        "models_intermediate_",
        "models_expert_",
        "models_beginner_"
    ]
    
    legacy_dirs = {}
    
    # Check current directory for legacy model folders
    for item in os.listdir("."):
        if os.path.isdir(item):
            for pattern in legacy_patterns:
                if item.startswith(pattern):
                    if "legacy" not in legacy_dirs:
                        legacy_dirs["legacy"] = []
                    legacy_dirs["legacy"].append(item)
                    break
    
    return legacy_dirs


def migrate_legacy_models():
    """
    Migrate legacy model directories to the new organized structure.
    This function can be called to help transition existing models.
    """
    legacy_dirs = get_legacy_model_directories()
    
    if not legacy_dirs.get("legacy"):
        print("✅ No legacy model directories found.")
        return
    
    print("🔄 Found legacy model directories:")
    for legacy_dir in legacy_dirs["legacy"]:
        print(f"   📁 {legacy_dir}")
    
    print("\n💡 To migrate these models, manually move them to the new structure:")
    print("   Example: models_beginner_enhanced_v2/ → models/beginner/20241212_143022/")
    print("   Use get_model_save_dir() function for new training runs.")


if __name__ == "__main__":
    # Test the utility functions
    print("🧪 Testing model storage utilities...")
    
    # Test directory generation
    test_dir = get_model_save_dir("beginner")
    print(f"✅ Generated save directory: {test_dir}")
    
    # Test finding latest (should be the one we just created)
    latest = find_latest_model_dir("beginner")
    print(f"✅ Latest model directory: {latest}")
    
    # Check for legacy models
    migrate_legacy_models()
    
    print("\n📖 Usage examples:")
    print("  get_model_save_dir('beginner') → models/beginner/20241212_143022")
    print("  find_latest_model_dir('intermediate') → models/intermediate/20241211_091045")
    print("  get_latest_checkpoint('expert') → models/expert/.../dqn_final.pth")
