"""
Debug utility for DQN training

This script helps diagnose issues with the DQN implementation,
especially focusing on command execution and logging problems.
"""

import os
import sys
import subprocess
import argparse

def check_environment():
    """Check Python environment and dependencies"""
    print("\n===== CHECKING PYTHON ENVIRONMENT =====")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Check if running in virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("Running in virtual environment: Yes")
        print(f"Virtual env path: {sys.prefix}")
    else:
        print("Running in virtual environment: No")
    
    # Check required packages
    required_packages = ['torch', 'numpy', 'gymnasium', 'ale_py', 'matplotlib']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"Package {package} - Found")
        except ImportError:
            print(f"Package {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print("\nSome required packages are missing. Install them with:")
        print(f"pip install {' '.join(missing_packages)}")
    else:
        print("\nAll required packages are installed.")

def test_command_execution():
    """Test execution of main training commands"""
    print("\n===== TESTING COMMAND EXECUTION =====")
    
    commands = [
        ["python", "main.py", "-h"],                       # Help command
        ["python", "main.py", "train", "-h"],              # Training help
        ["python", "main.py", "evaluate", "-h"],           # Evaluation help
        ["python", "logger.py", "--check"],                # Logger check
        ["python", "debug_logs.py", "--guide"]             # Usage guide
    ]
    
    for cmd in commands:
        cmd_str = " ".join(cmd)
        print(f"\nExecuting: {cmd_str}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Command failed with error code {result.returncode}")
                print("Error output:")
                print(result.stderr)
            else:
                print("Command executed successfully")
                print("First few lines of output:")
                output_lines = result.stdout.split('\n')[:5]
                for line in output_lines:
                    print(f"  {line}")
                if len(output_lines) < len(result.stdout.split('\n')):
                    print("  ...")
        except Exception as e:
            print(f"Failed to execute command: {e}")
            
    # Test if the main command pattern works correctly
    print("\nVerifying correct command syntax:")
    test_commands = [
        "python main.py train --episodes 10",      # Correct format
        "python main.py --train",                  # Incorrect format
    ]
    
    for cmd_str in test_commands:
        print(f"\nTesting: {cmd_str}")
        try:
            result = subprocess.run(cmd_str.split(), capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ Command format is valid")
            else:
                print("✗ Command format is invalid")
                if "unrecognized arguments: --train" in result.stderr:
                    print("ERROR: Use 'python main.py train' not 'python main.py --train'")
                    print("The correct format is: python main.py [command] [options]")
        except Exception as e:
            print(f"Error testing command: {e}")

def check_disk_space():
    """Check disk space in the project directory"""
    print("\n===== CHECKING DISK SPACE =====")
    try:
        # Get project directory
        project_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Check disk space
        if sys.platform == "win32":
            # Windows
            import ctypes
            free_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                ctypes.c_wchar_p(project_dir), None, None, ctypes.pointer(free_bytes)
            )
            free_gb = free_bytes.value / (1024 ** 3)
            print(f"Free disk space: {free_gb:.2f} GB")
        else:
            # Unix-like
            import shutil
            total, used, free = shutil.disk_usage(project_dir)
            print(f"Total disk space: {total/(1024**3):.2f} GB")
            print(f"Used disk space: {used/(1024**3):.2f} GB")
            print(f"Free disk space: {free/(1024**3):.2f} GB")
        
        # Check for available space
        if free_gb < 10:
            print("WARNING: Less than 10GB of free space available.")
            print("DQN training may require significant disk space for logs and model checkpoints.")
    except Exception as e:
        print(f"Failed to check disk space: {e}")

def verify_file_permissions():
    """Check if we have permissions to write to directories"""
    print("\n===== VERIFYING FILE PERMISSIONS =====")
    
    # Make sure project directory is readable
    if not os.access(".", os.R_OK):
        print("WARNING: Cannot read from project directory")
    else:
        print("Project directory is readable")
    
    # Check write permission
    try:
        test_file = ".test_write_permission"
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        print("Project directory is writable")
    except Exception as e:
        print(f"WARNING: Cannot write to project directory: {e}")
    
    # Check common output directories
    for directory in ["./results", "./logs", "./models"]:
        if os.path.exists(directory):
            if os.access(directory, os.W_OK):
                print(f"Directory {directory} exists and is writable")
            else:
                print(f"WARNING: Directory {directory} exists but is not writable")
        else:
            print(f"Directory {directory} does not exist, will be created when needed")

def print_usage_guide():
    """Print a guide with the correct usage commands"""
    print("\n===== CORRECT USAGE GUIDE =====")
    print("To run the DQN training, use the following commands:")
    
    print("\n1. Train a new model:")
    print("   python main.py train [options]")
    print("   Examples:")
    print("   - python main.py train")
    print("   - python main.py train --episodes 1000 --render")
    print("   - python main.py train --cpu")
    
    print("\n2. Evaluate a trained model:")
    print("   python main.py evaluate [model_path] [options]")
    print("   Examples:")
    print("   - python main.py evaluate results/run_20230101-010101/models/final_model.pth --render")
    
    print("\n3. Compare multiple models:")
    print("   python main.py compare [model1] [model2] [...]")
    print("   Example:")
    print("   - python main.py compare models/model1.pth models/model2.pth")
    
    print("\n4. Visualize training results:")
    print("   python main.py visualize [results_file]")
    print("   Example:")
    print("   - python main.py visualize results/run_20230101-010101/logs/training_stats.pkl")
    
    print("\n5. Check logger functionality:")
    print("   python logger.py --check")
    print("   or")
    print("   python logger.py --test")

def main():
    """Main function to run debug checks"""
    parser = argparse.ArgumentParser(description="Debug and diagnostic tool for DQN training")
    parser.add_argument("--all", action="store_true", help="Run all checks")
    parser.add_argument("--env", action="store_true", help="Check Python environment")
    parser.add_argument("--cmd", action="store_true", help="Test command execution")
    parser.add_argument("--disk", action="store_true", help="Check disk space")
    parser.add_argument("--perm", action="store_true", help="Verify file permissions")
    parser.add_argument("--guide", action="store_true", help="Print usage guide")
    
    args = parser.parse_args()
    
    # If no specific checks requested, run all checks
    if not any([args.env, args.cmd, args.disk, args.perm, args.guide]):
        args.all = True
    
    print("====================================")
    print("  DQN TRAINING DEBUG UTILITY")
    print("====================================")
    
    if args.all or args.env:
        check_environment()
    
    if args.all or args.cmd:
        test_command_execution()
    
    if args.all or args.disk:
        check_disk_space()
    
    if args.all or args.perm:
        verify_file_permissions()
    
    if args.all or args.guide:
        print_usage_guide()
    
    print("\n====================================")
    print("Run the appropriate commands shown above to start training or evaluation.")
    print("If you encounter any issues, review the debug information provided.")
    print("====================================")

if __name__ == "__main__":
    main()
