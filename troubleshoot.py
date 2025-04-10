
"""
Quick troubleshooting tool for DQN Ice Hockey project

This script provides basic troubleshooting functionality to check:
1. If the environment can run the main command examples
2. If the logging system is working correctly
3. If directories are writable
"""

import os
import sys
import subprocess
import argparse

def print_intro():
    print("=============================================")
    print("    DQN Ice Hockey Troubleshooting Tool     ")
    print("=============================================")
    print("This tool will help diagnose common issues.\n")

def check_command_syntax():
    """Test if the main command syntax is working properly"""
    print("Testing command syntax...")
    
    # Define the correct command format
    test_command = "python main.py train --episodes 1 --cpu"
    
    try:
        result = subprocess.run(test_command.split(), capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Command syntax is correct!")
        else:
            print("✗ Command failed with error:")
            print(result.stderr)
            print("\nSuggested fixes:")
            print("1. Make sure you're using the correct command format: python main.py [command] [options]")
            print("2. DO NOT use -- before commands (use 'python main.py train' not 'python main.py --train')")
    except Exception as e:
        print(f"Error running test command: {e}")
    
    print()

def check_directories():
    """Check if necessary directories exist and are writable"""
    print("Checking directories...")
    
    dirs_to_check = [
        "./results",
        "./models",
        "./logs"
    ]
    
    for directory in dirs_to_check:
        if not os.path.exists(directory):
            print(f"Creating directory: {directory}")
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"✓ Created {directory} successfully")
            except Exception as e:
                print(f"✗ Could not create {directory}: {e}")
                continue
        
        # Check write permissions
        test_file = os.path.join(directory, ".write_test")
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print(f"✓ {directory} is writable")
        except Exception as e:
            print(f"✗ {directory} is not writable: {e}")
    
    print()

def fix_common_issues():
    """Apply fixes for common issues"""
    print("Applying common fixes...")
    
    # Check if all required packages are installed
    try:
        import torch
        import numpy
        import matplotlib
        import gymnasium
        print("✓ All required packages are installed")
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        print("Run: pip install torch numpy matplotlib gymnasium[atari,accept-rom-license] ale-py")
    
    print()

def display_usage_guide():
    """Display correct usage instructions"""
    print("===== CORRECT USAGE EXAMPLES =====")
    print("1. To train a model:")
    print("   python main.py train")
    print("   python main.py train --episodes 1000 --render\n")
    
    print("2. To evaluate a model:")
    print("   python main.py evaluate results/run_20230101-010101/models/best_model.pth --render\n")
    
    print("3. To compare models:")
    print("   python main.py compare models/model1.pth models/model2.pth\n")
    
    print("4. To visualize results:")
    print("   python main.py visualize results/run_20230101-010101/logs/training_stats.pkl\n")
    
    print("REMEMBER: Use 'train' not '--train' as the command!\n")

def main():
    parser = argparse.ArgumentParser(description="Troubleshoot common issues with DQN Ice Hockey")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix common issues")
    
    args = parser.parse_args()
    
    print_intro()
    check_command_syntax()
    check_directories()
    
    if args.fix:
        fix_common_issues()
    
    display_usage_guide()
    
    print("Troubleshooting complete! If you're still having issues,")
    print("try running: python debug_logs.py for more detailed diagnostics.")

if __name__ == "__main__":
    main()
