#!/usr/bin/env python3
"""
Model Switch Utility for FPL Team Picker

Quick utility to switch between ML and rule-based XP models.
"""

import json
from pathlib import Path

def switch_to_ml_model():
    """Switch to ML XP model"""
    config_path = Path("config_example.json")
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Ensure structure exists
    if "xp_model" not in config:
        config["xp_model"] = {}
    
    # Set ML model configuration
    config["xp_model"]["use_ml_model"] = True
    config["xp_model"]["ml_ensemble_rule_weight"] = 0.2
    config["xp_model"]["ml_min_training_gameweeks"] = 3
    config["xp_model"]["ml_training_gameweeks"] = 5
    config["xp_model"]["ml_position_min_samples"] = 30
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ Switched to ML XP model")
    print("   - Ridge regression with position-specific models")
    print("   - 80% ML + 20% rule-based ensemble")
    print("   - Requires 3+ gameweeks of training data")

def switch_to_rule_model():
    """Switch to rule-based XP model"""
    config_path = Path("config_example.json")
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Ensure structure exists
    if "xp_model" not in config:
        config["xp_model"] = {}
    
    # Set rule-based model configuration
    config["xp_model"]["use_ml_model"] = False
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ Switched to rule-based XP model")
    print("   - Traditional form-weighted calculations")
    print("   - Works with minimal historical data")
    print("   - Reliable and interpretable")

def show_current_config():
    """Show current model configuration"""
    config_path = Path("config_example.json")
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        xp_config = config.get("xp_model", {})
        use_ml = xp_config.get("use_ml_model", True)
        
        print(f"📊 Current XP Model: {'ML Model' if use_ml else 'Rule-Based Model'}")
        
        if use_ml:
            ensemble_weight = xp_config.get("ml_ensemble_rule_weight", 0.2)
            min_gw = xp_config.get("ml_min_training_gameweeks", 3)
            train_gw = xp_config.get("ml_training_gameweeks", 5)
            min_samples = xp_config.get("ml_position_min_samples", 30)
            
            print(f"   - Ensemble: {(1-ensemble_weight)*100:.0f}% ML + {ensemble_weight*100:.0f}% Rule-based")
            print(f"   - Training: {train_gw} gameweeks (min {min_gw})")
            print(f"   - Position models: {min_samples}+ samples required")
        else:
            form_weight = xp_config.get("form_weight", 0.7)
            form_window = xp_config.get("form_window", 5)
            print(f"   - Form weight: {form_weight*100:.0f}% recent + {(1-form_weight)*100:.0f}% season")
            print(f"   - Form window: {form_window} gameweeks")
    else:
        print("⚠️ No configuration file found (using defaults)")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("FPL Model Switch Utility")
        print("========================")
        print("")
        show_current_config()
        print("")
        print("Usage:")
        print("  python switch_model.py ml      # Switch to ML model")
        print("  python switch_model.py rule    # Switch to rule-based model")
        print("  python switch_model.py status  # Show current configuration")
        print("")
        print("After switching, restart the gameweek manager to use the new model.")
    elif sys.argv[1] == "ml":
        switch_to_ml_model()
    elif sys.argv[1] == "rule":
        switch_to_rule_model()
    elif sys.argv[1] == "status":
        show_current_config()
    else:
        print(f"Unknown command: {sys.argv[1]}")
        print("Use: ml, rule, or status")
