import sys
import yaml
from pathlib import Path
from termcolor import colored
from typing import Dict

from libpy import Hypervision

class ConfigError(Exception):
    """Base exception for configuration errors"""

def load_config(config_path: str) -> Dict:
    """
    Load and validate configuration from YAML file
    Returns validated configuration dictionary
    Raises ConfigError on validation failures
    """
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_file, 'r') as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                error_msg = format_yaml_error(e, config_path)
                raise ValueError(error_msg) from e

        required_fields = [
            'screen_width', 'screen_height', 'activation_range',
            'confidence_threshold', 'nms_threshold', 'model_path',
            'enable_aim', 'display'
        ]
        validate_required_fields(config, required_fields)

        config['model_path'] = resolve_model_path(config['model_path'])
        
        return config

    except Exception as e:
        handle_config_error(e, config_path)
        raise ConfigError("Configuration validation failed") from e

def format_yaml_error(error: yaml.YAMLError, config_path: str) -> str:
    """Create detailed YAML error message"""
    if not hasattr(error, 'problem_mark'):
        return f"YAML parsing error: {str(error)}"
    
    mark = error.problem_mark
    return (
        f"YAML syntax error in {config_path}:\n"
        f"  → Line {mark.line + 1}, Column {mark.column + 1}\n"
        f"  → Problem: {error.problem}\n"
        "Common fixes:\n"
        "  - Check for missing quotes around values\n"
        "  - Verify indentation levels\n"
        "  - Use forward slashes in paths (/) instead of backslashes\n"
        "  - Remove special characters from unquoted strings"
    )

def validate_required_fields(config: Dict, fields: list):
    """Validate presence of required configuration fields"""
    missing = [field for field in fields if field not in config]
    if missing:
        raise ValueError(
            f"Missing required fields: {', '.join(missing)}"
        )

def resolve_model_path(model_path: str) -> str:
    """Convert relative path to absolute and validate existence"""
    path = Path(model_path)
    if not path.is_absolute():
        path = Path(__file__).parent / path
    
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    
    return str(path.resolve())

def handle_config_error(error: Exception):
    """Display appropriate error message for configuration issues"""
    if isinstance(error, FileNotFoundError):
        print(colored(f"\n[CONFIG ERROR] {str(error)}", "red"))
    elif isinstance(error, ValueError):
        if "YAML syntax error" in str(error):
            print(colored(f"\n[CONFIG ERROR] {str(error)}", "red"))
            print(colored("\nCheck your YAML formatting using:", "yellow"))
            print(colored("1. Online validators like yamllint.com", "yellow"))
            print(colored("2. IDE plugins with YAML support", "yellow"))
            print(colored("3. Double-check special characters", "yellow"))
        else:
            print(colored(f"\n[CONFIG ERROR] {str(error)}", "red"))
    else:
        print(colored(f"\n[CONFIG ERROR] Unexpected error: {str(error)}", "red"))

if __name__ == "__main__":
    try:
        config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
        config = load_config(config_path)
        
        print(colored("\n[CONFIG] Loaded configuration:", "cyan"))
        for key, value in config.items():
            print(f"  {key.ljust(20)}: {value}")
            
        HV = Hypervision(config)
        HV.run()
        
    except ConfigError:
        sys.exit(1)
    except Exception as e:
        print(colored(f"\n[FATAL] Unexpected error: {str(e)}", "red"))
        sys.exit(1)