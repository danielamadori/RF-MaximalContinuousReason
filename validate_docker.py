#!/usr/bin/env python3
"""
Docker Configuration Validator
Tests Docker setup without requiring Docker to be installed
"""
import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if Path(filepath).exists():
        print(f"[OK] {description}: EXISTS")
        return True
    else:
        print(f"[ERROR] {description}: MISSING")
        return False

def check_dockerfile_syntax():
    """Basic Dockerfile syntax validation"""
    print("\n=== Dockerfile Syntax Check ===")

    try:
        with open('Dockerfile', 'r') as f:
            content = f.read()

        required_instructions = ['FROM', 'WORKDIR', 'COPY', 'RUN', 'CMD']
        found = []

        for instruction in required_instructions:
            if instruction in content:
                print(f"[OK] {instruction} instruction found")
                found.append(instruction)
            else:
                print(f"[ERROR] {instruction} instruction missing")

        return len(found) == len(required_instructions)
    except Exception as e:
        print(f"[ERROR] Error reading Dockerfile: {e}")
        return False

def check_supervisord_config():
    """Check supervisord configuration"""
    print("\n=== Supervisord Config Check ===")

    config_file = 'docker/supervisord.conf'
    try:
        with open(config_file, 'r') as f:
            content = f.read()

        checks = {
            'supervisord': '[supervisord]' in content,
            'redis': '[program:redis]' in content,
            'nodaemon': 'nodaemon=true' in content,
        }

        for name, result in checks.items():
            if result:
                print(f"[OK] {name} configured")
            else:
                print(f"[ERROR] {name} not configured")

        return all(checks.values())
    except Exception as e:
        print(f"[ERROR] Error reading {config_file}: {e}")
        return False

def main():
    print("="*60)
    print(" DOCKER CONFIGURATION VALIDATOR")
    print("="*60)
    print()

    # Check files
    print("=== Required Files ===")
    files_ok = all([
        check_file_exists('Dockerfile', 'Dockerfile'),
        check_file_exists('docker/supervisord.conf', 'Supervisord Config'),
        check_file_exists('run.bat', 'Windows Runner'),
        check_file_exists('run.sh', 'Linux/macOS Runner'),
        check_file_exists('requirements.txt', 'Python Requirements'),
    ])

    # Check Dockerfile syntax
    dockerfile_ok = check_dockerfile_syntax()

    # Check supervisord config
    supervisord_ok = check_supervisord_config()

    # Summary
    print()
    print("="*60)
    print(" VALIDATION SUMMARY")
    print("="*60)

    if files_ok and dockerfile_ok and supervisord_ok:
        print()
        print("[SUCCESS] All Docker configuration files are present and valid!")
        print()
        print("Docker setup is ready to use.")
        print()
        print("To use Docker:")
        print("  1. Install Docker Desktop")
        print("  2. Run: run.bat (Windows) or ./run.sh (Linux/macOS)")
        print()
        return 0
    else:
        print()
        print("[FAILED] Some Docker configuration issues found")
        print()
        print("Please fix the issues above before using Docker")
        print()
        return 1

if __name__ == '__main__':
    sys.exit(main())

