#!/usr/bin/env python3
"""
Final execution script for Mumbai Airport Flight Scheduling Optimizer
Honeywell Hackathon 2024

This script ensures all components are properly set up and launches the application.
"""

import os
import sys
import subprocess
import platform
import importlib
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Error: Python 3.7 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_required_files():
    """Check if all required files exist"""
    required_files = {
        'app.py': 'Main Streamlit application',
        'src/data_processing.py': 'Data processing module',
        'src/analysis.py': 'Flight analysis module', 
        'src/ml_models.py': 'Machine learning models',
        'src/nlp_interface.py': 'NLP query interface'
    }
    
    missing_files = []
    
    for file_path, description in required_files.items():
        if not os.path.exists(file_path):
            missing_files.append(f"{file_path} ({description})")
        else:
            print(f"✅ Found: {file_path}")
    
    if missing_files:
        print("\n❌ Missing required files:")
        for file_name in missing_files:
            print(f"   - {file_name}")
        return False
    
    # Create __init__.py if it doesn't exist
    init_file = Path('src/__init__.py')
    if not init_file.exists():
        init_file.touch()
        print("✅ Created: src/__init__.py")
    
    return True

def install_requirements():
    """Install required packages"""
    requirements = [
        'streamlit>=1.28.0',
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'plotly>=5.15.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'scikit-learn>=1.3.0',
        'scipy>=1.11.0',
        'openpyxl>=3.1.0'
    ]
    
    # Optional ML/NLP packages
    optional_requirements = [
        'transformers>=4.30.0',
        'torch>=2.0.0',
        'tokenizers'
    ]
    
    print("📦 Installing required packages...")
    
    failed_packages = []
    
    # Install core requirements
    for package in requirements:
        try:
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', package],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"✅ Installed: {package.split('>=')[0]}")
        except subprocess.CalledProcessError:
            failed_packages.append(package)
            print(f"❌ Failed: {package.split('>=')[0]}")
    
    # Try to install optional packages
    print("\n🎯 Installing optional ML/NLP packages...")
    for package in optional_requirements:
        try:
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', package],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"✅ Installed: {package.split('>=')[0]}")
        except subprocess.CalledProcessError:
            print(f"⚠️ Optional package failed: {package.split('>=')[0]}")
    
    if failed_packages:
        print(f"\n⚠️ Warning: {len(failed_packages)} packages failed to install")
        print("The application may still work with reduced functionality")
    
    return len(failed_packages) == 0

def check_data_files():
    """Check for data files and create sample if needed"""
    data_locations = [
        'Flight_Data.xlsx',
        'data/Flight_Data.xlsx'
    ]
    
    data_found = False
    for location in data_locations:
        if os.path.exists(location):
            print(f"✅ Found data file: {location}")
            data_found = True
            break
    
    if not data_found:
        print("⚠️ No Flight_Data.xlsx found - application will use sample data")
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        print("✅ Created data directory")
    
    return True

def test_imports():
    """Test if all required modules can be imported"""
    test_modules = {
        'streamlit': 'Streamlit web framework',
        'pandas': 'Data manipulation',
        'numpy': 'Numerical computing',
        'plotly': 'Interactive visualizations',
        'sklearn': 'Machine learning'
    }
    
    print("🧪 Testing module imports...")
    
    failed_imports = []
    
    for module, description in test_modules.items():
        try:
            importlib.import_module(module)
            print(f"✅ {module}: {description}")
        except ImportError:
            failed_imports.append(f"{module} ({description})")
            print(f"❌ {module}: {description}")
    
    # Test custom modules
    sys.path.append('src')
    custom_modules = {
        'data_processing': 'Flight data processing',
        'analysis': 'Flight analysis',
        'ml_models': 'Machine learning models',
        'nlp_interface': 'NLP interface'
    }
    
    for module, description in custom_modules.items():
        try:
            importlib.import_module(module)
            print(f"✅ {module}: {description}")
        except ImportError as e:
            failed_imports.append(f"{module} ({description})")
            print(f"❌ {module}: {description} - {str(e)[:50]}...")
    
    if failed_imports:
        print(f"\n❌ Import failures: {len(failed_imports)}")
        return False
    
    return True

def launch_application():
    """Launch the Streamlit application"""
    print("\n🚀 Launching Mumbai Airport Flight Scheduling Optimizer...")
    print("📱 The application will open in your default web browser")
    print("🌐 URL: http://localhost:8501")
    print("\n" + "="*60)
    print("🎯 HONEYWELL HACKATHON - FLIGHT SCHEDULING OPTIMIZER")
    print("="*60)
    
    try:
        # Launch Streamlit app
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'app.py'])
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching application: {e}")
        print("\nTry running manually: streamlit run app.py")

def main():
    """Main execution function"""
    print("🛫 Mumbai Airport Flight Scheduling Optimizer")
    print("🏆 Honeywell Hackathon 2024")
    print("=" * 50)
    
    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)
    
    print("\n" + "="*50)
    
    # Step 2: Check required files
    print("📁 Checking required files...")
    if not check_required_files():
        print("\n❌ Setup incomplete - missing required files")
        print("Make sure all Python files are in the correct directories")
        sys.exit(1)
    
    print("\n" + "="*50)
    
    # Step 3: Install requirements
    print("📦 Installing dependencies...")
    install_success = install_requirements()
    
    print("\n" + "="*50)
    
    # Step 4: Check data files
    print("📊 Checking data files...")
    check_data_files()
    
    print("\n" + "="*50)
    
    # Step 5: Test imports
    print("🧪 Testing module imports...")
    if not test_imports():
        print("\n⚠️ Some modules failed to import")
        print("The application may have reduced functionality")
        
        user_input = input("\nContinue anyway? (y/n): ")
        if user_input.lower() != 'y':
            sys.exit(1)
    
    print("\n" + "="*50)
    print("✅ Setup completed successfully!")
    print("\n🎯 Ready to launch the application!")
    
    # Step 6: Launch application
    user_input = input("\nPress Enter to launch the application (or 'q' to quit): ")
    if user_input.lower() != 'q':
        launch_application()
    else:
        print("\n👋 Setup completed. Run 'streamlit run app.py' to start the application manually.")

if __name__ == "__main__":
    main()