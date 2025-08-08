#!/usr/bin/env python3
"""
Setup script for RAG Clause Generator
"""

import os
import sys
import subprocess

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = ".env"
    if os.path.exists(env_file):
        print("✅ .env file already exists")
        return True
    
    print("\n🔑 Creating .env file...")
    env_content = """# API Keys for RAG Clause Generator
# Choose ONE of the following options:

# Option 1: Google Gemini (Recommended - Free tier available)
# Get your API key from: https://makersuite.google.com/app/apikey
GEMINI_API_KEY=your_gemini_api_key_here

# Option 2: OpenAI GPT-4 (Paid service)
# Get your API key from: https://platform.openai.com/api-keys
# OPENAI_API_KEY=your_openai_api_key_here

# Instructions:
# 1. Replace the placeholder with your actual API key
# 2. Comment out the option you're not using
# 3. Save the file
"""
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ .env file created")
        print("⚠️  Please edit .env file and add your API key before running the application")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def check_contract_files():
    """Check if contract files exist"""
    contract_dir = "full_contract_txt"
    if not os.path.exists(contract_dir):
        print(f"❌ Contract directory '{contract_dir}' not found")
        return False
    
    txt_files = [f for f in os.listdir(contract_dir) if f.endswith('.txt')]
    if not txt_files:
        print(f"❌ No .txt files found in '{contract_dir}' directory")
        return False
    
    print(f"✅ Found {len(txt_files)} contract files")
    return True

def main():
    """Main setup function"""
    print("🚀 RAG Clause Generator Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check contract files
    if not check_contract_files():
        print("\n⚠️  Please ensure you have contract .txt files in the 'full_contract_txt' directory")
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Create .env file
    create_env_file()
    
    print("\n🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Edit .env file and add your API key")
    print("2. Run: python app.py")
    print("3. Open http://localhost:5000 in your browser")
    print("\nFor API keys:")
    print("- Google Gemini (free): https://makersuite.google.com/app/apikey")
    print("- OpenAI GPT-4 (paid): https://platform.openai.com/api-keys")

if __name__ == "__main__":
    main() 