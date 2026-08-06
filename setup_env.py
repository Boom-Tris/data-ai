import subprocess
import sys

def install_packages(packages, index_url=None):
    # ใช้ sys.executable เพื่อให้มั่นใจว่าจะใช้ pip ของ Python 3.11 ตัวปัจจุบันที่กำลังรันไฟล์นี้อยู่
    cmd = [sys.executable, "-m", "pip", "install"] + packages
    if index_url:
        cmd.extend(["--index-url", index_url])
    
    print(f">>> Running: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
        print("+++ Installation Successful +++\n")
    except subprocess.CalledProcessError as e:
        print(f"--- Error during installation: {e} ---\n")
        sys.exit(1)

def main():
    print("===================================================")
    print("  Installing Dependencies for HR Typhoon (Local)")
    print("  Target: CUDA 12.1 (RTX 3050) / Python 3.11")
    print("===================================================\n")

    print("[Step 1] Installing PyTorch with CUDA 12.1...")
    pytorch_pkgs = ["torch", "torchvision", "torchaudio"]
    install_packages(pytorch_pkgs, index_url="https://download.pytorch.org/whl/cu121")

    print("[Step 2] Installing additional AI libraries...")
    ai_pkgs = [
        "bitsandbytes", "transformers", "peft", "accelerate", 
        "datasets", "matplotlib", "seaborn", "pythainlp", 
        "python-dotenv", "pandas", "scikit-learn"
    ]
    install_packages(ai_pkgs)

    print("===================================================")
    print("  Setup Complete!")
    print("  Now you can run: python check_env.py")
    print("===================================================")

if __name__ == "__main__":
    main()
