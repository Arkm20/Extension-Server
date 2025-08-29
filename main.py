import os
import sys
import json
import subprocess

# --- Configuration ---
MODULES_DIR = "modules"
EXTENSIONS_DIR = "extensions"
FRONTEND_DIR = "animex"
FRONTEND_REPO = "https://github.com/Arkm20/animex.git"
REQUIREMENTS_FILE = "requirements.txt"

def run_command(command, cwd=None):
    """Runs a command and prints its output in real-time."""
    print(f"Running command: {' '.join(command)}")
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        for line in process.stdout:
            print(line, end='')
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)
    except FileNotFoundError:
        print(f"Error: Command not found: {command[0]}")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with exit code {e.returncode}")
        sys.exit(e.returncode)

def install_dependencies():
    """
    Installs dependencies from requirements.txt and discovers additional
    requirements from modules and extensions.
    """
    print("--- Installing dependencies ---")
    
    all_requirements = set()
    
    # 1. Base requirements from requirements.txt
    if os.path.exists(REQUIREMENTS_FILE):
        with open(REQUIREMENTS_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    all_requirements.add(line)
    
    # 2. Discover module requirements
    if os.path.isdir(MODULES_DIR):
        for root, _, files in os.walk(MODULES_DIR):
            for file in files:
                if file.endswith(".module"):
                    module_path = os.path.join(root, file)
                    try:
                        with open(module_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            meta_str, _, _ = content.partition("\n---\n")
                            if meta_str:
                                meta = json.loads(meta_str)
                                for req in meta.get("requirements", []):
                                    all_requirements.add(req)
                    except Exception as e:
                        print(f"Warning: Could not parse module {module_path}: {e}")

    # 3. Discover extension requirements
    if os.path.isdir(EXTENSIONS_DIR):
        for root, _, files in os.walk(EXTENSIONS_DIR):
            if "package.json" in files:
                pkg_path = os.path.join(root, "package.json")
                try:
                    with open(pkg_path, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                        for req in meta.get("requirements", []):
                            all_requirements.add(req)
                except Exception as e:
                    print(f"Warning: Could not parse package.json in {root}: {e}")
    
    # 4. Install all unique requirements
    if not all_requirements:
        print("No requirements found to install.")
        return
        
    print(f"Found {len(all_requirements)} unique requirements to install.")
    
    for req in sorted(list(all_requirements)):
        print(f"Installing {req}...")
        run_command([sys.executable, "-m", "pip", "install", req])
    
    print("--- Dependencies installed successfully ---")

def setup_frontend():
    """Clones or updates the frontend repository."""
    print("--- Setting up frontend ---")
    if os.path.isdir(os.path.join(FRONTEND_DIR, ".git")):
        print(f"'{FRONTEND_DIR}' exists. Pulling latest changes...")
        run_command(["git", "pull"], cwd=FRONTEND_DIR)
    else:
        print(f"'{FRONTEND_DIR}' not found. Cloning repository...")
        run_command(["git", "clone", FRONTEND_REPO, FRONTEND_DIR])
    print("--- Frontend setup complete ---")

def main():
    """Main build execution for Vercel."""
    print("Starting Vercel build process...")
    install_dependencies()
    print("Vercel build process finished successfully.")

if __name__ == "__main__":
    main()
