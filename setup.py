import subprocess
import os

clone_dir = "wave-unet"
repo_url = "https://github.com/Wieland3/Wave-U-net-TF2.git"


def clone_repo():
    if os.path.isdir(clone_dir):
        print("The Wave Unet Repo already has been cloned.")
    else:
        try:
            subprocess.check_call(['git', 'clone', repo_url, clone_dir])
            print(f"Repository cloned into ./{clone_dir}")
        except subprocess.CalledProcessError as error:
            print(f"Failed to clone repository: {error}")


if __name__ == "__main__":
    clone_repo()
