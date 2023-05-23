import subprocess


def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    output, error = process.communicate()


run_command(
    "git clone https://github.com/open-mmlab/mmcv.git && cd mmcv && pip3 install -r requirements.txt"
)
