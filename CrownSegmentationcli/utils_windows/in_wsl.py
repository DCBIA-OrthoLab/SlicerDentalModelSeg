import sys
import os
import time
import subprocess
import rpyc

def call(name,args):
    '''
    Call the server that will run the code of ALI_IOS on the new environnement
    '''
    home_dir = os.path.expanduser("~")
    default_install_path = "~/miniconda3"
    path_activate = "~/miniconda3/bin/activate"
    python_path_env = f"{home_dir}/miniconda3/envs/{name}/bin/python"
    

    current_file_path = os.path.abspath(__file__)

    current_directory = os.path.dirname(current_file_path)

    path_server = os.path.join(current_directory, 'server.py')
    
    path_activate = f"{home_dir}/miniconda3/bin/activate"
   
    command = f"/bin/bash -c 'source {path_activate} {name} && {python_path_env} {path_server} \"{sys.argv[1]}\" \"{sys.argv[2]}\" \"{sys.argv[3]}\" \"{sys.argv[4]}\" \"{sys.argv[5]}\" \"{sys.argv[6]}\" \"{sys.argv[7]}\" \"{sys.argv[8]}\" \"{sys.argv[9]}\"'"
    
    print("command in wsl : ", command)
    result = subprocess.run(command,shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
    if result.returncode != 0:
            print(f"Error processing the code in server3t. Return code: {result.returncode}")
            print("result.stdout : ","1"*150)
            print(result.stdout)
            print("result.stderr : ","1"*150)
            print(result.stderr)
    else:
        print(result.stdout)
        print("Process run succesfully")




def windows_to_linux_path(windows_path):
    '''
    Convert a windows path to a path that wsl can read
    '''
    windows_path = windows_path.strip()

    path = windows_path.replace('\\', '/')

    if ':' in path:
        drive, path_without_drive = path.split(':', 1)
        path = "/mnt/" + drive.lower() + path_without_drive

    return path




if __name__ == "__main__":
    if len(sys.argv) > 3 :

        args = {
       "input": sys.argv[1],
        "output": sys.argv[2],
        "subdivision_level": sys.argv[3],
        "resolution": sys.argv[4],
        "model": sys.argv[5],
        "predictedId": sys.argv[6],

        "sepOutputs": sys.argv[7],
        "chooseFDI": sys.argv[8],
        "logPath": sys.argv[9]
    }
        
        
        name = sys.argv[10]
        
    else:
        args = []
        name = "CrownSegmentationCli"
        
    call(name, args)