#!/usr/bin/env python-real

from slicer.util import pip_install
import argparse
import sys
import os
import numpy as np
import platform
import urllib
import subprocess
import shutil





def InstallDependencies():
  # Install dependencies
  import platform 
  system = platform.system()
  print('Installing dependencies...')
  pip_install('--upgrade pip')
  pip_install('tqdm==4.64.0') # tqdm
  pip_install('pandas==1.4.2') # pandas
  #pip_install('--no-cache-dir torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html') # torch
  # pip_install('--no-cache-dir torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113')
  pip_install('torch==1.12.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113')
  pip_install('monai==0.7.0') # monai
  pip_install('fvcore==0.1.5.post20220504')
  pip_install('iopath==0.1.9')
  if system == "Linux":
    try:
      code_path = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/'))
      print(code_path)
      pip_install(f'{code_path}/_CrownSegmentationcli/pytorch3d-0.7.0-cp39-cp39-linux_x86_64.whl') # py39_cu113_pyt1120
    except:
      pip_install('--force-reinstall --no-deps --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1120/download.html')

  else:
    raise Exception('Module only works with Linux systems.')
    # pip_install("--force-reinstall git+https://github.com/facebookresearch/pytorch3d.git")


system = platform.system()
if system !="Windows" :
  if sys.argv[1] == '-1':
    InstallDependencies()
      


  else:
    # normal execution
    try:
      from tqdm import tqdm
    except ImportError:
      pip_install('tqdm==4.64.0')
      from tqdm import tqdm

    try:
      import pandas
    except ImportError:
      pip_install('pandas==1.4.2')

    try:
      import torch
      pyt_version_str=torch.__version__.split("+")[0].replace(".", "")
      version_str="".join([f"py3{sys.version_info.minor}_cu",torch.version.cuda.replace(".",""),f"_pyt{pyt_version_str}"])  
      if version_str != 'py39_cu113_pyt1120':
        raise ImportError
    except ImportError:
      # pip_install('--no-cache-dir torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113')
      pip_install('--force-reinstall torch==1.12.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113')
      import torch


    try:
      import pytorch3d
      if pytorch3d.__version__ != '0.7.0':
        raise ImportError
    except ImportError:
      InstallDependencies()
      

    try:
      import monai
      if monai.__version__ != '0.8.dev2143':
        raise ImportError
    except ImportError:
      pip_install('monai==0.7.0')


  try: 
    import pytorch_lightning
  except ImportError :
    #  pip_install('pytorch_lightning==1.7.7')
    pip_install('pytorch_lightning==2.1')
    import pytorch_lightning




  from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
  from torch.utils.data import DataLoader






  from __CrownSegmentation import (MonaiUNet, TeethDataset, UnitSurfTransform, Write, RemoveIslands, 
                    DilateLabel, ErodeLabel, Threshold, ConvertFDI)




def checkMiniconda():
    '''
    Check if miniconda3 is install 
    Return :
    default install path of miniconda3 
    bool if install
    '''
    print("je suis dans checkminiconda")
    user_home = os.path.expanduser("~")
    default_install_path = os.path.join(user_home, "miniconda3")
    return(os.path.exists(default_install_path),default_install_path)
  
def InstallConda(default_install_path):
    ''''
    Install miniconda3 on Windows
    Input : default install path of miniconda3 on the computer
    '''
    system = platform.system()
    machine = platform.machine()

    miniconda_base_url = "https://repo.anaconda.com/miniconda/"

    # Construct the filename based on the operating system and architecture
    if system == "Windows":
        if machine.endswith("64"):
            filename = "Miniconda3-latest-Windows-x86_64.exe"
        else:
            filename = "Miniconda3-latest-Windows-x86.exe"
    elif system == "Linux":
        if machine == "x86_64":
            filename = "Miniconda3-latest-Linux-x86_64.sh"
        else:
            filename = "Miniconda3-latest-Linux-x86.sh"
    else:
        raise NotImplementedError(f"Unsupported system: {system} {machine}")

    print(f"Selected Miniconda installer file: {filename}")

    miniconda_url = miniconda_base_url + filename
    print(f"Full download URL: {miniconda_url}")

    print(f"Default Miniconda installation path: {default_install_path}")

    path_exe = os.path.join(os.path.expanduser("~"), "tempo")
       
    os.makedirs(path_exe, exist_ok=True)
    # Define paths for the installer and conda executable
    path_installer = os.path.join(path_exe, filename)
    path_conda = os.path.join(default_install_path, "Scripts", "conda.exe")
    
    

    print(f"path_installer : {path_installer}")
    print(f"path_conda : {path_conda}")

    if not os.path.exists(default_install_path):
        os.makedirs(default_install_path)

        try:
            # Download the Anaconda installer
            urllib.request.urlretrieve(miniconda_url, path_installer)
            print("Installer downloaded successfully.")
            print("Installing Miniconda...")
            
            # Run the Anaconda installer with silent mode
            print("path_installer : ",path_installer)
            print("default_install_path : ",default_install_path)
            path_miniconda = os.path.join(default_install_path,"miniconda")

            # Commande pour une installation silencieuse avec Miniconda
            install_command = f'"{path_installer}" /InstallationType=JustMe /AddToPath=1 /RegisterPython=0 /S /D={default_install_path}'

            # Exécutez la commande d'installation
            subprocess.run(install_command, shell=True)

            subprocess.run(f"{path_conda} init cmd.exe", shell=True)
            print("Miniconda installed successfully.")
            
            try:
                shutil.rmtree(path_exe)
                print(f"Dossier {path_exe} et son contenu ont été supprimés avec succès.")
            except Exception as e:
                print(f"Une erreur s'est produite lors de la suppression du dossier : {str(e)}")
                return True
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return False
    else:
        print("Unsupported system. This code is intended for Windows.")
        return False
  

def main(args):
    print('strat crown segmentation')
    print(f' args {args}')


    sytem = platform.system()
    #WINDOWS ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    if system == "Windows" :
        miniconda,default_install_path = checkMiniconda()
        if not miniconda : 
          print("appelle InstallConda")
          # write_txt("Installation of miniconda3, this task can take a few minutes")
          InstallConda(default_install_path)
    
        current_file_path = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_file_path)
        path_func_miniconda = os.path.join(current_directory,'utils_windows', 'first.py') #Next files to call

        python_path = os.path.join(default_install_path,"python") #python path in miniconda3
        #command to call first.py with python in miniconda3 on windows and give it the argument parser = argparse.ArgumentParser()

        command_to_execute = [python_path,path_func_miniconda,"setup",default_install_path,args.input,args.output,str(args.subdivision_level),str(args.resolution),args.model,args.predictedId,str(args.sepOutputs),str(args.chooseFDI),args.logPath]  
        print(f"command_to_execute in slicer : {command_to_execute}")

        env = dict(os.environ)
        if 'PYTHONPATH' in env:
            del env['PYTHONPATH']
        if 'PYTHONHOME' in env:
            del env['PYTHONHOME']
 
        
        print("command to execute slicer : ",command_to_execute)



        result = subprocess.run(command_to_execute, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,env=env)


        if result.returncode != 0:
            print(f"Error creating the environment. Return code: {result.returncode}")
            print("result.stdout : ","*"*150)
            print(result.stdout)
            print("result.stderr : ","*"*150)
            print(result.stderr)
        else:
            print(result.stdout)
            print("Environment created successfully.")

        print("%"*300)

    #END WINDOWS ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    else : 
      with open(args.logPath,'w') as log_f:
          # clear log file
          log_f.truncate(0)


      class_weights = None
      out_channels = 34

      model = MonaiUNet( out_channels = out_channels, class_weights=class_weights, image_size=args.resolution, subdivision_level=args.subdivision_level)

      model.model.module.load_state_dict(torch.load(args.model))


      ds = TeethDataset(args.input, transform=UnitSurfTransform())

      dataloader = DataLoader(ds, batch_size=1, num_workers=4, persistent_workers=True, pin_memory=True)
      

      device = torch.device('cuda')
      model.to(device)
      model.eval()

      softmax = torch.nn.Softmax(dim=2)

      with torch.no_grad():

          for idx, batch in enumerate(dataloader):

              V, F, CN = batch

              V = V.cuda(non_blocking=True)
              F = F.cuda(non_blocking=True)
              CN = CN.cuda(non_blocking=True).to(torch.float32)

              x, X, PF = model((V, F, CN))
              x = softmax(x*(PF>=0))

              P_faces = torch.zeros(out_channels, F.shape[1]).to(device)
              V_labels_prediction = torch.zeros(V.shape[1]).to(device).to(torch.int64)

              PF = PF.squeeze()
              x = x.squeeze()

              for pf, pred in zip(PF, x):
                  P_faces[:, pf] += pred

              P_faces = torch.argmax(P_faces, dim=0)

              faces_pid0 = F[0,:,0]
              V_labels_prediction[faces_pid0] = P_faces

              surf = ds.getSurf(idx)

              V_labels_prediction = numpy_to_vtk(V_labels_prediction.cpu().numpy())
              V_labels_prediction.SetName(args.predictedId)
              surf.GetPointData().AddArray(V_labels_prediction)


              #Post Process
              RemoveIslands(surf,V_labels_prediction,33,500, ignore_neg1=True)
              for label in tqdm(range(out_channels),desc= 'Remove island'):
                  RemoveIslands(surf,V_labels_prediction, label, 200, ignore_neg1=True)




              for label in tqdm(range(1,out_channels),desc= 'Closing operation'):
                  DilateLabel(surf,V_labels_prediction, label, iterations=2, dilateOverTarget=False, target = None)
                  ErodeLabel(surf,V_labels_prediction, label, iterations=2, target=None)


              if args.chooseFDI :
                  surf = ConvertFDI(surf,args.predicteId)
                  gum_label = 0
              else :
                  gum_label = 33




              output_fn = os.path.join(args.output, ds.getName(idx))

              output_dir = os.path.dirname(output_fn)

              if(not os.path.exists(output_dir)):
                  os.makedirs(output_dir)




              if args.sepOutputs:
                  # Isolate each label
                  surf_point_data = surf.GetPointData().GetScalars(args.predictedId) 
                  labels = np.unique(surf_point_data)
                  out_basename = output_fn[:-4]
                  for label in tqdm(labels, desc = 'Isolating labels'):
                      thresh_label = Threshold(surf, args.predictedId ,label-0.5,label+0.5)
                      if label != gum_label:
                          Write(thresh_label,f'{out_basename}_id_{label}.vtk',print_out=False) 
                      else:
                      # gum
                          Write(thresh_label,f'{out_basename}_gum.vtk',print_out=False) 
                  # all teeth 
                  no_gum = Threshold(surf, args.predictedId ,gum_label-0.5,gum_label+0.5,invert=True)
                  Write(no_gum,f'{out_basename}_all_teeth.vtk',print_out=False)


              Write(surf , output_fn, print_out=False)

              with open(args.logPath,'r+') as log_f :
                log_f.write(str(idx))





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input',type=str)
    parser.add_argument('output',type=str)
    parser.add_argument('subdivision_level',type = int)
    parser.add_argument('resolution',type=int)
    parser.add_argument('model',type=str)
    parser.add_argument('predictedId',type=str)
    parser.add_argument('sepOutputs',type=int)
    parser.add_argument('chooseFDI',type=int)
    parser.add_argument('logPath',type=str)

    args = parser.parse_args()
    main(args)

