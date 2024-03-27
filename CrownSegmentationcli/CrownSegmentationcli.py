#!/usr/bin/env python-real



import sys
import os
import argparse
import platform
import subprocess
from pathlib import Path


def check_environment_wsl():
      '''
      check if the file is running into wsl
      '''
      try:
            with open('/proc/version', 'r') as file:
                  content = file.read().lower()
            if 'microsoft' in content or 'wsl' in content:
                  return True
            else:
                  return False
      except FileNotFoundError:
            return False


  

def main(args):
    print('start crown segmentation cli')
    
    if platform.system()=="Linux" and not check_environment_wsl():
      print("_"*25,"RUN_IN_LINUX","_"*25)
      
      command = [args.dentalmodelseg_path, "--out",args.out, "--overwrite", args.overwrite, "--crown_segmentation", args.crown_segmentation, "--array_name", args.array_name, "--fdi", args.fdi, "--suffix", args.suffix]
      print("command : ",command)
      if args.surf != "None":
            command.append("--surf")
            command.append(args.surf)
      if args.input_csv != "None":
            command.append("--csv")
            command.append(args.input_csv)
      if args.model!="latest":
            command.append("--model")
            command.append(args.model)
      if args.vtk_folder!="None":
            command.append("--vtk_folder")
            command.append(args.vtk_folder)
            
      print("command : ",command)
      
      result = subprocess.run(command,stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
      print("Output : ",result.stdout)
      print("Error : ",result.stderr)
      
      
      print("_"*25,"______","_"*25)
          
    elif check_environment_wsl() :
      print("_"*25,"RUN_IN_WSL","_"*25)

      
      command = [args.dentalmodelseg_path, "--out",windows_to_linux_path(args.out), "--overwrite", args.overwrite, "--crown_segmentation", args.crown_segmentation, "--array_name", args.array_name, "--fdi", args.fdi, "--suffix", args.suffix]
      if args.surf != "None":
            command.append("--surf")
            command.append(windows_to_linux_path(args.surf))
      if args.input_csv != "None":
            command.append("--csv")
            command.append(windows_to_linux_path(args.input_csv))
      if args.model!="latest":
            command.append("--model")
            command.append(windows_to_linux_path(args.model))
      if args.vtk_folder!="None":
            command.append("--vtk_folder")
            command.append(windows_to_linux_path(args.vtk_folder))
      
      print("command : ",command)
      subprocess.run(command)


def windows_to_linux_path(windows_path):
      '''
      Convert a windows path to a wsl path
      '''
      windows_path = windows_path.strip()

      path = windows_path.replace('\\', '/')

      if ':' in path:
          drive, path_without_drive = path.split(':', 1)
          path = "/mnt/" + drive.lower() + path_without_drive

      return path




if __name__ == '__main__':
    print("Starting crownsegmentation cli")
    parser = argparse.ArgumentParser()
    parser.add_argument('surf',type=str)
    parser.add_argument('input_csv',type = str)
    parser.add_argument('out',type=str)
    parser.add_argument('overwrite',type=str)
    parser.add_argument('model',type=str)
    parser.add_argument('crown_segmentation',type=str)
    parser.add_argument('array_name',type=str)
    parser.add_argument('fdi',type=str)
    parser.add_argument('suffix',type=str)
    parser.add_argument('vtk_folder',type=str)
    parser.add_argument('dentalmodelseg_path',type=str)
    args = parser.parse_args()
    print("args : ",args)
    main(args)

