#!/usr/bin/env python-real



import sys
import os
import argparse
import platform
import subprocess

from pathlib import Path


# from CondaSetUp import  CondaSetUpCall,CondaSetUpCallWsl



  

def main(args):
    print('start crown segmentation cli')
    model = args.model
    if args.model == "latest":
      model = None
    else :
      if platform.system()=="Linux":
        model=model
      else :
        model = windows_to_linux_path(model)
    
    if platform.system()=="Linux":
      print("bonjour")
      print("_"*25,"TEST_1","_"*25)
      command = [f'dentalmodelseg --vtk \"{args.input_vtk}\" --stl \"{args.input_stl}\" --csv \"{args.input_csv}\" --out \"{args.out}\" --overwrite \"{args.overwrite}\" --model \"{model}\" --crown_segmentation \"{args.crown_segmentation}\" --array_name \"{args.array_name}\" --fdi \"{args.fdi}\" --suffix \"{args.suffix}\" --vtk_folder \"{args.vtk_folder}\"']
      
      path_dentalmodelseg = "/home/luciacev/APP/Slicer-5.6.1-linux-amd64/lib/Python/lib/python3.9/site-packages/shapeaxi/dental_model_seg.py"
      # print(f"CE PATH {path_dentalmodelseg} EST UN FICHIER ?  : {Path(path_dentalmodelseg).is_file()}")
      command = [f'{path_dentalmodelseg} --csv {args.input_csv} --out {args.out} --overwrite {0} --model {model} --crown_segmentation {0} --array_name {args.array_name} --fdi {args.fdi} --suffix {args.suffix} --vtk_folder {args.vtk_folder}']
      # command = [dentalmodelseg --vtk args.input_vtk --stl args.input_stl --csv args.input_csv --out args.out --overwrite args.overwrite --model model --crown_segmentation args.crown_segmentation --array_name args.array_name --fdi args.fdi --suffix args.suffix --vtk_folder args.vtk_folder]
      print("command : ",command)
      result = subprocess.run(command,stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
      print("Output : ",result.stdout)
      print("Error : ",result.stderr)
      
      # print("_"*25,"TEST_2","_"*25)
      # command = [f'dentalmodelseg --vtk \"{args.input_vtk}\" --stl \"{args.input_stl}\" --csv \"{args.input_csv}\" --out \"{args.out}\" --overwrite \"{args.overwrite}\" --model \"{model}\" --crown_segmentation \"{args.crown_segmentation}\" --array_name \"{args.array_name}\" --fdi \"{args.fdi}\" --suffix \"{args.suffix}\" --vtk_folder \"{args.vtk_folder}\"']
      # result = subprocess.run(command,stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
      # print("Output : ",result.stdout)
      # print("Error : ",result.stderr)
      
      print("_"*25,"______","_"*25)
          
    elif platform.system()=="Windows":
      print("*"*150)
      command = [f'dentalmodelseg --vtk \"{windows_to_linux_path(args.input_vtk)}\" --stl \"{windows_to_linux_path(args.input_stl)}\" --csv \"{windows_to_linux_path(args.input_csv)}\" --out \"{windows_to_linux_path(args.out)}\" --overwrite \"{args.overwrite}\" --model \"{model}\" --crown_segmentation \"{args.crown_segmentation}\" --array_name \"{args.array_name}\" --fdi \"{args.fdi}\" --suffix \"{args.suffix}\" --vtk_folder \"{windows_to_linux_path(args.vtk_folder)}\"']
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
    parser.add_argument('input_vtk',type=str)
    parser.add_argument('input_stl',type=str)
    parser.add_argument('input_csv',type = str)
    parser.add_argument('out',type=str)
    parser.add_argument('overwrite',type=str)
    parser.add_argument('model',type=str)
    parser.add_argument('crown_segmentation',type=str)
    parser.add_argument('array_name',type=str)
    parser.add_argument('fdi',type=str)
    parser.add_argument('suffix',type=str)
    parser.add_argument('vtk_folder',type=str)
    args = parser.parse_args()
    print("args : ",args)
    main(args)

