#!/usr/bin/env python-real

from slicer.util import pip_install
import argparse
import sys
import os
import numpy as np







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






def main(args):
    print('strat crown segmentation')
    print(f' args {args}')


        
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

