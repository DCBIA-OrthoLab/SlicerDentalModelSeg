import argparse
import sys
import os
import numpy as np
import platform
import urllib
import subprocess
import shutil

      
# normal execution

from tqdm import tqdm



import pandas


import torch

import pytorch3d




import monai
 



import pytorch_lightning




from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from torch.utils.data import DataLoader

import sys
actual_path = os.path.abspath(__file__)
print("!"*150)
print("actual_path : ",actual_path)
sys.path.append('/mnt/c/Users/luciacev.UMROOT/Documents/SlicerDentalModelSeg/CrownSegmentationcli')



print("!"*150)

from __CrownSegmentation import (MonaiUNet, TeethDataset, UnitSurfTransform, Write, RemoveIslands, 
                DilateLabel, ErodeLabel, Threshold, ConvertFDI)




def running(args):
    print("args : ",args)
    chemin_fichier_actuel = os.path.abspath(__file__)
    print(chemin_fichier_actuel)
    print(os.path.dirname(chemin_fichier_actuel))
    
    
    with open(args["logPath"],'w') as log_f:
          # clear log file
        log_f.truncate(0)


        class_weights = None
        out_channels = 34

        model = MonaiUNet( out_channels = out_channels, class_weights=class_weights, image_size=args["resolution"], subdivision_level=args["subdivision_level"])

        model.model.module.load_state_dict(torch.load(args["model"]))

        print("2"*150)
        ds = TeethDataset(args["input"], transform=UnitSurfTransform())
        print("3"*150)

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
                V_labels_prediction.SetName(args["predictedId"])
                surf.GetPointData().AddArray(V_labels_prediction)


                #Post Process
                RemoveIslands(surf,V_labels_prediction,33,500, ignore_neg1=True)
                for label in tqdm(range(out_channels),desc= 'Remove island'):
                    RemoveIslands(surf,V_labels_prediction, label, 200, ignore_neg1=True)




                for label in tqdm(range(1,out_channels),desc= 'Closing operation'):
                    DilateLabel(surf,V_labels_prediction, label, iterations=2, dilateOverTarget=False, target = None)
                    ErodeLabel(surf,V_labels_prediction, label, iterations=2, target=None)


                if args["chooseFDI"] :
                    surf = ConvertFDI(surf,args.predicteId)
                    gum_label = 0
                else :
                    gum_label = 33




                output_fn = os.path.join(args["output"], ds.getName(idx))

                output_dir = os.path.dirname(output_fn)

                if(not os.path.exists(output_dir)):
                    os.makedirs(output_dir)




                if args["sepOutputs"]:
                    # Isolate each label
                    surf_point_data = surf.GetPointData().GetScalars(args["predictedId"]) 
                    labels = np.unique(surf_point_data)
                    out_basename = output_fn[:-4]
                    for label in tqdm(labels, desc = 'Isolating labels'):
                        thresh_label = Threshold(surf, args["predictedId"] ,label-0.5,label+0.5)
                        if label != gum_label:
                            Write(thresh_label,f'{out_basename}_id_{label}.vtk',print_out=False) 
                        else:
                        # gum
                            Write(thresh_label,f'{out_basename}_gum.vtk',print_out=False) 
                    # all teeth 
                    no_gum = Threshold(surf, args["predictedId"] ,gum_label-0.5,gum_label+0.5,invert=True)
                    Write(no_gum,f'{out_basename}_all_teeth.vtk',print_out=False)


                Write(surf , output_fn, print_out=False)

                with open(args["logPath"],'r+') as log_f :
                    log_f.write(str(idx))
                    
    print("TOUT C'EST BIEN PASSE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        

def windows_to_linux_path(windows_path):
    # Supprime le caractère de retour chariot
    windows_path = windows_path.strip()

    # Remplace les backslashes par des slashes
    path = windows_path.replace('\\', '/')

    # Remplace le lecteur par '/mnt/lettre_du_lecteur'
    if ':' in path:
        drive, path_without_drive = path.split(':', 1)
        path = "/mnt/" + drive.lower() + path_without_drive

    return path

if __name__ == "__main__":
  
    if len(sys.argv) > 3 :
    
        args = {
        "input": windows_to_linux_path(sys.argv[1]),
        "output": windows_to_linux_path(sys.argv[2]),
        "subdivision_level": int(sys.argv[3]),
        "resolution": int(sys.argv[4]),
        "model": windows_to_linux_path(sys.argv[5]),
        "predictedId": sys.argv[6],

        "sepOutputs": int(sys.argv[7]),
        "chooseFDI": int(sys.argv[8]),
        "logPath": windows_to_linux_path(sys.argv[9])
        }
        
        running(args)

