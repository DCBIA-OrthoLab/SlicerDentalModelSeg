from torch.utils.data import Dataset


import os
import torch

import glob

from __CrownSegmentation.utils import ReadSurf, ComputeNormals, GetColorArray, GetUnitSurf, RandomRotation


from vtk.util.numpy_support import vtk_to_numpy



class TeethDataset(Dataset):
    def __init__(self, path, transform=None):
        self.df = self.setup_df(path)
        self.transform = transform

    def setup_df(self,path):

        if os.path.isdir(path):
            l_inputs = glob.glob(f"{path}/*.vtk")
        elif os.path.isfile(path):
            l_inputs = [path]
        else:
            raise Exception ('Incorrect input.')
        print(f'intput file {l_inputs}')
        
        return l_inputs
    

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        

        surf_path = self.df[idx]
        surf = ReadSurf(surf_path)     


        if self.transform:
            surf = self.transform(surf)

        surf = ComputeNormals(surf)
        color_normals = torch.tensor(vtk_to_numpy(GetColorArray(surf, "Normals"))).to(torch.float32)/255.0
        verts = torch.tensor(vtk_to_numpy(surf.GetPoints().GetData())).to(torch.float32)
        faces = torch.tensor(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]).to(torch.int64)


        return verts, faces, color_normals

    def getSurf(self, idx):
        surf_path = self.df[idx]
        return ReadSurf(surf_path)
    
    def getName(self,idx):
        name = os.path.basename(self.df[idx])
        return name



class UnitSurfTransform:

    def __init__(self, random_rotation=False):
        
        self.random_rotation = random_rotation

    def __call__(self, surf):

        surf = GetUnitSurf(surf)
        if self.random_rotation:
            surf, _a, _v = RandomRotation(surf)
        return surf

