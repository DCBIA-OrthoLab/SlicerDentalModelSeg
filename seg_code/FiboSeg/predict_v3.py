print("Importing libraries...")


####
####
"""
V3: Ambient lights to have faster prediction (rotate camera instead of surface)
    Icosahedron 
    Choice: UNETR or UNET
    Random crown removal
"""
####
####


import os
import argparse
import torch
import time
from tqdm import tqdm
import numpy as np
import random
import math


# datastructures
from pytorch3d.structures import Meshes

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, HardPhongShader, PointLights,AmbientLights,TexturesVertex
)
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import sys
sys.path.insert(0,'..')
import fly_by_features as fbf
import post_process

import monai
from monai.inferers import (sliding_window_inference,SimpleInferer)
from monai.transforms import ToTensor

print("Initializing model...")
# Set the cuda device 
if torch.cuda.is_available():
  device = torch.device("cuda:0")
  torch.cuda.set_device(device)
else:
  device = torch.device("cpu") 

def main(args):
  # Initialize a perspective camera.
  cameras = FoVPerspectiveCameras(device=device)
  image_size = args.res
  # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
  raster_settings = RasterizationSettings(
      image_size=image_size, 
      blur_radius=0, 
      faces_per_pixel=1, 
  )
  # We can add a point light in front of the object. 

  lights = AmbientLights(device=device)
  rasterizer = MeshRasterizer(
          cameras=cameras, 
          raster_settings=raster_settings
      )
  phong_renderer = MeshRenderer(
      rasterizer=rasterizer,
      shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
  )
  num_classes = 34
  # create UNet


  if args.unetr == 1:
    model = monai.networks.nets.UNETR(
        spatial_dims=2,
        in_channels=4,   # images: torch.cuda.FloatTensor[batch_size,224,224,4]
        img_size=image_size,
        out_channels=num_classes, 
    ).to(device)


  else:
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=4,   # images: torch.cuda.FloatTensor[batch_size,224,224,4]
        out_channels=num_classes, 
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

  model.load_state_dict(torch.load(args.model))
  path = args.surf

  softmax = torch.nn.Softmax(dim=1)

  nb_rotations = args.rot
  l_outputs = []

  ## Camera position
  dist_cam = 1.35
  
  SURF = fbf.ReadSurf(path)

  if args.rem != 0:
    surf_point_data = SURF.GetPointData().GetScalars("UniversalID") 
    ## Remove crown
    unique, counts  = np.unique(surf_point_data, return_counts = True)
    id_to_remove = args.rem
    if id_to_remove not in unique:
      print(f'Warning: ID {id_to_remove} not in id list. Removing random label...')
    while (id_to_remove in [-1,33]) or (id_to_remove not in unique): 
        id_to_remove = random.choice(unique)
    print(f'ID to remove: {id_to_remove}')
    SURF = post_process.Threshold(SURF, "UniversalID" ,id_to_remove-0.5,id_to_remove+0.5, invert=True)



  surf_unit = fbf.GetUnitSurf(SURF)
  num_faces = int(SURF.GetPolys().GetData().GetSize()/4)   
 
  array_faces = np.zeros((num_classes,num_faces))
  tensor_faces = torch.zeros(num_classes,num_faces).to(device)
  model.eval() # Switch to eval mode
  simple_inferer = SimpleInferer()


  (V, F, CN) = GetSurfProp(args,surf_unit)  # 0.7s to compute : now 0.45s 
  list_sphere_points = fibonacci_sphere(samples=nb_rotations, dist_cam=dist_cam)
  list_sphere_points[0] = (0.0001, 1.35, 0.0001) # To avoid "invalid rotation matrix" error
  list_sphere_points[-1] = (0.0001, -1.35, 0.0001)

  ## PREDICTION
  for coords in tqdm(list_sphere_points, desc = 'Prediction      '):
    camera_position = ToTensor(dtype=torch.float32, device=device)([list(coords)])
    R = look_at_rotation(camera_position, device=device)  # (1, 3, 3)
    T = -torch.bmm(R.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)

    textures = TexturesVertex(verts_features=CN)
    meshes = Meshes(verts=V, faces=F, textures=textures)
    image = phong_renderer(meshes_world=meshes.clone(), R=R, T=T)
    pix_to_face, zbuf, bary_coords, dists = phong_renderer.rasterizer(meshes.clone())
    pix_to_face = pix_to_face.squeeze()
    image = image.permute(0,3,1,2)
    inputs = image.to(device)
    outputs = simple_inferer(inputs,model)  

    outputs_softmax = softmax(outputs).squeeze().detach().cpu().numpy() # t: negligeable           
    for x in range(image_size):
        for y in range (image_size): # Browse pixel by pixel
            array_faces[:,pix_to_face[x,y]] += outputs_softmax[...,x,y]
      
      
  array_faces[:,-1][0] = 0 # pixels that are background (id: 0) =-1
  faces_argmax = np.argmax(array_faces,axis=0)
  mask = 33 * (faces_argmax == 0) # 0 when face is not assigned to any pixel : we change that to the ID of the gum
  final_faces_array = faces_argmax + mask

  surf = SURF
  nb_points = surf.GetNumberOfPoints()
  polys = surf.GetPolys()
  np_connectivity = vtk_to_numpy(polys.GetConnectivityArray())

  id_points = np.full((nb_points,),33) # fill with ID 33 (gum)

  for index,uid in enumerate(final_faces_array.tolist()):
      id_points[np_connectivity[3*index]] = uid

  vtk_id = numpy_to_vtk(id_points)
  vtk_id.SetName(args.scal)
  surf.GetPointData().AddArray(vtk_id)




  # Remove Islands
  for label in tqdm(range(num_classes),desc = 'Removing islands'):
    post_process.RemoveIslands(surf, vtk_id, label, 200)


  out_filename = args.out
  polydatawriter = vtk.vtkPolyDataWriter()
  polydatawriter.SetFileName(out_filename)
  polydatawriter.SetInputData(surf)
  polydatawriter.Write()
  print("Done.")


def fibonacci_sphere(samples, dist_cam):

    points = []
    phi = math.pi * (3. -math.sqrt(5.))  # golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y*y)  # radius at y
        theta = phi*i 
        x = math.cos(theta)*radius
        z = math.sin(theta)*radius
        points.append((x*dist_cam, y*dist_cam, z*dist_cam))
    return points


def GetSurfProp(args,surf_unit):     
    surf = fbf.ComputeNormals(surf_unit)



    color_normals = ToTensor(dtype=torch.float32, device=device)(vtk_to_numpy(fbf.GetColorArray(surf, "Normals"))/255.0)
    verts = ToTensor(dtype=torch.float32, device=device)(vtk_to_numpy(surf.GetPoints().GetData()))
    faces = ToTensor(dtype=torch.int64, device=device)(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:])
    return verts.unsqueeze(0), faces.unsqueeze(0), color_normals.unsqueeze(0)

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Choose a .vtk file.')
  parser.add_argument('--surf',type=str, help='Input surface (.vtk file)', required=True)
  parser.add_argument('--out',type=str, help = 'Output', required=True)
  parser.add_argument('--rot',type=int, help = 'Number of rotations (default: 70)', default=70)
  parser.add_argument('--model',type=str, help = 'Model for segmentation', default="best_metric_model_segmentation2d_array.pth")
  parser.add_argument('--res',type=int, help = 'Image resolution for the fly-by (default: 256)', default=320)
  parser.add_argument('--scal',type=str, help = 'Predicted ID name', default="PredictedID")
  parser.add_argument('--unetr',type=int, help = '1: unetr', default=0)
  parser.add_argument('--rem',type=int, help = 'remove crown (-1 for random removal)',default=0)
  args = parser.parse_args()
  main(args)
