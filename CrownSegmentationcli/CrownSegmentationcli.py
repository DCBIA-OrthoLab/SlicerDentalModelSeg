#!/usr/bin/env python-real
print("Importing libraries...")
from slicer.util import pip_install
import os
import sys

if sys.argv[1] == '-1':
  # Install dependencies
  print('Installing dependencies...')
  pip_install('tqdm==4.64.0') # tqdm
  pip_install('pandas==1.4.2') # pandas
  pip_install('--no-cache-dir torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html') # torch
  pip_install('--no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu111_pyt1101/download.html') # pytorch3d
  pip_install('itk==5.2.1.post1') # itk
  pip_install('monai==0.7.0') # monai


else:
  # normal execution
  try:
    from tqdm import tqdm
  except ImportError:
    pip_install('tqdm==4.64.0')

  try:
    import pandas
  except ImportError:
    pip_install('pandas==1.4.2')

  try:
    import torch
    pyt_version_str=torch.__version__.split("+")[0].replace(".", "")
    version_str="".join([f"py3{sys.version_info.minor}_cu",torch.version.cuda.replace(".",""),f"_pyt{pyt_version_str}"])  
    if version_str != 'py39_cu111_pyt190':
      raise ImportError
  except ImportError:
    pip_install('--no-cache-dir torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html')

  try:
    import pytorch3d
    if pytorch3d.__version__ != '0.6.0':
      raise ImportError
  except ImportError:
    try:
      import torch
      pyt_version_str=torch.__version__.split("+")[0].replace(".", "")
      version_str="".join([f"py3{sys.version_info.minor}_cu",torch.version.cuda.replace(".",""),f"_pyt{pyt_version_str}"])
      pip_install('--upgrade pip')
      pip_install('fvcore==0.1.5.post20220305')
      pip_install('--no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html')
    except: # install correct torch version
      pip_install('--no-cache-dir torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html') 
      pip_install('--no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu111_pyt1101/download.html')

  try:
    import itk
  except ImportError:
    pip_install('itk==5.2.1.post1')

  try:
    import monai
    if monai.__version__ != '0.8.dev2143':
      raise ImportError
  except ImportError:
    pip_install('monai==0.7.0')

  fileDir = os.path.dirname(os.path.abspath(__file__)).split('/')
  fileDir[-1] = 'seg_code'
  code_path = '/'.join(fileDir)
  sys.path.append(code_path)  

  import utils
  import post_process
  import argparse
  import numpy as np
  import math
  from vtk import vtkPolyDataWriter
  from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
  # datastructures
  from pytorch3d.structures import Meshes

  # rendering components
  from pytorch3d.renderer import (
      FoVPerspectiveCameras, look_at_rotation, 
      RasterizationSettings, MeshRenderer, MeshRasterizer, HardPhongShader, PointLights,AmbientLights,TexturesVertex
  )

  # monai imports
  from monai.inferers import (sliding_window_inference,SimpleInferer)
  from monai.transforms import ToTensor

  print("Initializing model...",flush=True)

  # Set the cuda device 
  if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
  else:
    device = torch.device("cpu") 


def main(surf,out,rot,res,unet_model,scal):
  #if not os.path.isfile(f'{code_path}/process.log'):


  with open(f'{code_path}/process.log','w') as log_f:
    # clear log file
    log_f.truncate(0)
  progress = 0

  # Initialize a perspective camera.
  cameras = FoVPerspectiveCameras(device=device)
  image_size = res

  # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
  raster_settings = RasterizationSettings(
      image_size=image_size, 
      blur_radius=0, 
      faces_per_pixel=1, 
  )
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

  model = monai.networks.nets.UNet(
      spatial_dims=2,
      in_channels=4,   # images: torch.cuda.FloatTensor[batch_size,224,224,4]
      out_channels=num_classes, 
      channels=(16, 32, 64, 128, 256),
      strides=(2, 2, 2, 2),
      num_res_units=2,
  ).to(device)

  model.load_state_dict(torch.load(unet_model))
  path = surf

  softmax = torch.nn.Softmax(dim=1)

  nb_rotations = rot
  l_outputs = []

  ## Camera position
  dist_cam = 1.35
  
  SURF = utils.ReadSurf(path)
  surf_unit = utils.GetUnitSurf(SURF)


  num_faces = int(SURF.GetPolys().GetData().GetSize()/4)   
 
  array_faces = np.zeros((num_classes,num_faces))
  tensor_faces = torch.zeros(num_classes,num_faces).to(device)
  model.eval() # Switch to eval mode
  simple_inferer = SimpleInferer()


  (V, F, CN) = GetSurfProp(surf_unit)  # 0.7s to compute : now 0.45s 
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

    progress += 1
    with open(f'{code_path}/process.log','r+') as log_f:
      log_f.write(str(progress))

    
  

  array_faces[:,-1][0] = 0 # pixels that are background (id: 0) =-1
  faces_argmax = np.argmax(array_faces,axis=0)
  mask = 33 * (faces_argmax == 0) # 0 when face is not assigned to any pixel : we change that to the ID of the gum
  final_faces_array = faces_argmax + mask
  unique, counts  = np.unique(final_faces_array, return_counts = True)

  surf = SURF
  nb_points = surf.GetNumberOfPoints()
  polys = surf.GetPolys()
  np_connectivity = vtk_to_numpy(polys.GetConnectivityArray())

  id_points = np.full((nb_points,),33) # fill with ID 33 (gum)

  for index,uid in enumerate(final_faces_array.tolist()):
      id_points[np_connectivity[3*index]] = uid

  vtk_id = numpy_to_vtk(id_points)
  vtk_id.SetName(scal)
  surf.GetPointData().AddArray(vtk_id)


  # Remove Islands
  for label in tqdm(range(num_classes),desc = 'Removing islands'):
    post_process.RemoveIslands(surf, vtk_id, label, 200,ignore_neg1 = True)  # adds -1 labels to isolated points: removed later
    progress += 1
  
  array = surf.GetPointData().GetScalars("PredictedID") 

  unique, counts  = np.unique(array, return_counts = True)

  out_filename = out
  polydatawriter = vtkPolyDataWriter()
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


def GetSurfProp(surf_unit):     
    surf = utils.ComputeNormals(surf_unit)

    color_normals = ToTensor(dtype=torch.float32, device=device)(vtk_to_numpy(utils.GetColorArray(surf, "Normals"))/255.0)
    verts = ToTensor(dtype=torch.float32, device=device)(vtk_to_numpy(surf.GetPoints().GetData()))
    faces = ToTensor(dtype=torch.int64, device=device)(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:])
    return verts.unsqueeze(0), faces.unsqueeze(0), color_normals.unsqueeze(0)



if __name__ == "__main__":
  if len (sys.argv) < 7:
    print("Usage: CrownSegmentationcli <surf> <out> <rot> <res> <model> <scal>")
    sys.exit (1)

  if sys.argv[1] != '-1':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5],sys.argv[6])