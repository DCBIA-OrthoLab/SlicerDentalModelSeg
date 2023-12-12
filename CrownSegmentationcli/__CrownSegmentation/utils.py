import vtk
import numpy as np
import os
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
from monai.transforms import ToTensor
import torch

actual_path = os.path.abspath(__file__)

if actual_path.startswith('/mnt'):
    system = "WSL"
else:
    system = "other"
    
if system!="WSL" :
    import __CrownSegmentation.LinearSubdivisionFilter as lsf
else :
    from __CrownSegmentation import LinearSubdivisionFilter as lsf


def Write(vtkdata, output_name, print_out = True):
    outfilename = output_name
    if print_out == True:
        print("Writing:", outfilename)
    polydatawriter = vtk.vtkPolyDataWriter()
    polydatawriter.SetFileName(outfilename)
    polydatawriter.SetInputData(vtkdata)
    polydatawriter.Write()






def ReadSurf(fileName):

    fname, extension = os.path.splitext(fileName)
    extension = extension.lower()
    if extension == ".vtk":
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(fileName)
        reader.Update()
        surf = reader.GetOutput()
    elif extension == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(fileName)
        reader.Update()
        surf = reader.GetOutput()    
    elif extension == ".stl":
        reader = vtk.vtkSTLReader()
        reader.SetFileName(fileName)
        reader.Update()
        surf = reader.GetOutput()
    elif extension == ".off":
        from readers import OFFReader
        reader = OFFReader()
        reader.SetFileName(fileName)
        reader.Update()
        surf = reader.GetOutput()
    elif extension == ".obj":
        if os.path.exists(fname + ".mtl"):
            obj_import = vtk.vtkOBJImporter()
            obj_import.SetFileName(fileName)
            obj_import.SetFileNameMTL(fname + ".mtl")
            textures_path = os.path.normpath(os.path.dirname(fname) + "/../images")
            if os.path.exists(textures_path):
                textures_path = os.path.normpath(fname.replace(os.path.basename(fname), ''))
                obj_import.SetTexturePath(textures_path)
            else:
                textures_path = os.path.normpath(fname.replace(os.path.basename(fname), ''))                
                obj_import.SetTexturePath(textures_path)
                    

            obj_import.Read()

            actors = obj_import.GetRenderer().GetActors()
            actors.InitTraversal()
            append = vtk.vtkAppendPolyData()

            for i in range(actors.GetNumberOfItems()):
                surfActor = actors.GetNextActor()
                append.AddInputData(surfActor.GetMapper().GetInputAsDataSet())
            
            append.Update()
            surf = append.GetOutput()
            
        else:
            reader = vtk.vtkOBJReader()
            reader.SetFileName(fileName)
            reader.Update()
            surf = reader.GetOutput()
    elif extension == '.gii':
        import nibabel as nib

        surf = nib.load(fileName)
        coords = surf.agg_data('pointset')
        triangles = surf.agg_data('triangle')

        points = vtk.vtkPoints()

        for c in coords:
            points.InsertNextPoint(c[0], c[1], c[2])

        cells = vtk.vtkCellArray()

        for t in triangles:
            t_vtk = vtk.vtkTriangle()
            t_vtk.GetPointIds().SetId(0, t[0])
            t_vtk.GetPointIds().SetId(1, t[1])
            t_vtk.GetPointIds().SetId(2, t[2])
            cells.InsertNextCell(t_vtk)

        surf = vtk.vtkPolyData()
        surf.SetPoints(points)
        surf.SetPolys(cells)

    return surf


def ComputeNormals(surf):
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(surf);
    normals.ComputeCellNormalsOn();
    normals.ComputePointNormalsOn();
    normals.SplittingOff();
    normals.Update()
    
    return normals.GetOutput()






def GetUnitSurf(surf, mean_arr = None, scale_factor = None, copy=True):
  unit_surf, surf_mean, surf_scale = ScaleSurf(surf, mean_arr, scale_factor, copy)
  return unit_surf




def ScaleSurf(surf, mean_arr = None, scale_factor = None, copy=True):
    if(copy):
        surf_copy = vtk.vtkPolyData()
        surf_copy.DeepCopy(surf)
        surf = surf_copy

    shapedatapoints = surf.GetPoints()

    #calculate bounding box
    mean_v = [0.0] * 3
    bounds_max_v = [0.0] * 3

    bounds = shapedatapoints.GetBounds()

    mean_v[0] = (bounds[0] + bounds[1])/2.0
    mean_v[1] = (bounds[2] + bounds[3])/2.0
    mean_v[2] = (bounds[4] + bounds[5])/2.0
    bounds_max_v[0] = max(bounds[0], bounds[1])
    bounds_max_v[1] = max(bounds[2], bounds[3])
    bounds_max_v[2] = max(bounds[4], bounds[5])

    shape_points = vtk_to_numpy(shapedatapoints.GetData())
    
    #centering points of the shape
    if mean_arr is None:
        mean_arr = np.array(mean_v)
    # print("Mean:", mean_arr)
    shape_points = shape_points - mean_arr

    #Computing scale factor if it is not provided
    if(scale_factor is None):
        bounds_max_arr = np.array(bounds_max_v)
        scale_factor = 1/np.linalg.norm(bounds_max_arr - mean_arr)

    #scale points of the shape by scale factor
    # print("Scale:", scale_factor)
    shape_points = np.multiply(shape_points, scale_factor)

    #assigning scaled points back to shape
    shapedatapoints.SetData(numpy_to_vtk(shape_points))

    return surf, mean_arr, scale_factor


def RandomRotation(surf):
    rotationAngle = np.random.random()*360.0
    rotationVector = np.random.random(3)*2.0 - 1.0
    rotationVector = rotationVector/np.linalg.norm(rotationVector)
    return RotateSurf(surf, rotationAngle, rotationVector), rotationAngle, rotationVector


def RotateSurf(surf, rotationAngle, rotationVector):
    transform = GetTransform(rotationAngle, rotationVector)
    return RotateTransform(surf, transform)



def GetTransform(rotationAngle, rotationVector):
    transform = vtk.vtkTransform()
    transform.RotateWXYZ(rotationAngle, rotationVector[0], rotationVector[1], rotationVector[2])
    return transform

def RotateTransform(surf, transform):
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transform)
    transformFilter.SetInputData(surf)
    transformFilter.Update()
    return transformFilter.GetOutput()



def GetColorArray(surf, array_name):
    colored_points = vtk.vtkUnsignedCharArray()
    colored_points.SetName('colors')
    colored_points.SetNumberOfComponents(3)

    normals = surf.GetPointData().GetArray(array_name)

    for pid in range(surf.GetNumberOfPoints()):
        normal = np.array(normals.GetTuple(pid))
        rgb = (normal*0.5 + 0.5)*255.0
        colored_points.InsertNextTuple3(rgb[0], rgb[1], rgb[2])
    return colored_points

def PolyDataToNumpy(surf):

    edges_filter = vtk.vtkExtractEdges()
    edges_filter.SetInputData(surf)
    edges_filter.Update()

    verts = vtk_to_numpy(surf.GetPoints().GetData())
    faces = vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]
    edges = vtk_to_numpy(edges_filter.GetOutput().GetLines().GetData()).reshape(-1, 3)[:,1:]
    
    return verts, faces, edges


def PolyDataToTensors(surf, device='cpu'):

    verts, faces, edges = PolyDataToNumpy(surf)
    
    verts = ToTensor(dtype=torch.float32, device=device)(verts)
    faces = ToTensor(dtype=torch.int32, device=device)(faces)
    edges = ToTensor(dtype=torch.int32, device=device)(edges)
    
    return verts, faces, edges

def CreateIcosahedron(radius, sl=0):
    icosahedronsource = vtk.vtkPlatonicSolidSource()
    icosahedronsource.SetSolidTypeToIcosahedron()
    icosahedronsource.Update()
    icosahedron = icosahedronsource.GetOutput()
    
    subdivfilter = lsf.LinearSubdivisionFilter()
    subdivfilter.SetInputData(icosahedron)
    subdivfilter.SetNumberOfSubdivisions(sl)
    subdivfilter.Update()

    icosahedron = subdivfilter.GetOutput()
    icosahedron = normalize_points(icosahedron, radius)

    return icosahedron



def normalize_points(poly, radius):
    polypoints = poly.GetPoints()
    for pid in range(polypoints.GetNumberOfPoints()):
        spoint = polypoints.GetPoint(pid)
        spoint = np.array(spoint)
        norm = np.linalg.norm(spoint)
        spoint = spoint/norm * radius
        polypoints.SetPoint(pid, spoint)
    poly.SetPoints(polypoints)
    return poly


def ConvertFDI(surf, scal):

  LUT = np.array([0,18,17,16,15,14,13,12,11,21,22,23,24,25,26,27,28,
                  38,37,36,35,34,33,32,31,41,42,43,44,45,46,47,48,0])
  # extract UniversalID array
  labels = vtk_to_numpy(surf.GetPointData().GetScalars(scal))
  
  # convert to their numbering system
  labels = LUT[labels]
  vtk_id = numpy_to_vtk(labels)
  vtk_id.SetName(scal)
  surf.GetPointData().AddArray(vtk_id)
  return surf
