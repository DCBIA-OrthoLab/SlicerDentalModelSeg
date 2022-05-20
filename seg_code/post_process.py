import vtk
import numpy as np
import argparse
import sys
import os
from collections import namedtuple
from utils import * 

# parser = argparse.ArgumentParser()
# parser.add_argument('--mesh', help='Insert mesh path')
# parser.add_argument('--out', help='Insert output path+name')
# arg = parser.parse_args()

def ChangeLabel(vtkdata, label_array, label2change, change):
	# Set all the label 'label2change' in 'change'
	for pid in range (vtkdata.GetNumberOfPoints()):
		if int(label_array.GetTuple(pid)[0]) == label2change:
			label_array.SetTuple(pid, (change, ))
	return vtkdata, label_array

def Clip(vtkdata, value, scalars, InsideOut):
	# CLip the vtkdata following the scalar
	# scalars = 'RegionID' | 'Minimum_Curvature'
	# InsideOut = 'on' | 'off'
	vtkdata.GetPointData().SetActiveScalars(scalars)
	Clipper = vtk.vtkClipPolyData()
	Clipper.SetInputData(vtkdata)
	Clipper.SetValue(value)
	Clipper.GenerateClipScalarsOff()
	if InsideOut == 'off':
		Clipper.InsideOutOff()
	else:
		Clipper.InsideOutOn()
	Clipper.GetOutput().GetPointData().CopyScalarsOff()
	Clipper.Update()
	clipped_name = Clipper.GetOutput()
	return clipped_name

def Connectivity(vtkdata):
	# Labelize all the objects
	connectivityFilter = vtk.vtkConnectivityFilter()
	connectivityFilter.SetInputData(vtkdata)
	connectivityFilter.ScalarConnectivityOn()
	connectivityFilter.SetScalarRange([2,2])
	connectivityFilter.SetExtractionModeToAllRegions()
	connectivityFilter.ColorRegionsOn()
	connectivityFilter.Update()
	vtkdata = connectivityFilter.GetOutput()
	label_label = vtkdata.GetPointData().GetArray('RegionId')
	return vtkdata, label_label

def CountIDPoint(vtkdata, label_array):
	# Count the number of point of each IDs
	number_of_points = []
	for label in range(np.max(np.array(label_array)) + 1):
		current_nb = 0
		for pid in range(vtkdata.GetNumberOfPoints()):
			if int(label_array.GetTuple(pid)[0]) == label:
				current_nb += 1
		number_of_points.append(current_nb)
	return number_of_points

def ChangeSmallCompID(vtkdata, label_array, number_of_points, threshold, label):
	#Set the labek of all the object smaller than the threshold into 'label'
	for i in range(len(number_of_points)):
		number = number_of_points[i]
		if number < threshold :
			for pid in range (vtkdata.GetNumberOfPoints()):
				if int(label_array.GetTuple(pid)[0]) == i:
					label_array.SetTuple(pid, (label, ))
	return vtkdata


def LocateLabels(vtkdata_vtkdata, label_array_or, vtkdata_clipped, label_array_clip, label2change, label):
	# Assign the label 'label' to the vtkdata vtkdata points which are labeled 'label2change' in the clipped vtkdata 
	locator = vtk.vtkPointLocator()
	locator.SetDataSet(vtkdata_vtkdata) 
	locator.BuildLocator()
	number_of_changes = 0
	for pid in range(vtkdata_clipped.GetNumberOfPoints()):
		if int(label_array_clip.GetTuple(pid)[0]) == label2change:
			coordinates = vtkdata_clipped.GetPoint(pid)
			vtkdata_Equivalent_ID = locator.FindClosestPoint(coordinates)
			label_array_or.SetTuple(vtkdata_Equivalent_ID, (label, ))
			number_of_changes += 1
	# print('LocateLabels ===> ', 'Number of changes :', number_of_changes)
	return vtkdata_vtkdata

def Write(vtkdata, output_name):
	outfilename = output_name
	print("Writting:", outfilename)
	polydatawriter = vtk.vtkPolyDataWriter()
	polydatawriter.SetFileName(outfilename)
	polydatawriter.SetInputData(vtkdata)
	polydatawriter.Write()


def GetCurvature(vtkdata):
	curve=vtk.vtkCurvatures()
	curve.SetCurvatureTypeToMinimum()
	curve.SetInputData(vtkdata)
	curve.Update()
	vtkdata=curve.GetOutput()
	return vtkdata

def RegionGrowing(vtkdata, label_array, label2change, exception):
	#Take a look of the neghbor's id and take it if it s different set it to label2change. Exception is done to avoid unwanted labels.
	ids = []
	for pid in range(vtkdata.GetNumberOfPoints()):
		if int(label_array.GetTuple(pid)[0]) == label2change:
			ids.append(pid)
	first_len = len(ids)
	# print('RegionGrowing ===> ', 'Number of ids that will change :', first_len)
	count = 0
	while len(ids) > 0:
		count += 1
		for pid in ids[:]:
			neighbor_ids = NeighborPoints(vtkdata, pid)
			for nid in neighbor_ids:
				neighbor_label = int(label_array.GetTuple(nid)[0])
				if(neighbor_label != label2change) and (neighbor_label != exception):
					label_array.SetTuple(pid, (neighbor_label,))
					ids.remove(pid)
					count = 0
					break
		if count == 2:
			#Sometimes the while loop can t find any label != -1 
			print('RegionGrowing ===> WARNING :', len(ids), '/', first_len, 'label(s) has been undertermined. Setting to 0.')
			break		

	for pid in ids:
		#Then we set theses label to 0
		label_array.SetTuple(pid, (0,))
	return vtkdata

def NeighborPoints(vtkdata,CurrentID):
	cells_id = vtk.vtkIdList()
	vtkdata.GetPointCells(CurrentID, cells_id)
	all_neighbor_pid = []
	for ci in range(cells_id.GetNumberOfIds()):
		cells_id_inner = vtk.vtkIdList()
		vtkdata.GetCellPoints(cells_id.GetId(ci), cells_id_inner)
		for pi in range(cells_id_inner.GetNumberOfIds()):
			all_neighbor_pid.append(cells_id_inner.GetId(pi))
	
	all_neighbor_pid = np.unique(all_neighbor_pid)
	return all_neighbor_pid

def GetBoundaries(vtkdata, label_array, label1, label2, Set_label):
	# Set a label 'Set_label' each time label1 and label2 are connected
	for pid in range(vtkdata.GetNumberOfPoints()):
		if int(label_array.GetTuple(pid)[0]) == label1:
			neighbor_ids = NeighborPoints(vtkdata, pid)
			for nid in neighbor_ids:
				neighbor_label = int(label_array.GetTuple(nid)[0])
				if(neighbor_label == label2):
					label_array.SetTuple(nid, (Set_label, ))
					label_array.SetTuple(pid, (Set_label, ))
	return vtkdata


def RealLabels(vtkdata, label_array):
	# Change the label to the label used for the model training
	vtkdata, label_array= ChangeLabel(vtkdata, label_array, 1, 11)
	vtkdata, label_array= ChangeLabel(vtkdata, label_array, 2, 22)
	vtkdata, label_array= ChangeLabel(vtkdata, label_array, 0, 2)
	vtkdata, label_array= ChangeLabel(vtkdata, label_array, 11, 0)
	vtkdata, label_array= ChangeLabel(vtkdata, label_array, 22, 1)
	return vtkdata


def Post_processing(vtkdata):
	#Remove all the smalls comoponents by setting their label to 
	label_array = vtkdata.GetPointData().GetArray('RegionId')
	vtkdata = GetBoundaries(vtkdata, label_array, 1,2,0)
	teeth = Clip(vtkdata, 1.5, 'RegionId', 'off')

	teeth = Clip(vtkdata, 1.5, 'RegionId', 'off')
	teeth, teeth_label = Connectivity(teeth)
	nb_teeth = CountIDPoint(teeth, teeth_label)
	teeth = GetCurvature(teeth)
	teeth = ChangeSmallCompID(teeth, teeth_label, nb_teeth, 1000, -1)

	vtkdata, label_array= ChangeLabel(vtkdata, label_array, 1, 3)
	gum = Clip(vtkdata, 2.5, 'RegionId', 'off')
	vtkdata, label_array = ChangeLabel(vtkdata, label_array, 3, 1)
	gum, gum_label = Connectivity(gum)
	nb_gum = CountIDPoint(gum, gum_label)
	gum = ChangeSmallCompID(gum, gum_label, nb_gum, 1000, -1)

	bound = Clip(vtkdata, 0.5, 'RegionId', 'on')
	bound, bound_label = Connectivity(bound)
	nb_bound = CountIDPoint(bound, bound_label)
	bound = ChangeSmallCompID(bound, bound_label, nb_bound, 1000, -1)
	vtkdata = LocateLabels(vtkdata, label_array, teeth, teeth_label , -1, -2)
	vtkdata = LocateLabels(vtkdata, label_array, bound, bound_label , -1, -1)
	vtkdata = LocateLabels(vtkdata, label_array, gum, gum_label , -1, -3)

	vtkdata = GetBoundaries(vtkdata, label_array, 1,2,0)

	vtkdata = RegionGrowing(vtkdata, label_array, -1, -1)
	vtkdata = RegionGrowing(vtkdata, label_array, -2, 2)
	vtkdata = RegionGrowing(vtkdata, label_array, -3, 1)
	vtkdata = RealLabels(vtkdata, label_array)
	# Write(vtkdata, 'test.vtk')
	return vtkdata,label_array

def ReadFile(filename):
	inputSurface = filename
	reader = vtk.vtkPolyDataReader()
	reader.SetFileName(inputSurface)
	reader.Update()
	vtkdata = reader.GetOutput()
	label_array = vtkdata.GetPointData().GetArray('RegionId')
	return vtkdata, label_array

def Write(vtkdata, output_name):
	outfilename = output_name
	print("Writting:", outfilename)
	polydatawriter = vtk.vtkPolyDataWriter()
	polydatawriter.SetFileName(outfilename)
	polydatawriter.SetInputData(vtkdata)
	polydatawriter.Write()


def Label_Teeth(vtkdata, label_array):
	vtkdata, label_array= ChangeLabel(vtkdata, label_array, 1, 3)
	predict_teeth = Clip(vtkdata, 2.5, 'RegionId', 'off')
	vtkdata, label_array = ChangeLabel(vtkdata, label_array, 3, 1)

	predict_teeth, predict_teeth_label = Connectivity(predict_teeth)
	nb_predteeth = CountIDPoint(predict_teeth, predict_teeth_label)
	predict_teeth = ChangeSmallCompID(predict_teeth, predict_teeth_label, nb_predteeth, 1000, -1)

	predict_teeth = Clip(predict_teeth, -0.5, 'RegionId', 'off')
	predict_teeth, predict_teeth_label = Connectivity(predict_teeth) 
	for label in range(np.max(np.array(predict_teeth_label))+1):
		vtkdata = LocateLabels(vtkdata, label_array, predict_teeth, predict_teeth_label, label, label + 3) #Labels have to start at 3
		
	return vtkdata

def ConnectedRegion(vtkdata, pid, labels, label, pid_visited):

	neighbor_pids = GetNeighborIds(vtkdata, pid, labels, label, pid_visited)
	all_connected_pids = [pid]
	all_connected_pids.extend(neighbor_pids)

	while len(neighbor_pids):
		npid = neighbor_pids.pop()
		next_neighbor_pids = GetNeighborIds(vtkdata, npid, labels, label, pid_visited)
		neighbor_pids.extend(next_neighbor_pids)
		all_connected_pids = np.append(all_connected_pids, next_neighbor_pids)

	return np.unique(all_connected_pids)

def NeighborLabel(vtkdata, labels, label, connected_pids):
	neighbor_ids = []
	
	for pid in connected_pids:
		cells_id = vtk.vtkIdList()
		vtkdata.GetPointCells(int(pid), cells_id)
		for ci in range(cells_id.GetNumberOfIds()):
			points_id_inner = vtk.vtkIdList()
			vtkdata.GetCellPoints(cells_id.GetId(ci), points_id_inner)
			for pi in range(points_id_inner.GetNumberOfIds()):
				pid_inner = points_id_inner.GetId(pi)
				if labels.GetTuple(pid_inner)[0] != label:
					neighbor_ids.append(pid_inner)

	neighbor_ids = np.unique(neighbor_ids)
	neighbor_labels = []

	for nid in neighbor_ids:
		neighbor_labels.append(labels.GetTuple(nid)[0])
	
	if len(neighbor_labels) > 0:
		return max(neighbor_labels, key=neighbor_labels.count)
	return -1



def RemoveIslands(vtkdata, labels, label, min_count,ignore_neg1 = False):

	pid_visited = np.zeros(labels.GetNumberOfTuples())
	for pid in range(labels.GetNumberOfTuples()):
		if labels.GetTuple(pid)[0] == label and pid_visited[pid] == 0:
			connected_pids = ConnectedRegion(vtkdata, pid, labels, label, pid_visited)
			if connected_pids.shape[0] < min_count:
				neighbor_label = NeighborLabel(vtkdata, labels, label, connected_pids)
				if ignore_neg1 == True and neighbor_label != -1:
					for cpid in connected_pids:
						labels.SetTuple(int(cpid), (neighbor_label,))

def ConnectivityLabeling(vtkdata, labels, label, start_label):
	pid_visited = np.zeros(labels.GetNumberOfTuples())
	for pid in range(labels.GetNumberOfTuples()):
		if labels.GetTuple(pid)[0] == label and pid_visited[pid] == 0:
			connected_pids = ConnectedRegion(vtkdata, pid, labels, label, pid_visited)
			for cpid in connected_pids:
				labels.SetTuple(int(cpid), (start_label,))
			start_label += 1

def ErodeLabel(vtkdata, labels, label, ignore_label=None):
	
	pid_labels = []
	for pid in range(labels.GetNumberOfTuples()):
		if labels.GetTuple(pid)[0] == label:
			pid_labels.append(pid)

	while pid_labels:
		pid_labels_remain = pid_labels
		pid_labels = []

		all_neighbor_pids = []
		all_neighbor_labels = []

		while pid_labels_remain:

			pid = pid_labels_remain.pop()

			neighbor_pids = GetNeighbors(vtkdata, pid)
			is_neighbor = False

			for npid in neighbor_pids:
				neighbor_label = labels.GetTuple(npid)[0]
				if neighbor_label != label and (ignore_label == None or neighbor_label != ignore_label):
					all_neighbor_pids.append(pid)
					all_neighbor_labels.append(neighbor_label)
					is_neighbor = True
					break

			if not is_neighbor:
				pid_labels.append(pid)

		if(all_neighbor_pids):
			for npid, nlabel in zip(all_neighbor_pids, all_neighbor_labels):
				labels.SetTuple(int(npid), (nlabel,))
		else:
			break

def DilateLabel(vtkdata, labels, label, iterations=2):
	
	pid_labels = []

	while iterations > 0:
		#Get all neighbors to the 'label' that have a different label
		all_neighbor_labels = []
		for pid in range(labels.GetNumberOfTuples()):
			if labels.GetTuple(pid)[0] == label:
				neighbor_pids = GetNeighbors(vtkdata, pid)
				pid_labels.append(pid)

				for npid in neighbor_pids:
					neighbor_label = labels.GetTuple(npid)[0]
					if neighbor_label != label:
						all_neighbor_labels.append(npid)

		#Dilate them, i.e., change the value to label
		for npid in all_neighbor_labels:
			labels.SetTuple(int(npid), (label,))

		iterations -= 1

def ReLabel(surf, labels, label, relabel):
	for pid in range(labels.GetNumberOfTuples()):
		if labels.GetTuple(pid)[0] == label:
			labels.SetTuple(pid, (relabel,))

def Threshold(vtkdata, labels, threshold_min, threshold_max, invert=False):
	
	threshold = vtk.vtkThreshold()
	threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, labels)
	threshold.SetInputData(vtkdata)
	threshold.ThresholdBetween(threshold_min,threshold_max)
	threshold.SetInvert(invert)
	threshold.Update()

	geometry = vtk.vtkGeometryFilter()
	geometry.SetInputData(threshold.GetOutput())
	geometry.Update()
	return geometry.GetOutput()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Predict an input with a trained neural network', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--surf', type=str, help='Input surface mesh to label', required=True)
	parser.add_argument('--remove_islands', type=bool, help='Remove islands from mesh by labeling with the closes one', default=False)
	parser.add_argument('--connectivity', type=bool, help='Label all elements with unique labels', default=False)
	parser.add_argument('--connectivity_label', type=int, help='Connectivity label', default=2)
	parser.add_argument('--erode', type=bool, help='Erode label until it dissapears changing it with the neighboring label', default=False)
	parser.add_argument('--ignore', type=int, help='Ignore label when eroding', default=None)
	parser.add_argument('--dilate', type=bool, help='Erode label until it dissapears changing it with the neighboring label', default=False)
	parser.add_argument('--dilate_iterations', type=int, help='Number of dilate iterations', default=2)
	parser.add_argument('--label', type=int, help='Eroding/dilating/ReLabel label', default=0)
	parser.add_argument('--threshold', type=bool, help='Threshold between two values', default=False)
	parser.add_argument('--relabel', type=bool, help='relabel an input surface', default=False)
	parser.add_argument('--label_re', type=int, help='relabel with this label', default=-1)
	parser.add_argument('--threshold_min', type=int, help='Threshold min value', default=2)
	parser.add_argument('--threshold_max', type=int, help='Threshold max value', default=100)
	parser.add_argument('--min_count', type=int, help='Minimum count to remove', default=500)

	parser.add_argument('--out', type=str, help='Output model with labels', default="out.vtk")


	args = parser.parse_args()
	surf, labels = ReadFile(args.surf)

	if(args.remove_islands):
		labels_range = np.zeros(2)
		labels.GetRange(labels_range)
		for label in range(int(labels_range[0]), int(labels_range[1]) + 1):
			print("Removing islands:", label)
			RemoveIslands(surf, labels, label, args.min_count)

	if args.relabel:
		print("Relabel:", )
		ReLabel(surf, labels, args.label, args.label_re)
	
	if(args.connectivity):
		print("Connectivity...")
		ConnectivityLabeling(surf, labels, args.connectivity_label, 2)

	if(args.erode):
		print("Eroding...")
		ErodeLabel(surf, labels, args.label, args.ignore)

	if(args.dilate):
		print("Dilate...")
		DilateLabel(surf, labels, args.label, args.dilate_iterations)

	if(args.threshold):
		print("Thresholding...")
		surf = Threshold(surf, labels, args.threshold_min, args.threshold_max)

	Write(surf, args.out)





