import os
import sys
import unittest
import logging
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from enum import Enum


import webbrowser
import json

#
# CrownSegmentation
#

class CrownSegmentation(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Crown Segmentation - FiboSeg" 
    self.parent.categories = ["Segmentation"]  # TODO: set categories (folders where the module shows up in the module selector)
    self.parent.dependencies = []  # TODO: add here list of module names that this module requires
    self.parent.contributors = ["Mathieu Leclercq (University of North Carolina)", 
    "Juan Carlos Prieto (University of North Carolina)",
    "Martin Styner (University of North Carolina)",
    "Lucia Cevidanes (University of Michigan)",
    "Beatriz Paniagua (Kitware)",
    "Connor Bowley (Kitware)",
    "Antonio Ruellas (University of Michigan)",
    "Marcela Gurgel (University of Michigan)",
    "Marilia Yatabe (University of Michigan)",
    "Jonas Bianchi (University of Michigan)"]  # TODO: replace with "Firstname Lastname (Organization)"
    # TODO: update with short description of the module and a link to online module documentation
    self.parent.helpText = """
This extension provides a GUI for the deep learning method for jaw segmentation that we developed.
"""
    # TODO: replace with organization, grant and thanks
    self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""


#
# CrownSegmentationWidget
#

class InputChoice(Enum):
  VTK = 0
  MRML_NODE = 1

class CrownSegmentationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent=None):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)  # needed for parameter node observation

    self.logic = None
    self._parameterNode = None
    self._updatingGUIFromParameterNode = False
    self.fileName = ""
    self.surfaceFile = ""
    self.outputFolder = ""
    self.outputFile  = ""
    self.lArrays = []
    self.model = "" 
    self.resolution = 256
    self.predictedId = ""
    self.rotation = None
    self.inputChoice = InputChoice.VTK
    self.lNodes = []
    self.MRMLNode = None


  def setup(self):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.setup(self)

    # Load widget from .ui file (created by Qt Designer).
    # Additional widgets can be instantiated manually and added to self.layout.
    uiWidget = slicer.util.loadUI(self.resourcePath('UI/CrownSegmentation.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)


    # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
    # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
    # "setMRMLScene(vtkMRMLScene*)" slot.
    uiWidget.setMRMLScene(slicer.mrmlScene)

    # Create logic class. Logic implements all computations that should be possible to run
    # in batch mode, without a graphical user interface.
    self.logic = CrownSegmentationLogic()

    # Connections

    # These connections ensure that we update parameter node when scene is closed
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

    # UI elements
      
    # Inputs
    self.ui.applyChangesButton.connect('clicked(bool)',self.onApplyChangesButton)
    self.ui.rotationSpinBox.valueChanged.connect(self.onRotationSpinbox)
    self.ui.rotationSlider.valueChanged.connect(self.onRotationSlider)
    self.ui.browseSurfaceButton.connect('clicked(bool)',self.onBrowseSurfaceButton)
    self.ui.browseModelButton.connect('clicked(bool)',self.onBrowseModelButton)
    self.ui.surfaceLineEdit.textChanged.connect(self.onEditSurfaceLine)
    self.ui.modelLineEdit.textChanged.connect(self.onEditModelLine)    
    self.ui.githubButton.connect('clicked(bool)',self.onGithubButton)
    self.ui.surfaceComboBox.currentTextChanged.connect(self.onSurfaceModeChanged)
    self.ui.MRMLNodeComboBox.setMRMLScene(slicer.mrmlScene)
    self.ui.MRMLNodeComboBox.currentNodeChanged.connect(self.onNodeChanged)


    # Advanced 
    self.ui.predictedIdLineEdit.textChanged.connect(self.onEditPredictedIdLine)
    self.ui.resolutionComboBox.currentTextChanged.connect(self.onResolutionChanged)

    # Outputs 
    self.ui.browseOutputButton.connect('clicked(bool)',self.onBrowseOutputButton)
    self.ui.outputLineEdit.textChanged.connect(self.onEditOutputLine)
    self.ui.outputFileLineEdit.textChanged.connect(self.onEditOutputLine)
    self.ui.openOutButton.connect('clicked(bool)',self.onOutButton)

    self.ui.resetButton.connect('clicked(bool)',self.onReset)
    self.ui.cancelButton.connect('clicked(bool)', self.onCancel)

    self.ui.progressLabel.setHidden(True)
    self.ui.openOutButton.setHidden(True)
    self.ui.cancelButton.setHidden(True)
    self.ui.doneLabel.setHidden(True)
    self.ui.MRMLNodeComboBox.setHidden(True)

    #initialize variables
    self.model = self.ui.modelLineEdit.text
    self.surfaceFile = self.ui.surfaceLineEdit.text
    self.outputFolder = self.ui.outputLineEdit.text
    self.outputFile = self.ui.outputLineEdit.text + self.ui.outputFileLineEdit.text
    self.predictedId = self.ui.predictedIdLineEdit.text
    self.resolution = int(self.ui.resolutionComboBox.currentText)
    self.rotation = self.ui.rotationSlider.value
    self.MRMLNode = slicer.mrmlScene.GetNodeByID(self.ui.MRMLNodeComboBox.currentNodeID)
    #print(self.MRMLNode.GetName())

    # Make sure parameter node is initialized (needed for module reload)
    self.initializeParameterNode()

  def cleanup(self):
    """
    Called when the application closes and the module widget is destroyed.
    """
    self.removeObservers()

  def enter(self):
    """
    Called each time the user opens this module.
    """
    # Make sure parameter node exists and observed
    self.initializeParameterNode()

  def exit(self):
    """
    Called each time the user opens a different module.
    """
    # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
    self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

  def onSceneStartClose(self, caller, event):
    """
    Called just before the scene is closed.
    """
    # Parameter node will be reset, do not use it anymore
    self.setParameterNode(None)

  def onSceneEndClose(self, caller, event):
    """
    Called just after the scene is closed.
    """
    # If this module is shown while the scene is closed then recreate a new parameter node immediately
    if self.parent.isEntered:
      self.initializeParameterNode()

  def initializeParameterNode(self):
    """
    Ensure parameter node exists and observed.
    """
    # Parameter node stores all user choices in parameter values, node selections, etc.
    # so that when the scene is saved and reloaded, these settings are restored.

    self.setParameterNode(self.logic.getParameterNode())

    # Select default input nodes if nothing is selected yet to save a few clicks for the user
    if not self._parameterNode.GetNodeReference("InputVolume"):
      firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
      if firstVolumeNode:
        self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())

  def setParameterNode(self, inputParameterNode):
    """
    Set and observe parameter node.
    Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
    """

    # if inputParameterNode:
    #   self.logic.setDefaultParameters(inputParameterNode)

    # Unobserve previously selected parameter node and add an observer to the newly selected.
    # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
    # those are reflected immediately in the GUI.
    if self._parameterNode is not None:
      self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
    self._parameterNode = inputParameterNode
    if self._parameterNode is not None:
      self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    # Initial GUI update
    self.updateGUIFromParameterNode()

  def updateGUIFromParameterNode(self, caller=None, event=None):
    """
    This method is called whenever parameter node is changed.
    The module GUI is updated to show the current state of the parameter node.
    """

    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
    self._updatingGUIFromParameterNode = True


    # All the GUI updates are done
    self._updatingGUIFromParameterNode = False

  def updateParameterNodeFromGUI(self, caller=None, event=None):
    """
    This method is called when the user makes any change in the GUI.
    The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
    """

    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch


    self._parameterNode.EndModify(wasModified)


  def onResolutionChanged(self):
    self.resolution = int(self.ui.resolutionComboBox.currentText)


  def onProcessUpdate(self,caller,event):
    if self.logic.cliNode.GetStatus() & self.logic.cliNode.Completed:
      self.ui.applyChangesButton.setEnabled(True)
      self.ui.resetButton.setEnabled(True)
      self.ui.progressLabel.setHidden(False)         
      self.ui.cancelButton.setEnabled(False)
      self.ui.progressBar.setEnabled(False)
      self.ui.progressBar.setHidden(True)
      self.ui.progressLabel.setHidden(True)

      if os.path.isfile(self.outputFile): # if output file is found
        print('PROCESS DONE.')
        self.ui.doneLabel.setHidden(False)
        self.ui.openOutButton.setHidden(False) 

      else: # if no output file: error
        print ('Error: Output file was not found.') 
        msg = qt.QMessageBox()
        msg.setText("Output file was not found.\nThere may have been an error during prediction.")
        msg.setWindowTitle("Error")
        msg.exec_()

  def onProcessStarted(self):

    self.ui.cancelButton.setHidden(False)
    self.ui.cancelButton.setEnabled(True)
    self.ui.resetButton.setEnabled(False)
    self.ui.progressBar.setRange(0,0)
    self.ui.progressBar.setEnabled(True)
    self.ui.progressBar.setHidden(False)
    self.ui.progressBar.setTextVisible(True)
    self.ui.progressLabel.setHidden(False)


  def onOutButton(self):
    print(self.outputFile)
    jaw_model = slicer.util.loadModel(self.outputFile)
    print(type(jaw_model))


  def onGithubButton(self):
    webbrowser.open('https://github.com/MathieuLeclercq/fly-by-cnn/blob/master/src/py/FiboSeg/best_metric_model_segmentation2d_array_v2_5.pth')


  def onApplyChangesButton(self):
    print(self.inputChoice.name)
    if ((self.inputChoice is InputChoice.MRML_NODE and self.MRMLNode is not None) or os.path.isfile(self.surfaceFile))  and os.path.isdir(self.outputFolder) and os.path.isfile(self.model):
      self.ui.applyChangesButton.setEnabled(False)
      self.ui.progressBar.setEnabled(True)
      if self.inputChoice is InputChoice.VTK:
        self.logic = CrownSegmentationLogic(self.surfaceFile,self.outputFile,self.resolution, self.ui.rotationSpinBox.value,self.model, self.predictedId)
      else: # MRML node
        filename = self.writeVTKFromNode()
        self.logic = CrownSegmentationLogic(filename,self.outputFile,self.resolution, self.ui.rotationSpinBox.value,self.model, self.predictedId)

      self.ui.doneLabel.setHidden('True')
      self.ui.openOutButton.setHidden('True')
      self.logic.process()
      self.logic.cliNode.AddObserver('ModifiedEvent',self.onProcessUpdate)
      self.onProcessStarted()


    else:
      print('Error.')
      msg = qt.QMessageBox()
      if self.inputChoice is InputChoice.VTK and not(os.path.isfile(self.surfaceFile)):        
        msg.setText("Surface directory : \nIncorrect path.")
        print('Error: Incorrect path for surface directory.')
        self.ui.surfaceLineEdit.setText('')
        print(f'surface folder : {self.surfaceFile}')


      elif self.inputChoice is InputChoice.MRML_NODE and self.MRMLNode is None:        
        msg.setText("Input surface : \nPlease select a MRML node.")
        print('Error: No MRML node was selected.')
        self.ui.surfaceLineEdit.setText('')
        print(f'MRML node : {self.MRMLNode}')
     
      elif not(os.path.isdir(self.outputFolder)):
        msg.setText("Output directory : \nIncorrect path.")
        print('Error: Incorrect path for output directory.')
        self.ui.outputLineEdit.setText('')
        print(f'output folder : {self.outputFolder}')

      elif not(os.path.isfile(self.model)):
        msg.setText("Model : \nIncorrect path.")
        print('Error: Incorrect path for model.')
        self.ui.modelLineEdit.setText('')
        print(f'model path: {self.model}')

      else:
        msg.setText('Unknown error.')

      msg.setWindowTitle("Error")
      msg.exec_()

      return

  def onReset(self):
    self.ui.outputLineEdit.setText("")
    self.ui.surfaceLineEdit.setText("")
    self.ui.rotationSpinBox.value = 50
    self.ui.applyChangesButton.setEnabled(True)
    self.ui.progressLabel.setHidden(True)
    self.ui.openOutButton.setHidden(True)
    self.ui.progressBar.setValue(0)
    self.ui.doneLabel.setHidden(True)
    self.ui.surfaceComboBox.setCurrentIndex(0)

  def onCancel(self):
    self.logic.cliNode.Cancel()
    self.ui.applyChangesButton.setEnabled(True)
    self.ui.resetButton.setEnabled(True)
    self.ui.progressBar.setEnabled(False)
    self.ui.progressBar.setRange(0,100)
    self.ui.progressLabel.setHidden(True)
    self.ui.cancelButton.setEnabled(False)

    
    print("Process successfully cancelled.")


  def onBrowseSurfaceButton(self):
    newsurfaceFile = qt.QFileDialog.getOpenFileName(self.parent, "Select a surface",'',".vtk file (*.vtk)")
    if newsurfaceFile != '':
      self.surfaceFile = newsurfaceFile
      self.ui.surfaceLineEdit.setText(self.surfaceFile)
    #print(f'Surface directory : {self.surfaceFile}')


  def onBrowseModelButton(self):
    newModel = qt.QFileDialog.getOpenFileName(self.parent, "Select a model")
    if newModel != '':
      self.model = newModel
      self.ui.modelLineEdit.setText(self.model)
    #print(f'Surface directory : {self.surfaceFile}')



  def onBrowseOutputButton(self):
    newoutputFolder = qt.QFileDialog.getExistingDirectory(self.parent, "Select a directory")
    if newoutputFolder != '':
      if newoutputFolder[-1] != "/":
        newoutputFolder += '/'
      self.outputFolder = newoutputFolder
      print(self.outputFolder)
      self.ui.outputLineEdit.setText(self.outputFolder)
      print(self.outputFile)
    #print(f'Output directory : {self.outputFile}')      


  def onEditModelLine(self):
    self.model = self.ui.modelLineEdit.text


  def onEditPredictedIdLine(self):
    self.predictedId = self.ui.predictedIdLineEdit.text


  def onEditSurfaceLine(self):
    self.surfaceFile = self.ui.surfaceLineEdit.text    


  def onEditOutputLine(self): # called when either output folder line or output file line is modified
    self.outputFolder = self.ui.outputLineEdit.text
    self.outputFile = self.ui.outputLineEdit.text + self.ui.outputFileLineEdit.text


  def onRotationSlider(self):
    self.ui.rotationSpinBox.value = self.ui.rotationSlider.value
    self.rotation = self.ui.rotationSlider.value

  def onRotationSpinbox(self):
    self.ui.rotationSlider.value = self.ui.rotationSpinBox.value
    self.rotation = self.ui.rotationSlider.value

  def onSurfaceModeChanged(self):

    choice = self.ui.surfaceComboBox.currentText
    if choice == 'Select .vtk file':
      self.inputChoice = InputChoice.VTK
      self.ui.surfaceLineEdit.setHidden(False)
      self.ui.browseSurfaceButton.setHidden(False)
      self.surfaceFile = self.ui.surfaceLineEdit.text
    else:
      self.inputChoice = InputChoice.MRML_NODE
      self.ui.surfaceLineEdit.setHidden(True)
      self.ui.browseSurfaceButton.setHidden(True)
      self.ui.MRMLNodeComboBox.setHidden(False)

  def onNodeChanged(self):
    self.MRMLNode = slicer.mrmlScene.GetNodeByID(self.ui.MRMLNodeComboBox.currentNodeID)
    if self.MRMLNode is not None:
      print(self.MRMLNode.GetName())


  def writeVTKFromNode(self):
    poly = self.MRMLNode.GetPolyData()    
    filename = self.outputFile[0:-4]+"_input.vtk"
    print(filename)
    polydatawriter = vtk.vtkPolyDataWriter()
    polydatawriter.SetFileName(filename)
    polydatawriter.SetInputData(poly)
    polydatawriter.Write()
    return filename
      

#
# CrownSegmentationLogic
#

class CrownSegmentationLogic(ScriptedLoadableModuleLogic):
  """
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, surfaceFile= None,outputFile=None, resolution=None, rotation=None,model=None,predictedId=None):
    """
    Called when the logic class is instantiated. Can be used for initializing member variables.
    """
    ScriptedLoadableModuleLogic.__init__(self)
    self.surfaceFile = surfaceFile
    self.outputFile = outputFile
    self.resolution = resolution
    self.rotation = rotation
    self.model = model
    self.predictedId = predictedId
    self.nbOperation = 0
    self.progress = 0
    self.cliNode = None
    print(f"model: {self.model}")
    print(f'surfaceFile : {self.surfaceFile}')
    print(f'outptutfile : {self.outputFile}')
    print(f'resolution : {self.resolution}')
    print(f'rotation : {self.rotation}')
    print(f'predictedId : {self.predictedId}')


  # def setDefaultParameters(self, parameterNode):
  #   """
  #   Initialize parameter node with default settings.
  #   """

  def process(self):
    print('process')
    parameters = {}
    parameters ["surfaceFile"] = self.surfaceFile
    parameters ["outputFile"] = self.outputFile
    parameters ["rotation"] = self.rotation
    parameters ['resolution'] = self.resolution
    parameters ['model'] = self.model
    parameters ['predictedId'] = self.predictedId
    #parameters ['codePath'] = self.codePath
    env = slicer.util.startupEnvironment()
    print('\n\n\n\n')
    #print ('parameters : ', parameters)

    with open('env.json', 'w') as convert_file:
      convert_file.truncate(0)
      convert_file.write(json.dumps(env))
    
    flybyProcess = slicer.modules.crownsegmentationcli
    self.cliNode = slicer.cli.run(flybyProcess,None, parameters)    
    return flybyProcess


#
# CrownSegmentationTest
#

# class CrownSegmentationTest(ScriptedLoadableModuleTest):
#   """
#   This is the test case for your scripted module.
#   Uses ScriptedLoadableModuleTest base class, available at:
#   https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
#   """

#   def setUp(self):
#     """ Do whatever is needed to reset the state - typically a scene clear will be enough.
#     """
#     slicer.mrmlScene.Clear()

#   def runTest(self):
#     """Run as few or as many tests as needed here.
#     """
#     self.setUp()
#     self.test_CrownSegmentation1()

#   def test_CrownSegmentation1(self):
#     """ Ideally you should have several levels of tests.  At the lowest level
#     tests should exercise the functionality of the logic with different inputs
#     (both valid and invalid).  At higher levels your tests should emulate the
#     way the user would interact with your code and confirm that it still works
#     the way you intended.
#     One of the most important features of the tests is that it should alert other
#     developers when their changes will have an impact on the behavior of your
#     module.  For example, if a developer removes a feature that you depend on,
#     your test should break so they know that the feature is needed.
#     """

#     self.delayDisplay("Starting the test")

#     # Get/create input data

#     import SampleData
#     registerSampleData()
#     inputVolume = SampleData.downloadSample('CrownSegmentation1')
#     self.delayDisplay('Loaded test data set')

#     inputScalarRange = inputVolume.GetImageData().GetScalarRange()
#     self.assertEqual(inputScalarRange[0], 0)
#     self.assertEqual(inputScalarRange[1], 695)

#     outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
#     threshold = 100

#     # Test the module logic

#     # logic = CrownSegmentationLogic()

#     self.delayDisplay('Test passed')
