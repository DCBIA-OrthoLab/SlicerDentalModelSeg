import os
import glob
import vtk, qt, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin, pip_install
from enum import Enum
import subprocess
import platform

from CondaSetUp import  CondaSetUpCall,CondaSetUpCallWsl

import webbrowser
import csv
import io
import time
import threading
import sys

from pathlib import Path
import re
#
# CrownSegmentation
#


def func_import(install=False): 
  try : 
    import pkg_resources
    shapeaxi_version = pkg_resources.get_distribution("shapeaxi").version 
    print("Distribution    Found for shapeaxi")
    # pytorch3d = pkg_resources.get_distribution("pytorch3d").version 
    import pytorch3d
    return True
  except : 
    print("ERROR")
    if install :
      pip_install("shapeaxi")
      try:
        import torch
      except ImportError:
        pip_install(f'torch')
      try:
        import pytorch3d
      except ImportError:
        try : 
          import torch
          pyt_version_str = torch.__version__.split("+")[0].replace(".", "")
          version_str = "".join([f"py3{sys.version_info.minor}_cu", torch.version.cuda.replace(".", ""), f"_pyt{pyt_version_str}"])
          pip_install('--upgrade pip')
          pip_install('fvcore==0.1.5.post20220305')
          pip_install(f'--no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html')
        except:
            pip_install('--no-cache-dir torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113')
            pip_install('--no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1110/download.html')

      return True
    return False
    

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
This extension provides a GUI for a deep learning automated teeth segmentation algorithm. The inputs are 3D IOS scans, and 
the dental crowns are segmented according to the <a href="https://en.wikipedia.org/wiki/Universal_Numbering_System">Universal Number System</a>. <br>

<h2 style="color: #2e6c80;">Running the module :</h2>
 <br> <br>
- The input file must be a .vtk file or a MRMLModelNode of a IOS scan for a lower or upper jaw. The model works better with models of jaws with no wisdom teeth. 
You can find examples in the "Examples" folder. <br> <br> 

- Number of views: this sets the number of 2D views used for one prediction. A low number takes less time to compute, but results can be inaccurate.<br> <br>

- Model for segmentation: this is the path for the neural network model. Resolution: This sets the resolution of the 2D views. 320 px is recommended. 
Name of predicted labels: this is the name the array with the predicted labels on the output vtk file.   <br><br> 



<span style="color: ##2e6c80;"><strong>More options can be found in the "Advanced" section:</strong></span> <br><br> 


- Resolution: this sets the resolution of the 2D views used for the prediction. This should usually be set to 320px. <br><br>

- Name of predicted labels: The name of the VTK array that stores the labels for each vertex in the output surface file. <br><br>

- "Install/Check dependencies" button: This forces the installation of all dependencies. If you don't use this button the first time you run a prediction, it will 
automatically install all dependencies before starting the prediction. <br><br>

- "Create one output file for each label": Check this box if you want one separate output file for each tooth. <br><br>

- "Numbering system": lets you choose between <a href="https://en.wikipedia.org/wiki/Universal_Numbering_System">Universal Number System</a> and <a href="https://en.wikipedia.org/wiki/FDI_World_Dental_Federation_notation">FDI notation</a>.
<br><br>

When prediction is over, you can open the output surface as a MRML node in Slicer by pushing the "Open output surface".
You can change the color table in Slicer's "Models" module. <br><br>

More help can be found on the <a href="https://github.com/DCBIA-OrthoLab/SlicerDentalModelSeg">Github repository</a> for the extension.
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
  FOLDER = 2

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
    self.input = ""
    self.outputFolder = ""
    self.output  = ""
    self.lArrays = []
    self.model = "" 
    self.nbFiles = 1
    self.resolution = 256
    self.predictedId = ""
    self.rotation = None
    self.inputChoice = InputChoice.VTK
    self.lNodes = []
    self.MRMLNode = None
    self.log_path = os.path.join(slicer.util.tempDirectory(), 'process.log')
    self.time_log = 0 # for progress bar
    self.progress = 0
    self.currentPredDict = {}
    self.previous_time = 0
    self.start_time = 0


  def setup(self):
    self.removeObservers()
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
    
    self.ui.dependenciesButton.connect('clicked(bool)',self.checkDependencies)

    # Inputs
    self.ui.applyChangesButton.connect('clicked(bool)',self.onApplyChangesButton)
    self.ui.rotationSpinBox.valueChanged.connect(self.onRotationSpinbox)
    self.ui.rotationSlider.valueChanged.connect(self.onRotationSlider)
    self.ui.browseSurfaceButton.connect('clicked(bool)',self.onBrowseSurfaceButton)
    self.ui.inputFolderPushButton.connect('clicked(bool)',self.onBrowseInputFolderButton)
    self.ui.browseModelButton.connect('clicked(bool)',self.onBrowseModelButton)
    self.ui.surfaceLineEdit.textChanged.connect(self.onEditSurfaceLine)
    self.ui.inputFolderLineEdit.textChanged.connect(self.onEditInputFolderLine)
    self.ui.modelLineEdit.textChanged.connect(self.onEditModelLine)    
    self.ui.githubButton.connect('clicked(bool)',self.onGithubButton)
    self.ui.checkBoxLatestModel.stateChanged.connect(self.useLatestModel)
    self.ui.checkBoxOverwrite.stateChanged.connect(self.overwrite)
    self.ui.surfaceComboBox.currentTextChanged.connect(self.onSurfaceModeChanged)
    self.ui.MRMLNodeComboBox.setMRMLScene(slicer.mrmlScene)
    self.ui.MRMLNodeComboBox.currentNodeChanged.connect(self.onNodeChanged)
    self.ui.MRMLNodeComboBox.setHidden(True)
    self.ui.inputFolderLineEdit.setHidden(True)
    self.ui.inputFolderPushButton.setHidden(True)

    # Advanced 
    self.ui.advancedCollapsibleButton.collapsed = 1 # Set to 1
    self.ui.predictedIdLineEdit.textChanged.connect(self.onEditPredictedIdLine)
    self.ui.resolutionComboBox.currentTextChanged.connect(self.onResolutionChanged)
    self.ui.installProgressBar.setEnabled(False)
    self.ui.installSuccessLabel.setHidden(True)
    self.ui.labelComboBox.currentTextChanged.connect(self.onFDI)

    # Outputs 
    self.ui.browseOutputButton.connect('clicked(bool)',self.onBrowseOutputButton)
    self.ui.outputLineEdit.textChanged.connect(self.onEditOutputLine)
    self.ui.outputFileLineEdit.textChanged.connect(self.onEditOutputLine)
    self.ui.openOutSurfButton.connect('clicked(bool)',self.onOpenOutSurfButton)
    self.ui.openOutFolderButton.connect('clicked(bool)',self.onOpenOutFolderButton)
    self.ui.resetButton.connect('clicked(bool)',self.onReset)
    self.ui.cancelButton.connect('clicked(bool)', self.onCancel)
    self.ui.progressLabel.setHidden(True)
    self.ui.openOutSurfButton.setHidden(True)
    self.ui.openOutFolderButton.setHidden(True)
    self.ui.cancelButton.setHidden(True)
    self.ui.doneLabel.setHidden(True)
    self.ui.githubButton.setHidden(True)
    self.ui.timeLabel.setHidden(True)
    self.ui.progressBar.setHidden(True)
    

    #initialize variables
    if qt.QSettings().value('TeethSeg_ModelPath') != None:
      self.ui.modelLineEdit.setText(qt.QSettings().value('TeethSeg_ModelPath'))
    self.model = self.ui.modelLineEdit.text
    self.input = self.ui.surfaceLineEdit.text
    self.outputFolder = self.ui.outputLineEdit.text
    self.output = self.ui.outputLineEdit.text + self.ui.outputFileLineEdit.text
    self.predictedId = self.ui.predictedIdLineEdit.text
    self.resolution = int(self.ui.resolutionComboBox.currentText)
    self.rotation = self.ui.rotationSlider.value
    self.MRMLNode = slicer.mrmlScene.GetNodeByID(self.ui.MRMLNodeComboBox.currentNodeID)
    self.chooseFDI = self.ui.labelComboBox.currentIndex
    #print(self.MRMLNode.GetName())

    #Hidden
    self.ui.rotationSpinBox.setHidden(True)
    self.ui.label_3.setHidden(True)
    self.ui.rotationSlider.setHidden(True)
    self.ui.resolutionComboBox.setHidden(True)
    self.ui.label.setHidden(True)
    self.ui.dependenciesButton.setHidden(True)
    self.ui.installProgressBar.setHidden(True)



    # qt.QSettings().setValue("TeethSegVisited",None)
    if qt.QSettings().value('TeethSegVisited') is None:
        self.msg = qt.QMessageBox()
        self.msg.setText(f'Welcome to this module!\n'
          'The module works with Linux only. You also need a CUDA capable GPU.\n'
          'If you are running it for the first time, The installation of the dependencies will take time.\n' )
        self.msg.setWindowTitle("Welcome!")
        self.cb = qt.QCheckBox()
        self.cb.setText("Don't show this again")
        self.cb.stateChanged.connect(self.onCBchecked)
        self.msg.setCheckBox(self.cb)
        self.msg.exec_()


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





  def onCBchecked(self):
    state = self.cb.checkState()
    if state==0:
      qt.QSettings().setValue("TeethSegVisited",None)
    else:
      qt.QSettings().setValue("TeethSegVisited",1)

      

  ###
  ### INPUTS
  ###

  def onBrowseSurfaceButton(self):
    newsurfaceFile = qt.QFileDialog.getOpenFileName(self.parent, "Select a surface", '', "VTK and STL files (*.vtk *.stl)")
    if newsurfaceFile != '':
      self.input = newsurfaceFile
      self.ui.surfaceLineEdit.setText(self.input)

    if self.ui.checkBoxOverwrite.checked :
      self.ui.outputLineEdit.setText(os.path.dirname(self.ui.surfaceLineEdit.text))
    #print(f'Surface directory : {self.surfaceFile}')


  def onBrowseInputFolderButton(self):
    newInputFolder = qt.QFileDialog.getExistingDirectory(self.parent, "Select a directory")
    if newInputFolder != '':
      self.input = newInputFolder
      print(self.input)
      self.ui.inputFolderLineEdit.setText(self.input)

    if self.ui.checkBoxOverwrite.checked :
      self.ui.outputLineEdit.setText(self.ui.inputFolderLineEdit.text)

    
    #print(f'Output directory : {self.output}')   


  def onBrowseModelButton(self):
    newModel = qt.QFileDialog.getOpenFileName(self.parent, "Select a model")
    if newModel != '':
      self.model = newModel
      self.ui.modelLineEdit.setText(self.model)
    #print(f'Surface directory : {self.surfaceFile}')

  def overwrite(self):
    if self.ui.checkBoxOverwrite.checked:
      self.ui.outputFileLineEdit.setEnabled(False)
      self.ui.outputLineEdit.setEnabled(False)
      self.ui.outputFileLineEdit.setText("None")
      print(self.ui.surfaceLineEdit.text)
      print(self.ui.surfaceComboBox.currentText)
      if self.ui.surfaceComboBox.currentText=="Select file":
        self.ui.outputLineEdit.setText(os.path.dirname(self.ui.surfaceLineEdit.text))
      else : 
        self.ui.outputLineEdit.setText(self.ui.inputFolderLineEdit.text)
        print(self.ui.inputFolderLineEdit.text)


    else : 
      self.ui.outputFileLineEdit.setEnabled(True)
      self.ui.outputLineEdit.setEnabled(True)
      self.ui.outputFileLineEdit.setText("predict")


  def useLatestModel(self):
    if self.ui.checkBoxLatestModel.checked:
      self.ui.browseModelButton.setEnabled(False)
      self.ui.modelLineEdit.setEnabled(False)
      self.ui.modelLineEdit.setText("latest")

    else : 
      self.ui.browseModelButton.setEnabled(True)
      self.ui.modelLineEdit.setEnabled(True)


  def onGithubButton(self):
    # webbrowser.open('https://github.com/MathieuLeclercq/fly-by-cnn/blob/master/src/py/FiboSeg/best_metric_model_segmentation2d_array_v2_5.pth')
    # webbrowser.open('https://github.com/MathieuLeclercq/fly-by-cnn/blob/master/src/py/challenge-teeth/checkpoints/07-21-22_val-loss0.169.pth')
    # webbrowser.open('https://github.com/DCBIA-OrthoLab/SlicerDentalModelSeg/releases/tag/v3.0')
    webbrowser.open('https://github.com/DCBIA-OrthoLab/Fly-by-CNN/releases/tag/3.0')


  def onEditModelLine(self):
    self.model = self.ui.modelLineEdit.text


  def onEditSurfaceLine(self):
    self.input = self.ui.surfaceLineEdit.text 


  def onEditInputFolderLine(self):
    self.input = self.ui.inputFolderLineEdit.text 

  def onRotationSlider(self):
    self.ui.rotationSpinBox.value = self.ui.rotationSlider.value
    self.rotation = self.ui.rotationSlider.value

  def onRotationSpinbox(self):
    self.ui.rotationSlider.value = self.ui.rotationSpinBox.value
    self.rotation = self.ui.rotationSlider.value

  def onSurfaceModeChanged(self):
    choice = self.ui.surfaceComboBox.currentText
    self.input = ""
    self.ui.MRMLNodeComboBox.setHidden(True)
    self.ui.surfaceLineEdit.setHidden(True)
    self.ui.browseSurfaceButton.setHidden(True)
    self.ui.inputFolderLineEdit.setHidden(True)
    self.ui.inputFolderPushButton.setHidden(True)
    self.ui.outputFileLineEdit.setHidden(False)
    self.ui.outputFileLabel.setHidden(False)



    if choice == 'Select file':
      self.inputChoice = InputChoice.VTK
      self.ui.surfaceLineEdit.setHidden(False)
      self.ui.browseSurfaceButton.setHidden(False)
      self.input = self.ui.surfaceLineEdit.text
      self.ui.inputFolderLineEdit.setText("")
    elif choice == 'Select MRMLModelNode':
      self.inputChoice = InputChoice.MRML_NODE
      self.ui.MRMLNodeComboBox.setHidden(False)
    else: # Select folder 
      self.inputChoice = InputChoice.FOLDER
      self.ui.inputFolderLineEdit.setHidden(False)
      self.ui.inputFolderPushButton.setHidden(False)
      self.input = self.ui.inputFolderLineEdit.text
      # self.ui.outputFileLineEdit.setText("")
      # self.ui.surfaceLineEdit.setText("")
      # self.ui.outputFileLabel.setHidden(True)
      # self.ui.outputFileLineEdit.setHidden(True)




  def onNodeChanged(self):
    self.MRMLNode = slicer.mrmlScene.GetNodeByID(self.ui.MRMLNodeComboBox.currentNodeID)
    if self.MRMLNode is not None:
      print(self.MRMLNode.GetName())


  def writeVTKFromNode(self):
    poly = self.MRMLNode.GetPolyData()    
    filename = self.output[0:-4]+"_input.vtk"
    print(filename)
    polydatawriter = vtk.vtkPolyDataWriter()
    polydatawriter.SetFileName(filename)
    polydatawriter.SetInputData(poly)
    polydatawriter.Write()
    return filename

  ###
  ### ADVANCED
  ###


  def onEditPredictedIdLine(self):
    self.predictedId = self.ui.predictedIdLineEdit.text


  def onResolutionChanged(self):
    self.resolution = int(self.ui.resolutionComboBox.currentText)

  def checkDependencies(self): #TODO: ALSO CHECK FOR CUDA 
    self.ui.dependenciesButton.setEnabled(False)
    self.ui.applyChangesButton.setEnabled(False)
    self.ui.installProgressBar.setEnabled(True)
    self.installLogic = CrownSegmentationLogic('-1',0,0,0,0,0,0,0,0) # -1: flag so that CLI module knows it's only to install dependencies
    self.installLogic.process()
    self.ui.installProgressBar.setRange(0,0)
    self.installObserver = self.installLogic.cliNode.AddObserver('ModifiedEvent',self.onInstallationProgress)
    

  def onInstallationProgress(self,caller,event):
    if self.installLogic.cliNode.GetStatus() & self.installLogic.cliNode.Completed:
      if self.installLogic.cliNode.GetStatus() & self.installLogic.cliNode.ErrorsMask:
        # error
        errorText = self.installLogic.cliNode.GetErrorText()
        print("CLI execution failed: \n \n" + errorText)
        msg = qt.QMessageBox()
        msg.setText(f'There was an error during the installation:\n \n {errorText} ')
        msg.setWindowTitle("Error")
        msg.exec_()
      else:
        # success
        print('SUCCESS')
        print(self.installLogic.cliNode.GetOutputText())
        self.ui.installSuccessLabel.setHidden(False)
      self.ui.installProgressBar.setRange(0,100)
      self.ui.installProgressBar.setEnabled(False)
      self.ui.dependenciesButton.setEnabled(True)
      self.ui.applyChangesButton.setEnabled(True)


  def onFDI(self):
    self.chooseFDI = self.ui.labelComboBox.currentIndex
    print(f'chooseFDI: {self.chooseFDI}')



  ###
  ### OUTPUTS
  ###

  def onOpenOutSurfButton(self):
    print(self.currentPredDict["output"])
    jaw_model = slicer.util.loadModel(self.currentPredDict["output"])
    jaw_model.GetDisplayNode().SetActiveScalar(self.currentPredDict["PredictedID"], vtk.vtkAssignAttribute.POINT_DATA)
    jaw_model.GetDisplayNode().SetAndObserveColorNodeID("vtkMRMLColorTableNodeFileViridis.txt")
    jaw_model.GetDisplayNode().SetScalarVisibility(True)

  def onOpenOutFolderButton(self):
    webbrowser.open(self.output)

  def onBrowseOutputButton(self):
    newoutputFolder = qt.QFileDialog.getExistingDirectory(self.parent, "Select a directory")
    if newoutputFolder != '':
      if newoutputFolder[-1] != "/":
        newoutputFolder += '/'
      self.outputFolder = newoutputFolder
      print(self.outputFolder)
      self.ui.outputLineEdit.setText(self.outputFolder)
      print(self.output)
    #print(f'Output directory : {self.output}')   


  def onEditOutputLine(self): # called when either output folder line or output file line is modified
    self.outputFolder = self.ui.outputLineEdit.text
    self.output = self.ui.outputLineEdit.text + self.ui.outputFileLineEdit.text
  ###
  ### PROCESS
  ###



  def onApplyChangesButton(self):
    '''
    This function is called when the user want to run dentalmodelseg
    For Linux system : - check the installation of shapeaxi and pytorch3d in Slicer, if no with consent of user it install them
                       - run crownsegmentationcli as a module of Slicer
                       
    For Windows sytem : - check the installation of wsl, if no show a message asking asking for the user to do it and stop the process
                        - check the installation of miniconda in wsl, if no show a message asking for the user to do it and sop the process
                        - check if the environnement "shapeaxi" exist, if no it will create it (with consent of user) with the required librairies (shapeaxi and pytorch3d)
                        - run the file CrownSegmentationcli.py into wsl in the environment 'shapeaxi'
    '''
    self.ui.applyChangesButton.setEnabled(False)
    #if ((self.inputChoice is InputChoice.MRML_NODE and self.MRMLNode is not None) or os.path.isfile(self.input) or os.path.isdir(self.input))  and os.path.isdir(self.outputFolder) and os.path.isfile(self.model):
    if not(os.path.isdir(self.outputFolder) and (os.path.isfile(self.model) or self.model=="latest")):
      print('Error.')
      msg = qt.QMessageBox()
      if not(os.path.isdir(self.outputFolder)):
        msg.setText("Output directory : \nIncorrect path.")
        print('Error: Incorrect path for output directory.')
        self.ui.outputLineEdit.setText('')
        print(f'output folder : {self.outputFolder}')

      elif (not(os.path.isfile(self.model)) and self.model!="latest"):
        msg.setText("Model : \nIncorrect path.")
        print('Error: Incorrect path for model.')
        self.ui.modelLineEdit.setText('')
        print(f'model path: {self.model}')

      else:
        msg.setText('Unknown error.')

      msg.setWindowTitle("Error")
      msg.exec_()
      return

    elif not((self.inputChoice is InputChoice.MRML_NODE and self.MRMLNode is not None) or os.path.isfile(self.input) or os.path.isdir(self.input)):
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

      else:
        msg.setText('Unknown error.')

      msg.setWindowTitle("Error")
      msg.exec_()
      return

    else:
        # create input parameters
        surf = "None"
        input_csv = "None"
        vtk_folder = "None"
        if os.path.isfile(self.input):
            extension = os.path.splitext(self.input)[1]
            if extension == ".vtk" or extension == ".stl":
              surf = self.input
              
        elif os.path.isdir(self.input):
          input_csv = self.create_csv()
          vtk_folder = self.input

        self.ui.timeLabel.setHidden(False)
        if platform.system() != "Windows" : #if linux system
            env_ok = func_import(False)
            if not env_ok : 
              userResponse = slicer.util.confirmYesNoDisplay("Some of the required libraries are not installed in Slicer. Would you like to install them?\nThis may take a few minutes.", windowTitle="Env doesn't exist")
              if userResponse : 
                self.parall_process(func_import,[True],"Installing the required packages in Slicer")
                env_ok = True
                # env_ok=func_import(True)
            if env_ok : 
              slicer_path = slicer.app.applicationDirPath()
              dentalmodelseg_path = os.path.join(slicer_path,"..","lib","Python","bin","dentalmodelseg")
              print("dentalmodelseg_path : ",dentalmodelseg_path)
              self.logic = CrownSegmentationLogic(surf,
                                              input_csv, 
                                              self.ui.outputLineEdit.text,
                                              "1" if self.ui.checkBoxOverwrite.checked else "0", 
                                              self.model, 
                                              "1" if self.ui.sepOutputsCheckbox.isChecked() else "0",
                                              self.predictedId,
                                              self.chooseFDI,
                                              self.ui.outputFileLineEdit.text,
                                              vtk_folder,
                                              dentalmodelseg_path)


              
              self.logic.process()
              self.addObserver(self.logic.cliNode,vtk.vtkCommand.ModifiedEvent,self.onProcessUpdate)
              self.onProcessStarted()
              file_path = os.path.abspath(__file__)
              folder_path = os.path.dirname(file_path)
              
        else : # if windows system
            # The code is run on wsl into an environment 'shapeaxi'
            
            self.conda_wsl = CondaSetUpCallWsl()  
            wsl = self.conda_wsl.testWslAvailable()
            ready = True
            self.ui.timeLabel.setHidden(False)
            self.ui.timeLabel.setText(f"Checking if wsl is installed, this task may take a moments")
            slicer.app.processEvents()
            if wsl : # if wsl is install
              lib = self.check_lib_wsl()
              if not lib : # if lib required are not install
                  self.ui.timeLabel.setText(f"Checking if the required librairies are installed, this task may take a moments")
                  messageBox = qt.QMessageBox()
                  text = "Code can't be launch. \nWSL doen't have all the necessary libraries, please download the installer and follow the instructin here : https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/wsl2_windows/installer_wsl2.zip\nDownloading may be blocked by Chrome, this is normal, just authorize it."
                  ready = False
                  messageBox.information(None, "Information", text)
            else : # if wsl not install, ask user to install it ans stop process
              messageBox = qt.QMessageBox()
              text = "Code can't be launch. \nWSL is not installed, please download the installer and follow the instructin here : https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/wsl2_windows/installer_wsl2.zip\nDownloading may be blocked by Chrome, this is normal, just authorize it."
              ready = False
              messageBox.information(None, "Information", text)
            
            if ready : # checking if miniconda installed on wsl
              self.ui.timeLabel.setText(f"Checking if miniconda is installed")
              if "Error" in self.conda_wsl.condaRunCommand([self.conda_wsl.getCondaExecutable(),"--version"]): # if conda is setup
                    messageBox = qt.QMessageBox()
                    text = "Code can't be launch. \nConda is not setup in WSL. Please go the extension CondaSetUp in SlicerConda to do it."
                    ready = False
                    messageBox.information(None, "Information", text)
            
            if ready : # checking if environment 'shapeaxi' exist on wsl and if no ask user permission to create and install required lib in it
              self.ui.timeLabel.setText(f"Checking if environnement exist")
              if not self.conda_wsl.condaTestEnv('shapeaxi') : # check is environnement exist, if not ask user the permission to do it
                userResponse = slicer.util.confirmYesNoDisplay("The environnement to run the segmentation doesn't exist, do you want to create it ? ", windowTitle="Env doesn't exist")
                if userResponse :
                  start_time = time.time()
                  previous_time = start_time
                  self.ui.timeLabel.setText(f"Creation of the new environment. This task may take a few minutes.\ntime: 0.0s")
                  name_env = "shapeaxi"
                  process = threading.Thread(target=self.conda_wsl.condaCreateEnv, args=(name_env,"3.9",["shapeaxi"],)) #run in paralle to not block slicer
                  process.start()
                  
                  while process.is_alive():
                    slicer.app.processEvents()
                    current_time = time.time()
                    gap=current_time-previous_time
                    if gap>0.3:
                        previous_time = current_time
                        elapsed_time = current_time - start_time
                        self.ui.timeLabel.setText(f"Creation of the new environment. This task may take a few minutes.\ntime: {elapsed_time:.1f}s")
              
                  start_time = time.time()
                  previous_time = start_time
                  self.ui.timeLabel.setText(f"Installation of librairies into the new environnement. This task may take a few minutes.\ntime: 0.0s")
                  name_env = "shapeaxi"
                  # file_path = os.path.realpath(__file__)
                  # folder = os.path.dirname(file_path)
                  # utils_folder = os.path.join(folder, "utils")
                  # utils_folder_norm = os.path.normpath(utils_folder)
                  # install_path = self.windows_to_linux_path(os.path.join(utils_folder_norm, 'install_pytorch.py'))
                  
                  PYTHONPATH = os.environ.get('PYTHONPATH')
                  search_term = "SlicerDentalModelSeg"
                  filename_to_search = "install_pytorch.py"
                  install_path = self.windows_to_linux_path(self.filter_and_search_paths(PYTHONPATH, search_term, filename_to_search))
                  path_pip = self.conda_wsl.getCondaPath()+"/envs/shapeaxi/bin/pip"
                  process = threading.Thread(target=self.conda_wsl.condaRunFilePython, args=(install_path,name_env,[path_pip],)) # launch install_pythorch.py with the environnement ali_ios to install pytorch3d on it
                  process.start()
                  
                  while process.is_alive():
                    slicer.app.processEvents()
                    current_time = time.time()
                    gap=current_time-previous_time
                    if gap>0.3:
                        previous_time = current_time
                        elapsed_time = current_time - start_time
                        self.ui.timeLabel.setText(f"Installation of librairies into the new environnement. This task may take a few minutes.\ntime: {elapsed_time:.1f}s")
                else :
                    ready = False
                    
            if ready : # if everything is ready launch dentalmodelseg on the environnement shapeaxi in wsl
              model = self.model
              if self.model == "latest":
                model = None
              else :
                model = self.windows_to_linux_path(model)

              name_env = "shapeaxi"

              # Creation of path to crownsegmentationcli.py
              # file_path = os.path.realpath(__file__)
              # folder = os.path.dirname(file_path)
              # cli_folder = os.path.join(folder, '../CrownSegmentationcli')
              # clis_folder_norm = os.path.normpath(cli_folder)
              # cli_path = os.path.join(clis_folder_norm, 'CrownSegmentationcli.py')
              
              PYTHONPATH = os.environ.get('PYTHONPATH')
              search_term = "SlicerDentalModelSeg"
              filename_to_search = "CrownSegmentationcli.py"
              cli_path = self.windows_to_linux_path(self.filter_and_search_paths(PYTHONPATH, search_term, filename_to_search))
              
              # Creation path in wsl to dentalmodelseg
              output_command = self.conda_wsl.condaRunCommand(["which","dentalmodelseg"],"shapeaxi").strip()
              clean_output = re.search(r"Result: (.+)", output_command)
              dentalmodelseg_path = clean_output.group(1).strip()
              dentalmodelseg_path_clean = dentalmodelseg_path.replace("\\n","")
              
              
              args = [surf,
                      input_csv,
                      self.ui.outputLineEdit.text,
                      "1" if self.ui.checkBoxOverwrite.checked else "0", 
                      self.model, 
                      "1" if self.ui.sepOutputsCheckbox.isChecked() else "0",
                      str(self.predictedId),
                      str(self.chooseFDI),
                      self.ui.outputFileLineEdit.text,
                      vtk_folder,
                      dentalmodelseg_path_clean]
              
              # Print args and path python to be run
              print(f"cli_path : {cli_path}")
              print(f"args : {args}")

              # running in // to not block Slicer
              process = threading.Thread(target=self.conda_wsl.condaRunFilePython, args=(cli_path,"shapeaxi",args))
              process.start()
              self.ui.applyChangesButton.setEnabled(False)
              self.ui.doneLabel.setHidden(True)
              self.ui.timeLabel.setHidden(False)
              self.ui.progressLabel.setHidden(False)
              self.ui.timeLabel.setText(f"time : 0.00s")
              start_time = time.time()
              previous_time = start_time
              while process.is_alive():
                  slicer.app.processEvents()
                  current_time = time.time()
                  gap=current_time-previous_time
                  if gap>0.3:
                      previous_time = current_time
                      elapsed_time = current_time - start_time
                      self.ui.timeLabel.setText(f"time : {elapsed_time:.2f}s")

              
              self.ui.progressLabel.setHidden(True)
              self.ui.doneLabel.setHidden(False)
              self.ui.applyChangesButton.setEnabled(True)

              # Delete csv file
              file_path = os.path.abspath(__file__)
              folder_path = os.path.dirname(file_path)
              csv_file = os.path.join(folder_path,"list_file.csv")
              if os.path.exists(csv_file):
                os.remove(csv_file)

    self.ui.applyChangesButton.setEnabled(True)
    
    def find_file_in_subdirectories(self,root_path, filename):
        """
        Search recursively for a file in given root directory and its subdirectories.
        """
        for dirpath, dirnames, files in os.walk(root_path):
            if filename in files:
                return os.path.join(dirpath, filename)
        return None

    def filter_and_search_paths(self,pythonpath, search_term, filename):
        """
        Filter paths containing a specific term, crop the path from that term,
        and search for a specific file in that directory and its subdirectories.
        """
        # Split the pythonpath into individual paths
        paths = pythonpath.split(';')
        
        # Filter and modify paths
        filtered_paths = [path.split(search_term)[0] + search_term for path in paths if search_term in path]
        
        # Search for the file in each path
        for path in filtered_paths:
            found_file = self.find_file_in_subdirectories(path, filename)
            if found_file:
                return found_file
        
        return "File not found in any specified directories."
    
  def parall_process(self,function,arguments=[],message=""):
        process = threading.Thread(target=function, args=tuple(arguments)) #run in paralle to not block slicer
        process.start()
        start_time = time.time()
        previous_time = time.time()
        while process.is_alive():
          slicer.app.processEvents()
          current_time = time.time()
          gap=current_time-previous_time
          if gap>0.3:
              previous_time = current_time
              elapsed_time = current_time - start_time
              self.ui.timeLabel.setText(f"{message}\ntime: {elapsed_time:.1f}s")
          
  def windows_to_linux_path(self,windows_path):
      '''
      Convert a windows path to a wsl path
      '''
      windows_path = windows_path.strip()

      path = windows_path.replace('\\', '/')

      if ':' in path:
          drive, path_without_drive = path.split(':', 1)
          path = "/mnt/" + drive.lower() + path_without_drive

      return path
              
  def check_lib_wsl(self)->bool:
      '''
      Check if wsl contains the require librairies
      '''
      result1 = subprocess.run("wsl -- bash -c \"dpkg -l | grep libxrender1\"", capture_output=True, text=True)
      output1 = result1.stdout.encode('utf-16-le').decode('utf-8')
      clean_output1 = output1.replace('\x00', '')

      result2 = subprocess.run("wsl -- bash -c \"dpkg -l | grep libgl1-mesa-glx\"", capture_output=True, text=True)
      output2 = result2.stdout.encode('utf-16-le').decode('utf-8')
      clean_output2 = output2.replace('\x00', '')

      return "libxrender1" in clean_output1 and "libgl1-mesa-glx" in clean_output2



  def create_csv(self):
    '''
    create a csv with the complete path of the files in the folder
    '''
    file_path = os.path.abspath(__file__)
    folder_path = os.path.dirname(file_path)
    csv_file = os.path.join(folder_path,"list_file.csv")
    with open(csv_file, 'w', newline='') as fichier:
        writer = csv.writer(fichier)
        # Écrire l'en-tête du CSV
        writer.writerow(["surf"])

        # Parcourir le dossier et ses sous-dossiers
        for root, dirs, files in os.walk(self.input):
            for file in files:
                if file.endswith(".vtk") or file.endswith(".stl"):
                    # Écrire le chemin complet du fichier dans le CSV
                    if platform.system() != "Windows" :    
                      writer.writerow([os.path.join(root, file)])
                    else :
                      file_path = os.path.join(root, file)
                      norm_file_path = os.path.normpath(file_path)
                      writer.writerow([self.windows_to_linux_path(norm_file_path)])


    return csv_file


  def is_ubuntu_installed(self)->bool:
      '''
      Check if wsl is install with Ubuntu
      '''
      result = subprocess.run(['wsl', '--list'], capture_output=True, text=True)
      output = result.stdout.encode('utf-16-le').decode('utf-8')
      clean_output = output.replace('\x00', '')  # Enlève tous les octets null

      return 'Ubuntu' in clean_output

  def onProcessStarted(self):
    self.start_time = time.time()
    self.previous_time = self.start_time  
    
    self.ui.applyChangesButton.setEnabled(False)
    self.ui.doneLabel.setHidden(True)
    self.ui.timeLabel.setHidden(False)
    self.ui.progressLabel.setHidden(False)
    self.ui.timeLabel.setText(f"time : 0.00s") 
     



  def onProcessUpdate(self,caller,event):
    # check log file
    current_time = time.time()
    gap = current_time - self.previous_time
    if gap > 0.3:
        self.previous_time = current_time
        elapsed_time = current_time - self.start_time
        self.ui.timeLabel.setText(f"Segmentation in process\ntime : {elapsed_time:.2f}s")
    


    if self.logic.cliNode.GetStatus() & self.logic.cliNode.Completed:
      # process complete
      self.ui.applyChangesButton.setEnabled(True)
      self.ui.resetButton.setEnabled(True)
      self.ui.progressLabel.setHidden(False)         
      self.ui.cancelButton.setEnabled(False)
      self.ui.progressBar.setEnabled(False)
      self.ui.progressBar.setHidden(True)
      self.ui.progressLabel.setHidden(True)

      if self.logic.cliNode.GetStatus() & self.logic.cliNode.ErrorsMask:
        # error
        errorText = self.logic.cliNode.GetErrorText()
        print("CLI execution failed: \n \n" + errorText)
        msg = qt.QMessageBox()
        msg.setText(f'There was an error during the process:\n \n {errorText} ')
        msg.setWindowTitle("Error")
        msg.exec_()

      else:
        # success
        print('PROCESS DONE.')

        self.ui.progressLabel.setHidden(True)
        self.ui.doneLabel.setHidden(False)
        self.ui.applyChangesButton.setEnabled(True)
        print("Process completed successfully.")
        self.ui.timeLabel.setText(f"time : {elapsed_time:.2f}s")

          
        print("*"*25,"Output cli","*"*25)
        print(self.logic.cliNode.GetOutputText())
        
        file_path = os.path.abspath(__file__)
        folder_path = os.path.dirname(file_path)
        csv_file = os.path.join(folder_path,"list_file.csv")
        if os.path.exists(csv_file):
          os.remove(csv_file)
        
  def onReset(self):
    self.ui.outputLineEdit.setText("")
    self.ui.surfaceLineEdit.setText("")
    self.ui.rotationSpinBox.value = 45
    self.ui.applyChangesButton.setEnabled(True)
    self.ui.progressLabel.setHidden(True)
    self.ui.openOutSurfButton.setHidden(True)
    self.ui.openOutFolderButton.setHidden(True)
    self.ui.progressBar.setValue(0)
    self.ui.doneLabel.setHidden(True)
    self.ui.surfaceComboBox.setCurrentIndex(0)
    self.ui.labelComboBox.setCurrentIndex(0)
    self.ui.sepOutputsCheckbox.setChecked(False)
    self.removeObservers()    

  def onCancel(self):
    self.logic.cliNode.Cancel()
    self.ui.applyChangesButton.setEnabled(True)
    self.ui.resetButton.setEnabled(True)
    self.ui.progressBar.setEnabled(False)
    self.ui.progressBar.setRange(0,100)
    self.ui.progressLabel.setHidden(True)
    self.ui.cancelButton.setEnabled(False)
    self.removeObservers()    
    print("Process successfully cancelled.")


#
# CrownSegmentationLogic
#

class CrownSegmentationLogic(ScriptedLoadableModuleLogic):
  """
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
      parser = argparse.ArgumentParser()
    parser.add_argument('input_vtk',type=str)
    parser.add_argument('input_stl',type=str)
    parser.add_argument('input_csv',type = int)
    parser.add_argument('out',type=int)
    parser.add_argument('overwrite',type=str)
    parser.add_argument('model',type=str)
    parser.add_argument('crown_segmentation',type=int)
    parser.add_argument('array_name',type=int)
    parser.add_argument('fdi',type=str)
    parser.add_argument('suffix',type=str)
    parser.add_argument('vtk_folder',type=str)

  """

  def __init__(self, surf = "None", input_csv = "None",out="None",overwrite="None",model="None",crown_segmentation="None",array_name="None",fdi="None",suffix="False",vtk_folder="None",dentalmodelseg_path="None"):
    """
    Called when the logic class is instantiated. Can be used for initializing member variables.
    """
    ScriptedLoadableModuleLogic.__init__(self)
    self.surf = surf
    self.input_csv = input_csv
    self.out = out
    self.overwrite = overwrite
    self.model = model
    self.crown_segmentation = crown_segmentation
    self.array_name = array_name
    self.fdi = fdi
    self.suffix = suffix
    self.vtk_folder = vtk_folder
    self.dentalmodelseg_path = dentalmodelseg_path
  




  def process(self):
    parameters = {}
    parameters ["surf"] =self.surf
    parameters ["input_csv"] =self.input_csv
    parameters ["out"] = self.out
    parameters ['overwrite'] = str(self.overwrite)
    parameters ['model'] = self.model
    parameters ['crown_segmentation'] = str(self.crown_segmentation)
    parameters ['array_name'] = str(self.array_name)
    parameters ['fdi'] = str(self.fdi)
    # parameters ['overwrite'] = str(self.overwrite)
    # parameters ['name_env'] = "shapeAxi"
    parameters ['suffix'] = self.suffix
    parameters ['vtk_folder'] = self.vtk_folder
    parameters ['dentalmodelseg_path'] = self.dentalmodelseg_path
    print ('parameters : ', parameters)
    flybyProcess = slicer.modules.crownsegmentationcli
    self.cliNode = slicer.cli.run(flybyProcess,None, parameters)    
    return flybyProcess

class DummyFile(io.IOBase):
        def close(self):
            pass