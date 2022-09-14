
# Automated 3D Dental Model Segmentation and Labeling

DentalModelSeg is an extension for 3D Slicer, a free, open-source software for medical, biomedical and related imaging research.
This extension aims to provide a Graphical User Interface for a deep-learning teeth segmentation tool that we developed at the University of North Carolina in Chapel Hill in collaboration with the University of Michigan in Ann Arbor. This tool takes 3D Intar Oral Surface (IOS) scans of teeth and automatically labels them. The dental crowns can be labeled according to the [Universal Number System](https://en.wikipedia.org/wiki/Universal_Numbering_System), or the [FDI World Dental Federation notation](https://en.wikipedia.org/wiki/FDI_World_Dental_Federation_notation).

![Screenshot of the Module](examples/screenshot_module.png?raw=true)


# Requirements

 - This extension only works with Linux for now. We are working on making it compatible with Windows.
 - The extension requires your GPU to support CUDA.

# How to use the extension
 
## Installation

You can download the extension on the Slicer Extension Manager. Slicer needs to restart after installation.


## Running the module


 - You will find the module under the name "CrownSegmentation - Fiboseg" in the "Segmentation" tab.
 - The input file must be a .vtk file or a MRMLModelNode  of a IOS scan for a lower or upper jaw, or a folder containing .vtk files of jaws. The model
   works better with models of jaws with no wisdom teeth. You can find
   examples in the "Examples" folder.
 - Number of views: this sets the number of 2D views used for one
   prediction. A low number takes less time to compute, but results can
   be inaccurate.
 - Model for segmentation: this is the path for the neural network
   model. 
 - Resolution: this sets the resolution of the 2D views used for the prediction.
   This should usually be set to 320px.
 - Name of predicted labels: The name of the VTK array that stores the labels for each vertex in the output surface file.
 - "Install/Check dependencies" button: This forces the installation of all dependencies.
   If you don't use this button the first time you run a prediction, it will automatically install all dependencies before starting the prediction.
 - "Create one output file for each label": Check this box if you want one separate output file for each tooth. 
 - "Numbering system": lets you choose between [Universal Number System](https://en.wikipedia.org/wiki/Universal_Numbering_System) and [FDI notation](https://en.wikipedia.org/wiki/FDI_World_Dental_Federation_notation).

When prediction is over, you can open the output surface as a MRML node in Slicer by pushing the "Open output surface" button. To visualize the results, open the output file and set scalars to "visible" and select the correct scalar in Slicer's "Models" module.

<!-- ![Example of a jaw model](examples/segmentation_example.png?raw=true) -->

![Example of a jaw model](examples/segmentation_example.png?raw=true)
