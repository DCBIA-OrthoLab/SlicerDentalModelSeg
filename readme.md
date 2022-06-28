
# README: Automated jaw segmentation extension

This extension aims to provide a GUI for the deep learning method for surface segmentation that we developed. The dental crowns are segmented according to the [Universal Number System](https://en.wikipedia.org/wiki/Universal_Numbering_System).




# How to run the extension
 

## Running the module

 - The input file must be a .vtk file or a MRMLModelNode  of a IOS scan for a lower or upper jaw, or a folder containing .vtk files of jaws. The model
   works better with models of jaws with no wisdom teeth. You can find
   examples in the "Examples" folder.
 - Number of views: this sets the number of 2D views used for one
   prediction. A low number takes less time to compute, but results can
   be inaccurate.
 - Model for segmentation: this is the path for the neural network
   model. Resolution: This sets the resolution of the 2D views. 320 px
   is recommended. Name of predicted labels: this is the name the array
   with the predicted labels on  the output vtk file.
 - Resolution: this sets the resolution of the 2D views used for the prediction.
   This should usually be set to 320px.
 - Name of predicted labels: The name of the VTK array that stores the labels for each vertex in the output surface file.
 - "Install/Check dependencies" button: This forces the installation of all dependencies.
   If you don't use this button the first time you run a prediction, it will automatically install all dependencies before starting the prediction.
 - "Create one output file for each label": Check this box if you want one separate output file for each tooth. 

When prediction is over, you can open the output surface as a MRML node in Slicer by pushing the "Open output surface" button. To visualize the results, open the output file and set scalars to "visible" and select the correct scalar in Slicer's "Models" module.

<!-- ![Example of a jaw model](examples/segmentation_example.png?raw=true) -->

![Example of a jaw model](examples/segmentation_example.png?raw=true)
