
﻿# README: Automated jaw segmentation extension

This extension aims to provide a GUI for the deep learning method for surface segmentation that we developed. The dental crowns are segmented according to the [Universal Number System](https://en.wikipedia.org/wiki/Universal_Numbering_System).



 *NB: The extension was tested with these specific versions, however it may work with more recent ones.
Additional packages may be required. you can install them with "pip install ... ".  No specific version should be required for these packages.*

# How to run the extension
 

## Running the module

 - The input file must be a .vtk file of a lower or upper jaw. The model
   works better with models of jaws with no wisdom teeth. You can find
   examples in the "Examples" folder.
 - Number of views: this sets the number of 2D views used for one
   prediction. A low number takes less time to compute, but results can
   be inaccurate.
 - Model for segmentation: this is the path for the neural network
   model. Resolution: This sets the resolution of the 2D views. 320 px
   is recommended. Name of predicted labels: this is the name the array
   with the predicted labels on  the output vtk file.
 - To visualize the results, open the output file and set scalars to
   "visible" and select the correct scalar in  Slicer's "Models" module.

When prediction is over, you can open the output surface as a MRML node in Slicer by pushing the "Open output surface" button.

<!-- ![Example of a jaw model](examples/segmentation_example.png?raw=true) -->

![Example of a jaw model](examples/segmentation_example.png?raw=true)
