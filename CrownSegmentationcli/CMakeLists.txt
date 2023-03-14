#-----------------------------------------------------------------------------
set(MODULE_NAME CrownSegmentationcli)

SlicerMacroBuildScriptedCLI(
  NAME ${MODULE_NAME}
)

set(FOLDER_LIBRARY __CrownSegmentation)

set(MODULE_PYTHON_SCRIPTS
  ${FOLDER_LIBRARY}/__init__.py 
  ${FOLDER_LIBRARY}/dataset.py 
  ${FOLDER_LIBRARY}/net.py 
  ${FOLDER_LIBRARY}/post_process.py 
  ${FOLDER_LIBRARY}/readers.py 
  ${FOLDER_LIBRARY}/LinearSubdivisionFilter.py 
  ${FOLDER_LIBRARY}/utils.py
  )

  slicerMacroBuildScriptedModule(
    NAME  ${FOLDER_LIBRARY}
    SCRIPTS ${MODULE_PYTHON_SCRIPTS}
)