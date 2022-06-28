#-----------------------------------------------------------------------------
set(MODULE_NAME CrownSegmentationcli)

SlicerMacroBuildScriptedCLI(
  NAME ${MODULE_NAME}
  )

# copy over helper files
# warning: expecting relative filepath
set(HELPER_FILES
  _CrownSegmentationcli/FiboSeg/predict_v3.py
  _CrownSegmentationcli/LinearSubdivisionFilter.py
  _CrownSegmentationcli/post_process.py
  _CrownSegmentationcli/readers.py
  _CrownSegmentationcli/utils.py
)

set(CLI_DEST "${CMAKE_BINARY_DIR}/${Slicer_CLIMODULES_BIN_DIR}")
set(CLI_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}")

foreach(FILE ${HELPER_FILES})
  message("File is ${FILE}")
  get_filename_component(FILE_RELATIVE_PATH ${FILE} DIRECTORY)
  get_filename_component(FILE_BASE_NAME ${FILE} NAME)
  message("relpath ${FILE_RELATIVE_PATH}")
  message("CLI_DEST ${CLI_DEST}")
  message("${CMAKE_COMMAND} -E make_directory ${CLI_DEST}/${FILE_RELATIVE_PATH}")
  message("${CMAKE_COMMAND} -E copy ${CLI_SOURCE}/${FILE} ${CLI_DEST}/${FILE_RELATIVE_PATH}")
  add_custom_command(
    OUTPUT "${CLI_DEST}/${FILE_RELATIVE_PATH}/${FILE_BASE_NAME}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CLI_DEST}/${FILE_RELATIVE_PATH}"
    COMMAND "${CMAKE_COMMAND}" -E copy "${CLI_SOURCE}/${FILE}" "${CLI_DEST}/${FILE_RELATIVE_PATH}"
    DEPENDS "${CLI_SOURCE}/${FILE}"
    )

  string(REPLACE "/" "_" FILETARGET ${FILE})
  string(REPLACE "." "_" FILETARGET ${FILETARGET})
  add_custom_target(${MODULE_NAME}_${FILETARGET} ALL
    SOURCES "${CLI_SOURCE}/${FILE}"
    DEPENDS "${CLI_DEST}/${FILE_RELATIVE_PATH}/${FILE_BASE_NAME}"
    COMMENT "Copying ${FILE} files to ${CLI_DEST}/${FILE_RELATIVE_PATH}"
    )

  install(FILES ${FILE}
    DESTINATION "${Slicer_INSTALL_CLIMODULES_BIN_DIR}"
    )
endforeach()
