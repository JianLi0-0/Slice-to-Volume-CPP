# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CMake.app/Contents/bin/cmake

# The command to remove a file.
RM = /Applications/CMake.app/Contents/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/lijian/Local/ITK/program

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/lijian/Local/ITK/program/build

# Include any dependencies generated for this target.
include CMakeFiles/Test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/Test.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/Test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Test.dir/flags.make

CMakeFiles/Test.dir/test.cpp.o: CMakeFiles/Test.dir/flags.make
CMakeFiles/Test.dir/test.cpp.o: ../test.cpp
CMakeFiles/Test.dir/test.cpp.o: CMakeFiles/Test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/lijian/Local/ITK/program/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Test.dir/test.cpp.o"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Test.dir/test.cpp.o -MF CMakeFiles/Test.dir/test.cpp.o.d -o CMakeFiles/Test.dir/test.cpp.o -c /Users/lijian/Local/ITK/program/test.cpp

CMakeFiles/Test.dir/test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Test.dir/test.cpp.i"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/lijian/Local/ITK/program/test.cpp > CMakeFiles/Test.dir/test.cpp.i

CMakeFiles/Test.dir/test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Test.dir/test.cpp.s"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/lijian/Local/ITK/program/test.cpp -o CMakeFiles/Test.dir/test.cpp.s

# Object files for target Test
Test_OBJECTS = \
"CMakeFiles/Test.dir/test.cpp.o"

# External object files for target Test
Test_EXTERNAL_OBJECTS =

Test: CMakeFiles/Test.dir/test.cpp.o
Test: CMakeFiles/Test.dir/build.make
Test: /usr/local/lib/libITKLabelMap-5.2.1.dylib
Test: /usr/local/lib/libITKFastMarching-5.2.1.dylib
Test: /usr/local/lib/libITKPolynomials-5.2.1.dylib
Test: /usr/local/lib/libITKBiasCorrection-5.2.1.dylib
Test: /usr/local/lib/libITKColormap-5.2.1.dylib
Test: /usr/local/lib/libITKConvolution-5.2.1.dylib
Test: /usr/local/lib/libITKDICOMParser-5.2.1.dylib
Test: /usr/local/lib/libITKDeformableMesh-5.2.1.dylib
Test: /usr/local/lib/libITKDenoising-5.2.1.dylib
Test: /usr/local/lib/libITKDiffusionTensorImage-5.2.1.dylib
Test: /usr/local/lib/libITKPDEDeformableRegistration-5.2.1.dylib
Test: /usr/local/lib/libITKIOBioRad-5.2.1.dylib
Test: /usr/local/lib/libITKIOBruker-5.2.1.dylib
Test: /usr/local/lib/libITKIOCSV-5.2.1.dylib
Test: /usr/local/lib/libITKIOGE-5.2.1.dylib
Test: /usr/local/lib/libITKIOHDF5-5.2.1.dylib
Test: /usr/local/lib/libITKIOJPEG2000-5.2.1.dylib
Test: /usr/local/lib/libITKIOLSM-5.2.1.dylib
Test: /usr/local/lib/libITKIOMINC-5.2.1.dylib
Test: /usr/local/lib/libITKIOMRC-5.2.1.dylib
Test: /usr/local/lib/libITKIOSiemens-5.2.1.dylib
Test: /usr/local/lib/libITKIOSpatialObjects-5.2.1.dylib
Test: /usr/local/lib/libITKIOStimulate-5.2.1.dylib
Test: /usr/local/lib/libITKIOTransformHDF5-5.2.1.dylib
Test: /usr/local/lib/libITKIOTransformInsightLegacy-5.2.1.dylib
Test: /usr/local/lib/libITKIOTransformMatlab-5.2.1.dylib
Test: /usr/local/lib/libITKKLMRegionGrowing-5.2.1.dylib
Test: /usr/local/lib/libITKMarkovRandomFieldsClassifiers-5.2.1.dylib
Test: /usr/local/lib/libITKQuadEdgeMeshFiltering-5.2.1.dylib
Test: /usr/local/lib/libITKRegionGrowing-5.2.1.dylib
Test: /usr/local/lib/libITKRegistrationMethodsv4-5.2.1.dylib
Test: /usr/local/lib/libITKTestKernel-5.2.1.dylib
Test: /usr/local/lib/libITKVideoIO-5.2.1.dylib
Test: /usr/local/lib/libITKVtkGlue-5.2.1.dylib
Test: /usr/local/lib/libITKWatersheds-5.2.1.dylib
Test: /usr/local/lib/libITKFFT-5.2.1.dylib
Test: /usr/local/lib/libitkopenjpeg-5.2.1.dylib
Test: /usr/local/lib/libitkminc2-5.2.1.dylib
Test: /usr/local/lib/libITKIOIPL-5.2.1.dylib
Test: /usr/local/lib/libITKIOXML-5.2.1.dylib
Test: /usr/local/lib/libitkhdf5_cpp-shared-5.2.1.dylib
Test: /usr/local/lib/libitkhdf5-shared-5.2.1.dylib
Test: /usr/local/lib/libITKIOTransformBase-5.2.1.dylib
Test: /usr/local/lib/libITKTransformFactory-5.2.1.dylib
Test: /usr/local/lib/libITKImageFeature-5.2.1.dylib
Test: /usr/local/lib/libITKOptimizersv4-5.2.1.dylib
Test: /usr/local/lib/libITKOptimizers-5.2.1.dylib
Test: /usr/local/lib/libitklbfgs-5.2.1.dylib
Test: /usr/local/lib/libITKIOBMP-5.2.1.dylib
Test: /usr/local/lib/libITKIOGDCM-5.2.1.dylib
Test: /usr/local/lib/libitkgdcmMSFF-5.2.1.dylib
Test: /usr/local/lib/libitkgdcmDICT-5.2.1.dylib
Test: /usr/local/lib/libitkgdcmIOD-5.2.1.dylib
Test: /usr/local/lib/libitkgdcmDSED-5.2.1.dylib
Test: /usr/local/lib/libitkgdcmCommon-5.2.1.dylib
Test: /usr/local/lib/libITKIOGIPL-5.2.1.dylib
Test: /usr/local/lib/libITKIOJPEG-5.2.1.dylib
Test: /usr/local/lib/libITKIOTIFF-5.2.1.dylib
Test: /usr/local/lib/libitktiff-5.2.1.dylib
Test: /usr/local/lib/libitkjpeg-5.2.1.dylib
Test: /usr/local/lib/libITKIOMeshBYU-5.2.1.dylib
Test: /usr/local/lib/libITKIOMeshFreeSurfer-5.2.1.dylib
Test: /usr/local/lib/libITKIOMeshGifti-5.2.1.dylib
Test: /usr/local/lib/libITKgiftiio-5.2.1.dylib
Test: /usr/local/lib/libITKEXPAT-5.2.1.dylib
Test: /usr/local/lib/libITKIOMeshOBJ-5.2.1.dylib
Test: /usr/local/lib/libITKIOMeshOFF-5.2.1.dylib
Test: /usr/local/lib/libITKIOMeshVTK-5.2.1.dylib
Test: /usr/local/lib/libITKIOMeshBase-5.2.1.dylib
Test: /usr/local/lib/libITKQuadEdgeMesh-5.2.1.dylib
Test: /usr/local/lib/libITKIOMeta-5.2.1.dylib
Test: /usr/local/lib/libITKMetaIO-5.2.1.dylib
Test: /usr/local/lib/libITKIONIFTI-5.2.1.dylib
Test: /usr/local/lib/libITKniftiio-5.2.1.dylib
Test: /usr/local/lib/libITKznz-5.2.1.dylib
Test: /usr/local/lib/libITKIONRRD-5.2.1.dylib
Test: /usr/local/lib/libITKNrrdIO-5.2.1.dylib
Test: /usr/local/lib/libITKIOPNG-5.2.1.dylib
Test: /usr/local/lib/libitkpng-5.2.1.dylib
Test: /usr/local/lib/libitkzlib-5.2.1.dylib
Test: /usr/local/lib/libITKIOVTK-5.2.1.dylib
Test: /usr/local/lib/libITKIOImageBase-5.2.1.dylib
Test: /usr/local/lib/libITKVideoCore-5.2.1.dylib
Test: /usr/local/lib/libITKVTK-5.2.1.dylib
Test: /opt/homebrew/lib/libvtkRenderingOpenGL2-9.0.9.0.1.dylib
Test: /opt/homebrew/lib/libvtkRenderingUI-9.0.9.0.1.dylib
Test: /opt/homebrew/lib/libGLEW.dylib
Test: /opt/homebrew/lib/libvtkInteractionWidgets-9.0.9.0.1.dylib
Test: /opt/homebrew/lib/libvtkInteractionStyle-9.0.9.0.1.dylib
Test: /opt/homebrew/lib/libvtkImagingSources-9.0.9.0.1.dylib
Test: /opt/homebrew/lib/libvtkIOImage-9.0.9.0.1.dylib
Test: /opt/homebrew/lib/libvtkRenderingFreeType-9.0.9.0.1.dylib
Test: /opt/homebrew/lib/libvtkfreetype-9.0.9.0.1.dylib
Test: /Library/Developer/CommandLineTools/SDKs/MacOSX11.3.sdk/usr/lib/libz.tbd
Test: /opt/homebrew/lib/libvtkImagingCore-9.0.9.0.1.dylib
Test: /opt/homebrew/lib/libvtkRenderingCore-9.0.9.0.1.dylib
Test: /opt/homebrew/lib/libvtkFiltersSources-9.0.9.0.1.dylib
Test: /opt/homebrew/lib/libvtkFiltersGeneral-9.0.9.0.1.dylib
Test: /opt/homebrew/lib/libvtkFiltersCore-9.0.9.0.1.dylib
Test: /opt/homebrew/lib/libvtkCommonExecutionModel-9.0.9.0.1.dylib
Test: /opt/homebrew/lib/libvtkCommonDataModel-9.0.9.0.1.dylib
Test: /opt/homebrew/lib/libvtkCommonTransforms-9.0.9.0.1.dylib
Test: /opt/homebrew/lib/libvtkCommonMisc-9.0.9.0.1.dylib
Test: /opt/homebrew/lib/libvtkCommonMath-9.0.9.0.1.dylib
Test: /opt/homebrew/lib/libvtkCommonCore-9.0.9.0.1.dylib
Test: /opt/homebrew/lib/libvtksys-9.0.9.0.1.dylib
Test: /usr/local/lib/libITKMathematicalMorphology-5.2.1.dylib
Test: /usr/local/lib/libITKStatistics-5.2.1.dylib
Test: /usr/local/lib/libitkNetlibSlatec-5.2.1.dylib
Test: /usr/local/lib/libITKSpatialObjects-5.2.1.dylib
Test: /usr/local/lib/libITKMesh-5.2.1.dylib
Test: /usr/local/lib/libITKTransform-5.2.1.dylib
Test: /usr/local/lib/libITKPath-5.2.1.dylib
Test: /usr/local/lib/libITKCommon-5.2.1.dylib
Test: /usr/local/lib/libitkdouble-conversion-5.2.1.dylib
Test: /usr/local/lib/libitksys-5.2.1.dylib
Test: /usr/local/lib/libITKVNLInstantiation-5.2.1.dylib
Test: /usr/local/lib/libitkvnl_algo-5.2.1.dylib
Test: /usr/local/lib/libitkvnl-5.2.1.dylib
Test: /usr/local/lib/libitkv3p_netlib-5.2.1.dylib
Test: /usr/local/lib/libitkvcl-5.2.1.dylib
Test: /usr/local/lib/libITKSmoothing-5.2.1.dylib
Test: CMakeFiles/Test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/lijian/Local/ITK/program/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Test.dir/build: Test
.PHONY : CMakeFiles/Test.dir/build

CMakeFiles/Test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Test.dir/clean

CMakeFiles/Test.dir/depend:
	cd /Users/lijian/Local/ITK/program/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/lijian/Local/ITK/program /Users/lijian/Local/ITK/program /Users/lijian/Local/ITK/program/build /Users/lijian/Local/ITK/program/build /Users/lijian/Local/ITK/program/build/CMakeFiles/Test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Test.dir/depend

