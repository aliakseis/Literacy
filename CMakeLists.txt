cmake_minimum_required(VERSION 3.8.2)

project(Literacy)


# Find includes in the build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Turn on automatic invocation of the MOC & UIC
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

# There may be a way to tell up front if Qt5 is going to be found, but I haven't found
# a foolproof way to do it yet, so settle for the default error message for now.
#if(NOT CMAKE_PREFIX_PATH AND NOT Qt5Widgets_DIR)
#    message(WARNING "CMAKE_PREFIX_PATH is not defined, so find_package may not work. Set the CMAKE_PREFIX_PATH "
#            "environment variable to the install prefix of Qt 5, either on the command line as "
#            "-DCMAKE_PREFIX_PATH=\"path/to/Qt5/lib/cmake\" or with set(CMAKE_PREFIX_PATH path/to/Qt5/lib/cmake)")
#endif(NOT CMAKE_PREFIX_PATH AND NOT Qt5Widgets_DIR)

# Add a compiler flag
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_STANDARD 17)

# Make this a GUI application on Windows
if(WIN32)
  set(CMAKE_WIN32_EXECUTABLE ON)
endif()

# Find the QtWidgets library
find_package(Qt5 REQUIRED COMPONENTS Widgets)

find_package(Tesseract CONFIG REQUIRED)
find_package(OpenCV REQUIRED) 

#target_include_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS})  

# Generate rules for building source files from the resources
#qt5_add_resources(QRCS tiles.qrc)

# Tell CMake to create the Tiles executable
add_executable(Literacy main.cpp mainwindow.cpp screencapturer.cpp ${QRCS})

# Add the Qt5 Widgets for linking
target_link_libraries(Literacy Qt5::Widgets libtesseract ${OpenCV_LIBS})