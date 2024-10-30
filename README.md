# dev setting

+ virtualenv installation
 
        > python3.9 -m pip install virtualenv
        > mkdir python39_tutorial
        > python3.9 -m virtualenv python39_tutorial

+ package installation

         > cd python39_tutorial
         (~/python39_tutorial)> source bin/activate
         (~/python39_tutorial)> cd ~/segmentation_toturial
         (~/segmentation_toturial)> pip install -r requirements.txt


+ numpy error


++ error message 

        A module that was compiled using NumPy 1.x cannot be run in
        NumPy 2.0.2 as it may crash. To support both 1.x and 2.x
        versions of NumPy, modules must be compiled with NumPy 2.0.
        Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.


++ solution

        > pip uninstall numpy
        > pip install "numpy<2.0"