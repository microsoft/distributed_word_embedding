# Windows Installation

1. Get and build the DMTK Framework [multiverso](https://github.com/Microsoft/multiverso.git).

2. Open distributed_word_embedding/distributed_word_embedding.sln, change configuration and platform to Release and x64, set the ```include``` and ```lib``` path of multiverso in project property. 

3.Enable openmp 2.0 support.

#To set this compiler option in the Visual Studio development environment
#1)Open the project's Property Pages dialog box. For details, see How to: Open Project Property Pages.
#2)Expand the Configuration Properties node.
#3)Expand the C/C++ node.
#4)Select the Language property page.
#5)Modify the OpenMP Support property.

4. Build the solution.
