# ParaViz3D
3D visualization of MPI traces

### INSTRUCTIONS:

## Preliminary steps:
# You should have Python installed
# You should have the NumPy, Math, and Mathutils modules
# You should have the module 'otf2', otherwise run:
pip install otf2
# You should have Blender installed, otherwise please visit:
https://www.blender.org/download/


## I. Trace converter script
# 1. (Optional) Run the following command from ParaViz3D's root folder:
# This step is optional because it was already performed
For the Cube case:
python trace_converter.py --root_folder="./traces/cube/cube864_2184362/"
For the Grid case:
python trace_converter.py --root_folder="./traces/grid/grid576_2482863/"
For the Sphere case:
python trace_converter.py --root_folder="./traces/sphere/sphere720_2205999/"


### II. Blender plotter script, to generate the trace's 3D scene

# 1. Please update the following absolute path to the ParaViz3D folder:
path_to_ParaViz3D = 'D:/ParaViz/'
# 2. Then open Blender, go to the the Scripting tab, and open this file there.
# 3. Press the Play button (next to the file name)
# 4. Select the Layout tab to see the result
# You can use Blender's controls to change the zoom and view angle
# 5. (Optional) Select the desired color and view options
# (use the downward arrow on the top-right corner to modify the shading options)
# 6. (Optional) To render a video : Select View > Viewport render animation