################################################################################
### README: Blender plotter script, to generate the trace's 3D scene

### INSTRUCTIONS:
# 1. Please update the following absolute path to the ParaViz3D folder:
path_to_ParaViz3D = 'D:/ParaViz/'
# 2. Then open Blender, go to the the Scripting tab, and open this file there.
# 3. Press the Play button (next to the file name)
# 4. Select the Layout tab to see the result
# You can use Blender's controls to change the zoom and view angle
# 5. (Optional) Select the desired color and view options
# (use the downward arrow on the top-right corner to modify the shading options)
# 6. (Optional) To render a video : Select View > Viewport render animation

################################################################################



import bpy
import numpy as np




################################################################################
#                                   CONCEPT                                    #
################################################################################

# If the desired topology of the numerical problem is not standard, the user 
# must give the desired coordinates corresponding to each ranks, and create an
# array of those coordinates ordered from the first rank 0 to the last to match 
# their correspondance.

# We then use the timestamp file, which also contains the MPI times for each 
# rank from the first to the last rank, to color each rank along the frames.




################################################################################
#                    SOLID CREATION/DESTRUCTION FUNCTIONS                      #
################################################################################

def create_solid(coordinates, type="cube", base_side_nb = 4, radius = 0.4, height = 0.4):

    angle = 2 * np.pi / base_side_nb

    
    if type == "cube":
        for [x,y,z], solid_counter in zip(coordinates, range(len(coordinates))):


            verts = []
            edges = []
            faces = []
        
            verts.append((x-radius/2, y-radius/2, z-radius/2))
            verts.append((x-radius/2, y+radius/2, z-radius/2))
            verts.append((x+radius/2, y+radius/2, z-radius/2))
            verts.append((x+radius/2, y-radius/2, z-radius/2))
            verts.append((x-radius/2, y-radius/2, z+radius/2))
            verts.append((x-radius/2, y+radius/2, z+radius/2))
            verts.append((x+radius/2, y+radius/2, z+radius/2))
            verts.append((x+radius/2, y-radius/2, z+radius/2))

            faces.append((0, 1, 2, 3))
            faces.append((7, 6, 5, 4))
            faces.append((4, 5, 1, 0))
            faces.append((7, 4, 0, 3))
            faces.append((6, 7, 3, 2))
            faces.append((5, 6, 2, 1))
            
            mesh_data = bpy.data.meshes.new(type+"_data%d" % solid_counter)
            mesh_data.from_pydata(verts,edges,faces)
            mesh_obj = bpy.data.objects.new(type+"_object%d" % solid_counter,mesh_data)
            bpy.context.collection.objects.link(mesh_obj)
            
    
    if type == "diamond":
        for [x,y,z], solid_counter in zip(coordinates, range(len(coordinates))):


            verts = []
            edges = []
            faces = []
        
            verts.append((x,   y,  z + height))
            for i in range(base_side_nb):
                verts.append((x + radius * np.cos(i*angle),   y + radius * np.sin(i*angle),  z))
            verts.append((x,   y,  z - height))

            for i in range(base_side_nb):
                faces.append((             0, i+1, 1+(i+1)%base_side_nb ))
                faces.append((base_side_nb+1, i+1, 1+(i+1)%base_side_nb ))
            
            mesh_data = bpy.data.meshes.new("cube_data%d" % solid_counter)
            mesh_data.from_pydata(verts,edges,faces)
            mesh_obj = bpy.data.objects.new("cube_object%d" % solid_counter,mesh_data)
            bpy.context.collection.objects.link(mesh_obj)

    return


def create_event(x_start, x_length, y_start, y_length, z, color):
    verts = []
    edges = []
    faces = []

    verts.append((x_start,            y_start,            z))
    verts.append((x_start + x_length, y_start,            z))
    verts.append((x_start + x_length, y_start + y_length, z))
    verts.append((x_start,            y_start + y_length, z))

    faces.append((0, 1, 2, 3))
    
    mesh_data = bpy.data.meshes.new("event_data")
    mesh_data.from_pydata(verts,edges,faces)
    mesh_obj = bpy.data.objects.new("event_object",mesh_data)
    bpy.context.collection.objects.link(mesh_obj)
    
    mat = bpy.data.materials.new(name="colored_event")
    mesh_obj.active_material = mat
    mat.diffuse_color = color


def deleteTypeObjects(type = 'VOLUME'):

    for o in bpy.context.scene.objects:
        if o.type == type:
            o.select_set(False)
        else:
            o.select_set(True)

    bpy.ops.object.delete()

    return


def deleteAllObjects():
    """
    Deletes all objects in the current scene
    """
    deleteListObjects = ['MESH', 'CURVE', 'SURFACE', 'META', 'FONT', 'HAIR', 'POINTCLOUD', 'VOLUME', 'GPENCIL',
                     'ARMATURE', 'LATTICE', 'EMPTY', 'LIGHT', 'LIGHT_PROBE', 'CAMERA', 'SPEAKER','TEXT']

    # Select all objects in the scene to be deleted:

    for o in bpy.context.scene.objects:
        for i in deleteListObjects:
            if o.type == i:
                o.select_set(False)
            else:
                o.select_set(True)
    # Deletes all selected objects in the scene:

    bpy.ops.object.delete()




################################################################################
#                       TOPOLOGIES COORDINATES FUNCTIONS                       #
################################################################################



def generate_circle(radius=10, nb_circle_points=24, z = 0):
    coordinates = []
    for i in range(nb_circle_points):
        x = radius * np.cos(2 * np.pi * i / nb_circle_points)
        y = radius * np.sin(2 * np.pi * i / nb_circle_points)
        coordinates.append((x, y, z))
    
    return coordinates


def generate_torus(R=0.55, r=0.5, n_major=24, n_minor=24):
    """
    Generate vertices and faces for a torus.
    
    Parameters:
    R (float): major radius
    r (float): minor radius
    n_major (int): number of segments around the major circle
    n_minor (int): number of segments around the minor circle

    Returns:
    vertices (list of lists): [x, y, z] coordinates
    faces (list of lists): [i1, i2, i3, i4] indices of quad face corners
    """
    vertices = []
    faces = []

    for i in range(n_major):
        theta = 2.0 * np.pi * i / n_major
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        for j in range(n_minor):
            phi = 2.0 * np.pi * j / n_minor
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)

            x = (R + r * cos_phi) * cos_theta
            y = (R + r * cos_phi) * sin_theta
            z = r * sin_phi
            vertices.append([x, y, z])

    for i in range(n_major):
        for j in range(n_minor):
            next_i = (i + 1) % n_major
            next_j = (j + 1) % n_minor

            a = i * n_minor + j
            b = next_i * n_minor + j
            c = next_i * n_minor + next_j
            d = i * n_minor + next_j

            faces.append([a, b, c, d])

    return vertices, faces


#cuboid - cube with irregular sides
def cuboid(x_nb_points=9, y_nb_points=8, z_nb_points=1, spacing_x=1, spacing_y=1, spacing_z=1):
    coordinates = []
    for k in range(z_nb_points):
        for j in range(y_nb_points):
            for i in range(x_nb_points):
                x = spacing_x * (i - 0.5 * (x_nb_points-1))
                y = spacing_y * (j - 0.5 * (y_nb_points-1))
                z = spacing_z * (k - 0.5 * (z_nb_points-1))
                coordinates.append((x, y, z))

    return coordinates


#cuboid - cube with irregular sides #use floats instead of ints for coordinates for true centering?
def cuboid_numa(x_nb_points=24, y_nb_points=24, z_nb_points=1, ccnum_size_x=6, ccnum_size_y=3, ccnum_size_z=1, proc_size_x=6, proc_size_y=6, proc_size_z=1, node_size_x=12, node_size_y=6, node_size_z=1, spacing=0.42):
    coordinates = []
    offset_x = -0.6 * spacing
    offset_y = -0.6 * spacing
    offset_z = -0.6 * spacing
    offset_x_tot = 0#((x_nb_points // node_size_x) - 1) * 2 + ((x_nb_points // node_size_x) - 1) // proc_size_x
    offset_x_tot += 0#offset_x_tot * ((node_size_x // proc_size) - 1) + (proc_size // ccnum_size) - 1
    offset_y_tot = 0
    offset_z_tot = 0
    for k in range(z_nb_points):
        if k % node_size_z == 0:
            offset_z += 0.6 * spacing
        elif k % proc_size_z == 0:
            offset_z += 0.4 * spacing
        elif k % ccnum_size_z == 0:
            offset_z += 0.2 * spacing
        for j in range(y_nb_points):
            if j % node_size_y == 0:
                offset_y += 2 * spacing
            elif j % proc_size_y == 0:
                offset_y += 1 * spacing
            elif j % ccnum_size_y == 0:
                offset_y += 0.5 * spacing
            for i in range(x_nb_points):
                if i % node_size_x == 0:
                    offset_x += 2 * spacing
                elif i % proc_size_x == 0:
                    offset_x += 1 * spacing
                elif i % ccnum_size_x == 0:
                    offset_x += 0.5 * spacing
                x = spacing * (i - 0.5 * (x_nb_points-1) + offset_x)
                y = spacing * (j - 0.5 * (y_nb_points-1) + offset_y)
                z = spacing * (k - 0.5 * (z_nb_points-1) + offset_z) - 0.3
                coordinates.append((x, y, z))
            offset_x = -0.6 * spacing
        offset_y = -0.6 * spacing
    
    return coordinates


#from icosphere import icosphere

# Icosphere helper functions in case the import fails

def icosphere(nu = 1, nr_verts = None):
    '''
    Returns a geodesic icosahedron with subdivision frequency nu. Frequency
    nu = 1 returns regular unit icosahedron, and nu>1 preformes subdivision.
    If nr_verts is given, nu will be adjusted such that icosphere contains
    at least nr_verts vertices. Returned faces are zero-indexed!
        
    Parameters
    ----------
    nu : subdivision frequency, integer (larger than 1 to make a change).
    nr_verts: desired number of mesh vertices, if given, nu may be increased.
        
    
    Returns
    -------
    subvertices : vertex list, numpy array of shape (20+10*(nu+1)*(nu-1)/2, 3)
    subfaces : face list, numpy array of shape (10*n**2, 3)
    
    '''
  
    # Unit icosahedron
    (vertices,faces) = icosahedron()

    # If nr_verts given, computing appropriate subdivision frequency nu.
    # We know nr_verts = 12+10*(nu+1)(nu-1)
    if not nr_verts is None:
        nu_min = np.ceil(np.sqrt(max(1+(nr_verts-12)/10, 1)))
        nu = max(nu, nu_min)
        
    # Subdividing  
    if nu>1:
        (vertices,faces) = subdivide_mesh(vertices, faces, nu)
        vertices = vertices/np.sqrt(np.sum(vertices**2, axis=1, keepdims=True))

    return(vertices, faces)

def icosahedron():
    '''' Regular unit icosahedron. '''
    
    # 12 principal directions in 3D space: points on an unit icosahedron
    phi = (1+np.sqrt(5))/2    
    vertices = np.array([
        [0, 1, phi], [0,-1, phi], [1, phi, 0],
        [-1, phi, 0], [phi, 0, 1], [-phi, 0, 1]])/np.sqrt(1+phi**2)
    vertices = np.r_[vertices,-vertices]
    
    # 20 faces
    faces = np.array([
        [0,5,1], [0,3,5], [0,2,3], [0,4,2], [0,1,4], 
        [1,5,8], [5,3,10], [3,2,7], [2,4,11], [4,1,9], 
        [7,11,6], [11,9,6], [9,8,6], [8,10,6], [10,7,6], 
        [2,11,7], [4,9,11], [1,8,9], [5,10,8], [3,7,10]], dtype=int)    
    
    return (vertices, faces)


def subdivide_mesh(vertices, faces, nu):
    '''
    Subdivides mesh by adding vertices on mesh edges and faces. Each edge 
    will be divided in nu segments. (For example, for nu=2 one vertex is added  
    on each mesh edge, for nu=3 two vertices are added on each mesh edge and 
    one vertex is added on each face.) If V and F are number of mesh vertices
    and number of mesh faces for the input mesh, the subdivided mesh contains 
    V + F*(nu+1)*(nu-1)/2 vertices and F*nu^2 faces.
    
    Parameters
    ----------
    vertices : vertex list, numpy array of shape (V,3) 
    faces : face list, numby array of shape (F,3). Zero indexed.
    nu : subdivision frequency, integer (larger than 1 to make a change).
    
    Returns
    -------
    subvertices : vertex list, numpy array of shape (V + F*(nu+1)*(nu-1)/2, 3)
    subfaces : face list, numpy array of shape (F*n**2, 3)
    
    Author: vand at dtu.dk, 8.12.2017. Translated to python 6.4.2021
    
    '''
        
    edges = np.r_[faces[:,:-1], faces[:,1:],faces[:,[0,2]]]
    edges = np.unique(np.sort(edges, axis=1),axis=0)
    F = faces.shape[0]
    V = vertices.shape[0]
    E = edges.shape[0] 
    subfaces = np.empty((F*nu**2, 3), dtype = int)
    subvertices = np.empty((V+E*(nu-1)+F*(nu-1)*(nu-2)//2, 3))
                        
    subvertices[:V] = vertices
    
    # Dictionary for accessing edge index from indices of edge vertices.
    edge_indices = dict()
    for i in range(V):
        edge_indices[i] = dict()
    for i in range(E):
        edge_indices[edges[i,0]][edges[i,1]] = i
        edge_indices[edges[i,1]][edges[i,0]] = -i
         
    template = faces_template(nu)
    ordering = vertex_ordering(nu)
    reordered_template = ordering[template]
    
    # At this point, we have V vertices, and now we add (nu-1) vertex per edge
    # (on-edge vertices).
    w = np.arange(1,nu)/nu # interpolation weights
    for e in range(E):
        edge = edges[e]
        for k in range(nu-1):
            subvertices[V+e*(nu-1)+k] = (w[-1-k] * vertices[edge[0]] 
                                         + w[k] * vertices[edge[1]])
  
    # At this point we have E(nu-1)+V vertices, and we add (nu-1)*(nu-2)/2 
    # vertices per face (on-face vertices).
    r = np.arange(nu-1)
    for f in range(F):
        # First, fixing connectivity. We get hold of the indices of all
        # vertices invoved in this subface: original, on-edges and on-faces.
        T = np.arange(f*(nu-1)*(nu-2)//2+E*(nu-1)+V, 
                      (f+1)*(nu-1)*(nu-2)//2+E*(nu-1)+V) # will be added
        eAB = edge_indices[faces[f,0]][faces[f,1]] 
        eAC = edge_indices[faces[f,0]][faces[f,2]] 
        eBC = edge_indices[faces[f,1]][faces[f,2]] 
        AB = reverse(abs(eAB)*(nu-1)+V+r, eAB<0) # already added
        AC = reverse(abs(eAC)*(nu-1)+V+r, eAC<0) # already added
        BC = reverse(abs(eBC)*(nu-1)+V+r, eBC<0) # already added
        VEF = np.r_[faces[f], AB, AC, BC, T]
        subfaces[f*nu**2:(f+1)*nu**2, :] = VEF[reordered_template]
        # Now geometry, computing positions of face vertices.
        subvertices[T,:] = inside_points(subvertices[AB,:],subvertices[AC,:])
    
    return (subvertices, subfaces)

def reverse(vector, flag): 
    '''' For reversing the direction of an edge. ''' 
    
    if flag:
        vector = vector[::-1]
    return(vector)


def faces_template(nu):
    '''
    Template for linking subfaces                  0
    in a subdivision of a face.                   / \
    Returns faces with vertex                    1---2
    indexing given by reading order             / \ / \
    (as illustratated).                        3---4---5
                                              / \ / \ / \
                                             6---7---8---9    
                                            / \ / \ / \ / \ 
                                           10--11--12--13--14 
    '''
  
    faces = []
    # looping in layers of triangles
    for i in range(nu): 
        vertex0 = i*(i+1)//2
        skip = i+1      
        for j in range(i): # adding pairs of triangles, will not run for i==0
            faces.append([j+vertex0, j+vertex0+skip, j+vertex0+skip+1])
            faces.append([j+vertex0, j+vertex0+skip+1, j+vertex0+1])
        # adding the last (unpaired, rightmost) triangle
        faces.append([i+vertex0, i+vertex0+skip, i+vertex0+skip+1])
        
    return (np.array(faces))


def vertex_ordering(nu):
    ''' 
    Permutation for ordering of                    0
    face vertices which transformes               / \
    reading-order indexing into indexing         3---6
    first corners vertices, then on-edges       / \ / \
    vertices, and then on-face vertices        4---12--7
    (as illustrated).                         / \ / \ / \
                                             5---13--14--8
                                            / \ / \ / \ / \ 
                                           1---9--10--11---2 
    '''
    
    left = [j for j in range(3, nu+2)]
    right = [j for j in range(nu+2, 2*nu+1)]
    bottom = [j for j in range(2*nu+1, 3*nu)]
    inside = [j for j in range(3*nu,(nu+1)*(nu+2)//2)]
    
    o = [0] # topmost corner
    for i in range(nu-1):
        o.append(left[i])
        o = o + inside[i*(i-1)//2:i*(i+1)//2]
        o.append(right[i])
    o = o + [1] + bottom + [2]
        
    return(np.array(o))


def inside_points(vAB,vAC):
    '''  
    Returns coordinates of the inside                 .
    (on-face) vertices (marked by star)              / \
    for subdivision of the face ABC when         vAB0---vAC0
    given coordinates of the on-edge               / \ / \
    vertices  AB[i] and AC[i].                 vAB1---*---vAC1
                                                 / \ / \ / \
                                             vAB2---*---*---vAC2
                                               / \ / \ / \ / \
                                              .---.---.---.---. 
    '''
   
    v = []
    for i in range(1,vAB.shape[0]):
        w = np.arange(1,i+1)/(i+1)
        for k in range(i):
            v.append(w[-1-k]*vAB[i,:] + w[k]*vAC[i,:])
    
    return(np.array(v).reshape(-1,3)) # reshape needed for empty return




################################################################################
#                          SCENE MANAGEMENT FUNCTIONS                          #
################################################################################

filename = path_to_ParaViz3D + 'traces/sphere/sphere720_2205999/extract/CompStopAndStart.txt'
timeJumpList_file = path_to_ParaViz3D + 'traces/sphere/sphere720_2205999/extract/TimeJumpList.txt'
ranks_file = path_to_ParaViz3D + 'traces/sphere/sphere720_2205999/extract/Ranks.txt'


timejumps = []
absolute_starting_timestamp = 0
with open(timeJumpList_file, 'r') as TimeJumpList:
    timestamp = TimeJumpList.readline()
    ignored_trace_initialization_offset = int(timestamp)
    timestamp = TimeJumpList.readline()
    while len(timestamp) != 0:
        jump = TimeJumpList.readline()
        timejumps.append([int(timestamp), int(jump)])
        timestamp = TimeJumpList.readline()
print(timejumps)

def recalculate_text(scene):
    frequency = 2400000000
    cyclesperframe = 200000
    nbcycles = bpy.context.scene.frame_current * cyclesperframe + ignored_trace_initialization_offset
    if timejumps == []:
        seconds = int(nbcycles // frequency)
        milliseconds = int(nbcycles * 1000 // frequency - seconds * 1000)
        microseconds = int(nbcycles * 1000000 // frequency - seconds * 1000000 - milliseconds * 1000)
        obj.data.body = f'{seconds}s {milliseconds:03d}ms {microseconds:03d}µs'
    else:
        for timestamp, jump in timejumps:
            while int(scene.frame_current) >= timestamp:
                print(nbcycles)
                nbcycles += jump
                print(nbcycles)
            seconds = int(nbcycles // frequency)
            milliseconds = int(nbcycles * 1000 // frequency - seconds * 1000)
            microseconds = int(nbcycles * 1000000 // frequency - seconds * 1000000 - milliseconds * 1000)
            obj.data.body = f'{seconds}s {milliseconds:03d}ms {microseconds:03d}µs'



def topology_colorize(coord, fading = 8):

    colors_topology = [(1,0,0,0),(1,0,0,0),(1,0,0,1),(1,0,0,0)]

    maxtime = 0
    for [x,y,z], cube_counter in zip(coord, range(len(coord))):
        mat = bpy.data.materials.new(name="colored%d" % cube_counter)
        ico = bpy.data.objects["cube_object%d" % cube_counter]
        ico.active_material = mat       


        frames_topology = [0, (1 + cube_counter) * 3 - fading, (1 + cube_counter) * 3, (1 + cube_counter) * 3 + fading]
        
        if maxtime < frames_topology[-1]:
            maxtime = frames_topology[-1]

        for f, c in zip(frames_topology, colors_topology):
            mat.diffuse_color = c
            mat.keyframe_insert(data_path="diffuse_color", frame=f, index = -1)
    bpy.context.scene.frame_end = int(maxtime)
    
    return


def trace_colorize(coord, fading = 1, comp_color = (0,0,1,0), MPI_begin_color = (1,0,0,1), MPI_end_color = (0,1,0,1), nb_ranks = 720):

    with open(filename, 'r') as file:
        
        frames = []
        colours = []
        maxtime = 0
        time = file.readline()
        for  cube_counter in range(nb_ranks):
            mat = bpy.data.materials.new(name="colored%d" % cube_counter)
            ico = bpy.data.objects["cube_object%d" % cube_counter]
            ico.active_material = mat       
            
            frames = []
            colours = []
            current_timestamp = 0
            previous_timestamp = current_timestamp
            next_color_is_red = False
            frames_appended = 0
            
            if len(time) != 0 and time[0] == '0':
                frames = []
                colours = []
                frames.append(0)
                frames_appended = 1
                colours.append(MPI_begin_color)
                time = file.readline()
                    

            while len(time) != 0 and time[0] != '0':
                previous_timestamp = current_timestamp
                current_timestamp = int(np.ceil(float(time)/200000)) #stop hardcoding
                if previous_timestamp < current_timestamp - fading:
                    frames.append(current_timestamp - fading)
                    frames.append(current_timestamp)
                    frames_appended = 2
                    if next_color_is_red:
                        colours.append(comp_color)    #transparent
                        colours.append(MPI_begin_color)    #red
                    else:
                        colours.append(MPI_end_color)    #green
                        colours.append(comp_color)    #transparent
                    next_color_is_red = not next_color_is_red
                elif previous_timestamp == current_timestamp - fading:
                    frames.append(current_timestamp)
                    frames_appended = 1
                    if next_color_is_red:
                        colours.append(MPI_begin_color)    #red
                    else:
                        colours.append(comp_color)    #transparent
                    next_color_is_red = not next_color_is_red
                else:
                    if frames_appended == 2:
                        upcolor = colours.pop()
                        if upcolor == comp_color:
                            colours.append(MPI_begin_color)
                        else:
                            colours.append(comp_color)
                        next_color_is_red = not next_color_is_red
                    if frames_appended == 1:
                        upcolor = colours.pop()
                        if upcolor == comp_color:
                            colours.append(MPI_begin_color)
                        else:
                            colours.append(comp_color)
                        next_color_is_red = not next_color_is_red
                time = file.readline()
                                                                    
            
            if maxtime < frames[-1]:
                maxtime = frames[-1]

            for f, c in zip(frames, colours):
                mat.diffuse_color = c
                mat.keyframe_insert(data_path="diffuse_color", frame=f, index = -1)
        
        print(len(frames))
        print(len(colours))
        for i in range(len(frames)):
            if colours[i] == (0,0,1,0):
                print(str(frames[i]) + " : transparent")
            else:
                print(str(frames[i]) + " : red")

        bpy.context.scene.frame_end = int(maxtime)
        
        bpy.app.handlers.frame_change_pre.append(recalculate_text)
                
    return

    
def vampir(scale=0.0000001, height=0.4, offset_x=0, offset_y=0):
    #with open(filenameRanks, 'r') as Ranks:
    #    nb_ranks = Ranks.readline()
    #    max_nb_events_for_a_rank = 500
    #    timelist = [[0] * max_nb_events_for_a_rank] * nb_ranks
    rank = -1
    import numpy as np
    ranks = [5, 4,14,6, 3,13,23,15,7, 2,12,22,32,24,16,8]
    rankstime = np.zeros((576,300))#[[0.0] * 9] * 576
    prev = 0
    cur = 0
    blob = 0
    with open(filename12, 'r') as file:
        time = file.readline()
        while len(time) != 0:
            if time[0] == '0':
                i = 0
                rank += 1
                color = (1,0,0,1)
                #cur = float(time)
                #rankstime[rank][i] = float(time)_data
            else:
                if i == 300:
                    i = 299
                #prev = cur
                #cur = float(time)
                #create_event(offset_x + scale * prev, scale * (cur -   prev), offset_y - height * rank, -height, -1, color)
                rankstime[rank][i] = (float(time))
                if color == (1,0,0,0):
                    color = (1,0,0,1)
                else:
                    color = (1,0,0,0)
            
            i += 1
            time = file.readline()

        offs = -1
        for j in ranks:
            offs += 1
            color = (1,0,0,1)
            for b in range(1,200):
                create_event(offset_x + scale * rankstime[j][b-1], scale * (rankstime[j][b] - rankstime[j][b-1]), offset_y - height * offs, -height, -1, color)
                if color == (1,0,0,0):
                    color = (1,0,0,1)
                else:
                    color = (1,0,0,0)
    
    
    

################################################################################
#                                     MAIN                                     #
################################################################################

deleteAllObjects()


# Clock
bpy.ops.object.text_add(enter_editmode=False, align='WORLD', location=(-3, -5, 0), scale=(1, 1, 1))
scene = bpy.context.scene
obj = bpy.context.scene.objects['Text']





verts = []
edges = []
faces = []   

# Creating the geometry
coordd, thefaces = icosphere(6)
coordd = [[4*x,4*y,4*z] for [x,y,z] in coordd]

#create_solid(coordd, type="cube")
for i in range(len(thefaces)):
    mesh_data = bpy.data.meshes.new("cube_data%d" % i)
    mesh_data.from_pydata(coordd,edges,[thefaces[i]])
    mesh_obj = bpy.data.objects.new("cube_object%d" % i, mesh_data)
    bpy.context.collection.objects.link(mesh_obj)
#topology_colorize(coordd)
trace_colorize(coordd)



# Link objects together to have them all move when moving one particular
for obj in  bpy.data.objects:
    if obj.name.startswith('event_object.'):
        obj.parent = bpy.data.objects["event_object"]
    if obj.name != "cube_object0" and obj.name.startswith('cube_object'):
        obj.parent = bpy.data.objects["cube_object0"]
    if obj.name.startswith('arrow'):
        obj.parent = bpy.data.objects["cube_object0"]



#ev.location += (1,1,1)
import mathutils

## get the object
#obj = bpy.data.objects["event_object"]
##obj.location.z += 5

# adjustment values
(x,y,z) = (-1.0,0,0)

## adding adjustment values to the property
#for f in range(0,25000,5000):
#    obj.keyframe_insert(data_path="location", frame=f, index = -1)
#    obj.location.x -= 25 #50000 * 0.0000001 * step
#    
##    obj.location += mathutils.Vector((x,y,z))



        



################################################################################
#                                 VIDEO OUTPUT                                 #
################################################################################ 


# CAMERA CREATION AND MOVEMENTS

scn = bpy.context.scene

# create the first camera
cam1 = bpy.data.cameras.new("Camera")
cam1.lens = 40

# create the first camera object
cam_obj1 = bpy.data.objects.new("Camera", cam1)
cam_obj1.location = (0, -20, 12)
cam_obj1.rotation_euler = (0.9774, 0, 0)
scn.collection.objects.link(cam_obj1)
#bpy.ops.view3d.object_as_camera() #(cam_obj1)


# LIGHT CREATION

# Create light datablock
light_data = bpy.data.lights.new(name="my-light-data", type='POINT')
light_data.energy = 1000

# Create new object, pass the light data 
light_object = bpy.data.objects.new(name="my-light", object_data=light_data)

# Link object to collection in context
bpy.context.collection.objects.link(light_object)

# Change light position
light_object.location = (0, 0, 5)


# RENDER OPTIONS

scn.render.engine = 'BLENDER_EEVEE_NEXT' #'CYCLES'
scn.cycles.device = 'CPU' #'GPU'
scn.render.filepath = path_to_ParaViz3D + "output_videos/"
scn.render.image_settings.file_format = 'FFMPEG' #'PNG'
#scn.render.image_settings.color_mode = 'RGBA'
#for png images #bpy.context.scene.render.image_settings.compression = 15 #in percent
scn.render.resolution_x = 1620 #480
scn.render.resolution_y = 1080 #270
scn.render.ffmpeg.format = 'MPEG4' #'MKV'
scn.render.ffmpeg.constant_rate_factor = 'HIGH' #'LOW' #'MEDIUM
scn.render.ffmpeg.ffmpeg_preset = 'BEST' #'GOOD' #'REALTIME'
scn.render.ffmpeg.codec = 'H264' #'AV1'
#scn.frame_start = 18
#scn.frame_end = 100
scn.render.fps = 60
#bpy.ops.wm.save_as_mainfile(filepath="/media/playerone/0851-0ECE/Buntu/vizhpc/frameworks/blender/blend_files/test.blend")
#bpy.data.materials["colored71"].node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.8, 0.8, 0.8, 1)
#bpy.data.materials["colored71"].node_tree.nodes["Principled BSDF"].inputs[18].default_value = 1


# APPERANCE

#bpy.context.space_data.shading.background_color = (0.00226413, 0.00226413, 0.00226413)
#bpy.context.space_data.overlay.show_overlays = False




#coord = circle_normal_to_vector(0, 0, 1, 3, (0, 0, 0), 12)
#create_solid(coorddd)


#bpy.data.materials["colored71"].node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.8, 0.8, 0.8, 1)




################################################################################
#                              SCENE DIRECTING                                 #
################################################################################

import math

bpy.ops.mesh.primitive_cube_add(size=0.01, location=(0, 0, 0))
target_object = bpy.context.active_object

# Create a camera
bpy.ops.object.camera_add(location=(0, 0, 20))
camera = bpy.context.active_object
camera.rotation_euler = (0.55, 0, 0)
camera.data.lens = 50
#camera.data.type = 'ORTHO'
#camera.data.ortho_scale = 15

# Add an empty object at the center to act as a pivot point
bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
pivot = bpy.context.active_object
    
# Parent the camera to the empty
camera.parent = pivot

# Make the camera always point to the cube
constraint = camera.constraints.new(type='TRACK_TO')
constraint.target = target_object
constraint.track_axis = 'TRACK_NEGATIVE_Z'
constraint.up_axis = 'UP_Y'

# Animate the empty to rotate around Z axis (turns the camera)
pivot.rotation_mode = 'XYZ'

frame_start = 0#600
frame_end = 5100#9000
bpy.context.scene.frame_start = frame_start
bpy.context.scene.frame_end = frame_end

# Set keyframes for rotation
pivot.rotation_euler = (0, 0, 0)
pivot.keyframe_insert(data_path="rotation_euler", frame=frame_start)

pivot.rotation_euler = (-math.radians(25), 0, 0*math.radians(40))
pivot.keyframe_insert(data_path="rotation_euler", frame=frame_end)

# Set interpolation to linear for constant speed
for fcurve in pivot.animation_data.action.fcurves:
    for keyframe in fcurve.keyframe_points:
        keyframe.interpolation = 'LINEAR'

for o in bpy.context.scene.objects:
    if o.name == "cube_object5":
        o.select_set(True)