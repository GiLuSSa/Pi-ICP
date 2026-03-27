""" 
 _____ _____   _   _                  _       _     _   ___ ____ ____  
|  ___| ____| | \ | | ___  _ __  _ __(_) __ _(_) __| | |_ _/ ___|  _ \ 
| |_  |  _|   |  \| |/ _ \| '_ \| '__| |/ _` | |/ _` |  | | |   | |_) |
|  _| | |___  | |\  | (_) | | | | |  | | (_| | | (_| |  | | |___|  __/ 
|_|   |_____| |_| \_|\___/|_| |_|_|  |_|\__, |_|\__,_| |___\____|_|    
                                        |___/                          
Developed by Giulio Lucio Sergio Sacco under the supervision of Sinan Acikgoz
In Oxford, 2024.
Thanks to Yiyan Liu who provided an implementation of NR ICP (Amberg et al. 2007) algorithm that stated this research.

The code is messy as my bedroom when i was a teenager and it's suposed to run in python3 on Rhino 8 where part of the pre and post processing is performed.

Imputs:
sourceMeshVer       = Rhino points. The points subjected to boundary conditions must be at the end of the list. [V] = [V_free, V_landmark]^T 
sourceMeshElem      = Rhino triangular mesh elements as a list of the 3 vertexes connected indexex (e.g. {123;12;45}). 
LandmarkUvec        = List of displacement vectors, zero or non zero.
targetGeometry      = Target geometry as rhino nurbs, mesh or point cloud
alpha_max           = float
alpha_min           = float
alpha_step          = int -> how many alpha step are required
epsilon_step        = int -> maximum number of epsilon iterations
epsilonThreshhold   = float -> epsilon loop ends if ||U_n - U_n-1|| < epsilonThreshhold
wk                  = the k component of W (W = w * r, 0 < w,r < 1) that represent the reability of the match in the distance term. []
cracks              = 3D curves representing cracks [curve list]

Remember to install all the libraries.
Have fun!
""" 
# requirements: numpy, scipy

import rhinoscriptsyntax as rs
import Rhino as rn
import numpy as np
from scipy import sparse
from scipy import linalg
import math
import ghpythonlib.treehelpers as th

#debug
import sys
import time
#import copy

""" DEBUG """
#GET START TIME
time_start = time.time()
time_start_string = time.strftime("%y-%m-%d %H:%M", time.gmtime(time_start))

#PLOT Status info - Initialization
with open("StatusDebug.txt", "w+") as stf:
    stf.write(f"{time_start_string}\n\n")
    
print("VOMO")

""" INPUT """
# Assign alpha values based on an exponential decay scheme, as suggested by Hasler et al., 2009
alpha = [alpha_max * np.exp(-(1 / alpha_step * np.log(alpha_max / alpha_min)) * i) for i in np.arange(alpha_step)]
# Ensure the last alpha value is exactly the minimum specified
alpha[-1] = alpha_min
epsilon = epsilon_maxStep

if wk == None:
    print("No wk in use")
    wk = 1

""" SEtuP """
source_mesh_vertices = []
for point in sourceMeshVer:  #Conversion of rhino mesh into vertex matrix
    source_mesh_vertices.append([point.X, point.Y, point.Z])
source_mesh_vertices = np.asarray(source_mesh_vertices)

sourceMeshVerLen = len(sourceMeshVer) 

source_mesh_triangles = []
for face in sourceMeshElem: #Conversion of rhino mesh into edges matrix
    source_mesh_triangles.append([face[0], face[1], face[2]])
source_mesh_triangles = np.asarray(source_mesh_triangles)

sourceMeshElemLen = len(sourceMeshElem)

imposed_displacements = []
for vec in landmarkUvec: #Conversion of rhino mesh into edge matrix
    imposed_displacements.append([vec.X, vec.Y, vec.Z])
imposed_displacements = np.asarray(imposed_displacements)

imposed_displacements_Len = len(imposed_displacements)

UVecExp = [] #this is important to initialize the vector
for i in sourceMeshVer:
    UVecExp.append(rs.CreateVector(0, 0, 0))

#B global:
row = []
col = []
data = []
i = 0
for element in source_mesh_triangles:
    v_i = source_mesh_vertices[element[0]]
    v_j = source_mesh_vertices[element[1]]
    v_k = source_mesh_vertices[element[2]]
    
    side1 = v_j - v_i
    side2 = v_k - v_i

    elementAreaInv = np.reciprocal(0.5*np.linalg.norm(np.cross(side1, side2)))
  
    row.append(i*3)
    col.append(i*9)
    data.append(elementAreaInv*(source_mesh_vertices[element[1]][1] - source_mesh_vertices[element[1]][2]))
        
    row.append(i*3)
    col.append(i*9+3)
    data.append(elementAreaInv*(source_mesh_vertices[element[1]][2] - source_mesh_vertices[element[1]][0]))
        
    row.append(i*3)
    col.append(i*9+6)
    data.append(elementAreaInv*(source_mesh_vertices[element[1]][0] - source_mesh_vertices[element[1]][1]))
    
    row.append(i*3+1)
    col.append(i*9+1)
    data.append(elementAreaInv*(source_mesh_vertices[element[0]][2] - source_mesh_vertices[element[0]][1]))
        
    row.append(i*3+1)
    col.append(i*9+4)
    data.append(elementAreaInv*(source_mesh_vertices[element[0]][0] - source_mesh_vertices[element[0]][2]))
        
    row.append(i*3+1)
    col.append(i*9+7)
    data.append(elementAreaInv*(source_mesh_vertices[element[0]][1] - source_mesh_vertices[element[0]][0]))

    row.append(i*3+2)
    col.append(i*9)
    data.append(elementAreaInv*(source_mesh_vertices[element[0]][2] - source_mesh_vertices[element[0]][1]))
       
    row.append(i*3+2)
    col.append(i*9+1)
    data.append(elementAreaInv*(source_mesh_vertices[element[1]][1] - source_mesh_vertices[element[1]][2]))
        
    row.append(i*3+2)
    col.append(i*9+3)
    data.append(elementAreaInv*(source_mesh_vertices[element[0]][0] - source_mesh_vertices[element[0]][2]))
        
    row.append(i*3+2)
    col.append(i*9+4)
    data.append(elementAreaInv*(source_mesh_vertices[element[1]][2] - source_mesh_vertices[element[1]][0]))
        
    row.append(i*3+2)
    col.append(i*9+6)
    data.append(elementAreaInv*(source_mesh_vertices[element[0]][1] - source_mesh_vertices[element[0]][0]))

    row.append(i*3+2)
    col.append(i*9+7)
    data.append(elementAreaInv*(source_mesh_vertices[element[1]][0] - source_mesh_vertices[element[1]][1]))
    
    i += 1

B_Global = sparse.csc_matrix((data, (row, col)), shape=(3*sourceMeshElemLen, 9*sourceMeshElemLen))

# T GLOGAL
Debug1 = []
row = []
col = []
data = []
rowdebug = []
coldebug = []
datadebug = []
i = 0
T_EXP = []
for element in source_mesh_triangles: # T GLOBAL LOOP
   
    #TRIANGOLI RIFERIMENTO ORIGINALI
    v1 = source_mesh_vertices[element[0]]
    v2 = source_mesh_vertices[element[1]]
    v3 = source_mesh_vertices[element[2]]
    
    #VERSOR OF THE ELEMENT COORDINATE SYSTEM
    x = (v2-v1) / np.linalg.norm(v2-v1)
    z = np.cross(v2-v1, v3-v1) / np.linalg.norm(np.cross(v2-v1, v3-v1))
    y = np.cross(z, x) / np.linalg.norm(np.cross(z, x))

    T_EXP.append([x,y,z])#this is needed for the postprocessing of the principal strains

    j = 0 
    for index in element:
        for r in range(3):
                row.append(9*i + 3*j)
                col.append(3*index + r)
                data.append(x[r])

                row.append(9*i + 3*j +1)
                col.append(3*index + r)
                data.append(y[r])

                row.append(9*i + 3*j +2)
                col.append(3*index + r)
                data.append(z[r])                        

                #DEBUG!
                rowdebug.append(9*i + 3*j + r)
                coldebug.append(3*index + r)
                datadebug.append(1)
        j += 1
    i += 1
T_EXP = np.asanyarray(T_EXP)
T_Global = sparse.csc_matrix((data, (row, col)), shape=(9*sourceMeshElemLen, 3*sourceMeshVerLen))

#DEBUG
T_Global_1 = sparse.csc_matrix((datadebug, (rowdebug, coldebug)), shape=(9*sourceMeshElemLen, 3*sourceMeshVerLen))

#COMPLETE GLOBAL FEM MATRIX -> A_FEM
A_FEM = B_Global * T_Global

"""-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^"""
# # #      QUESTSO E IL PUNTO IN CUI INIZIA IL LOOP      # # #
"""-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^"""

U_increment = [0,0] #initilization of this vector used to break the loop

for alp in alpha: #Alpha loop
    print("alpha:", alp, "and epsilon: ", end="")
    #PLOT Advancements - Alpha
    with open("StatusDebug.txt", "a") as stf:
        stf.write(f"Alpha: {alp}, and epsilon: ")

    for eps in range(epsilon): #Epsilon loop
        
        print(eps, "", end="")
        #PLOT Advancements - Epsilon
        with open("StatusDebug.txt", "a") as stf:
            stf.write(f"{eps}, ")

        new_template = [] #update the oritinal template displacing it by the previous U
        for ver, vec in zip(sourceMeshVer, UVecExp):
            new_template.append(ver + vec)

        #HERE I MISSI THE DEFINITION OF A W_w based on the distances measured here. 
        ww = 1

        target_vertices = [] #New correspondences must be evaluated on the target based on the new template
        if type(targetGeometry) == rn.Geometry.Brep: #Questa cosa può essere riscritta meglio con meno duplicazione
            for point in new_template:
                new_target_vertices_temp = rn.Geometry.Brep.ClosestPoint(targetGeometry, point)
                target_vertices.append([new_target_vertices_temp.X, new_target_vertices_temp.Y, new_target_vertices_temp.Z])
                #tarTempDist.append(new_target_vertices_temp.DistanceTo(point))
            target_vertices = np.asarray(target_vertices)
        elif type(targetGeometry) == rn.Geometry.Mesh:
            for point in new_template:
                new_target_vertices_temp = rn.Geometry.Mesh.ClosestPoint(targetGeometry, point)
                target_vertices.append([new_target_vertices_temp.X, new_target_vertices_temp.Y, new_target_vertices_temp.Z])
                #tarTempDist.append(new_target_vertices_temp.DistanceTo(point))
            target_vertices = np.asarray(target_vertices)
        elif type(targetGeometry) == rn.Geometry.PointCloud:
            #sys.exit("The use of point cloud as target is not supported yet. Just mesh it.")
            for point in new_template:
                new_target_vertices_index_temp = rn.Geometry.PointCloud.ClosestPoint(targetGeometry, point)
                new_target_vertices_temp = targetGeometry[new_target_vertices_index_temp]
                target_vertices.append([new_target_vertices_temp.X, new_target_vertices_temp.Y, new_target_vertices_temp.Z])
            target_vertices = np.asarray(target_vertices)
        else:
            print("Your impus is a",  type(targetGeometry))
            sys.exit("Give me proper data! I need a brep, mesh or point cloud.")
        #W vector (or a scalar if both ww and wk are scalars). 
        W = ww * wk

        #A_Dist: NB: this is just a huge I_(3*templateVertexes) scaled by W(w,r) !!! THIS example must be used to simplyty other constructions of I like matrixes.
        A_Dist = sparse.csc_matrix((np.repeat(W, 3), (np.arange(3*sourceMeshVerLen), np.arange(3*sourceMeshVerLen))), shape=(3*sourceMeshVerLen, 3*sourceMeshVerLen))
        
        #A matrix
        A = sparse.vstack([alp * A_FEM, A_Dist], format="csc")
        
        #B_f
        vFlatten = (source_mesh_vertices).flatten()
        uFlatten = (target_vertices).flatten()
        uTarget = np.repeat(W, 3) * (uFlatten - vFlatten) # Moltiplicare tutto per W inq uesto punto dovrebbe funzionare perché dovrebbero avere tutti la stessa lunghezza? Forse.

        row = []
        col = []
        data = []
        i = 0
        breaker = (3*sourceMeshElemLen + 3*(sourceMeshVerLen-imposed_displacements_Len))
        for U_component in uTarget:
            row.append(3*sourceMeshElemLen + i)
            col.append(0)
            data.append(U_component)
            i += 1
            if (3*sourceMeshElemLen + i) == breaker:
                break
        B_f = sparse.csc_matrix((data, (row, col)), shape=(3*sourceMeshElemLen + 3*(sourceMeshVerLen-imposed_displacements_Len), 1))

        #A_ff
        A_ff = A[:A.get_shape()[0] - imposed_displacements_Len*3, :A.get_shape()[1] - imposed_displacements_Len*3]

        #A_fl
        A_fl = A[:A.get_shape()[0] - imposed_displacements_Len*3, A.get_shape()[1] - imposed_displacements_Len*3:]

        #U_l
        row = []
        col = []
        data = []
        i = 0
        imposed_displacements_flat = imposed_displacements.flatten()
        for component in imposed_displacements_flat:
            row.append(i)
            col.append(0)
            data.append(component)
            i += 1
        U_l = sparse.csc_matrix((data, (row, col)), shape=(3*imposed_displacements_Len, 1))

        #B_tilde
        Bilde = B_f - A_fl*U_l

        #SOLVE! linalg.lsmr() is suposed to be faster, but I'm not sure
        U, istop, itn, normt = sparse.linalg.lsqr(A_ff, Bilde.toarray().flatten(), atol=1e-6)[:4]
       
        UExp = U.reshape((sourceMeshVerLen-imposed_displacements_Len, 3))
        
        UVecExp = [] #this is used to compute the next template
        for point in UExp:
            UVecExp.append(rs.CreateVector(point[0], point[1], point[2]))
        UVecExp.extend(landmarkUvec)

        U_increment[1] = np.linalg.norm(U)
        if abs(U_increment[0]-U_increment[1]) < epsilonThreshold:
            U_increment[0] = 0
            U_increment[1] = 0
            print("")
            break

        if eps == epsilon-1:
            print("")
            U_increment[0] = 0
            U_increment[1] = 0
        else:
            U_increment[0] = U_increment[1]
    #PLOT Advancements - new row
    with open("StatusDebug.txt", "a") as stf:
        stf.write(f"DONE!\n")

new_template = [] #update the oritinal template displacing it by the previous U
for ver, vec in zip(sourceMeshVer, UVecExp):
    new_template.append(ver + vec)

#CENTROID OF DEFORMED FACES
newElementCentroid = []
for element in sourceMeshElem:
    newElementCentroid.append((new_template[element[0]] + new_template[element[1]] + new_template[element[2]])/3)

#PLOT Advancements - Intermezzo
with open("StatusDebug.txt", "a") as stf:
    stf.write(f"\nSECOND LOOP \n\n")

#Second iteration adding cracks 
if cracks == None:
    print("You are a lazy ass! Implement cracks! It's free.")
else:
    print("Second Iretation with cracks")
    tmpfuck = 1/(cracks_influence/3)
    wr = np.zeros([len(cracks)+1, sourceMeshElemLen]) #the data structure is [[global w_r][crack_1 w_r]...[crack_n w_r]] so to be easily loopable
    cv = 0
    for curve in cracks:
        pt = 0
        for i in newElementCentroid:
            r, t = curve.ClosestPoint(i)
            ptOnCurve = curve.PointAt(t)
            dist = i.DistanceTo(ptOnCurve)
            wr[cv+1, pt] = math.exp( - pow(tmpfuck * dist, 2)) #Crack function
            wr[0, pt] += wr[cv+1, pt] #This should be limited between 0 and +1!
            pt += 1
        cv +=1
    wr = 1-wr #it's better to do this here because in this way it's easier to sum up the single crack r 
    
    #Output the lists for postprocessing of crack width or something
    w_r_exp = th.list_to_tree(np.ndarray.tolist(wr), source=[0,0]) #This is practicaly black magic to convert a numpy array into a grashopper tree

    """-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^""" #BUG
    """-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^""" #BUG
    """-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^""" #BUG
    """-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^""" #BUG

    alphaVec = wr[0, :]

    U_increment = [0,0] #initilization of this vector used to break the loop
    for alp in alpha: #Alpha loop
        print("alpha:", alp, "and epsilon: ", end="")
        #PLOT Advancements - Alpha
        with open("StatusDebug.txt", "a") as stf:
            stf.write(f"Alpha: {alp}, and epsilon: ")

        a_alphaCrack = alphaVec * alp
        a_alphar = sparse.csc_matrix((np.repeat(a_alphaCrack, 3), (np.arange(3*sourceMeshElemLen), np.arange(3*sourceMeshElemLen))), shape=(3*sourceMeshElemLen, 3*sourceMeshElemLen))

        for eps in range(epsilon): #Epsilon loop
            
            print(eps, "", end="")
            #PLOT Advancements - Epsilon
            with open("StatusDebug.txt", "a") as stf:
                stf.write(f"{eps}, ")

            new_template = [] #update the oritinal template displacing it by the previous U
            for ver, vec in zip(sourceMeshVer, UVecExp):
                new_template.append(ver + vec)

            #HERE I MISSI THE DEFINITION OF A W_w based on the distances measured here. 
            ww = 1

            target_vertices = [] #New correspondences must be evaluated on the target based on the new template
            if type(targetGeometry) == rn.Geometry.Brep: #Questa cosa può essere riscritta meglio con meno duplicazione
                for point in new_template:
                    new_target_vertices_temp = rn.Geometry.Brep.ClosestPoint(targetGeometry, point)
                    target_vertices.append([new_target_vertices_temp.X, new_target_vertices_temp.Y, new_target_vertices_temp.Z])
                    #tarTempDist.append(new_target_vertices_temp.DistanceTo(point))
                target_vertices = np.asarray(target_vertices)
            elif type(targetGeometry) == rn.Geometry.Mesh:
                for point in new_template:
                    new_target_vertices_temp = rn.Geometry.Mesh.ClosestPoint(targetGeometry, point)
                    target_vertices.append([new_target_vertices_temp.X, new_target_vertices_temp.Y, new_target_vertices_temp.Z])
                    #tarTempDist.append(new_target_vertices_temp.DistanceTo(point))
                target_vertices = np.asarray(target_vertices)
            elif type(targetGeometry) == rn.Geometry.PointCloud:
                for point in new_template:
                    new_target_vertices_index_temp = rn.Geometry.PointCloud.ClosestPoint(targetGeometry, point)
                    new_target_vertices_temp = targetGeometry[new_target_vertices_index_temp]
                    target_vertices.append([new_target_vertices_temp.X, new_target_vertices_temp.Y, new_target_vertices_temp.Z])
                target_vertices = np.asarray(target_vertices)
            else:
                print("Your impus is a",  type(targetGeometry))
                sys.exit("Give me proper data! I need a brep, mesh or point cloud.")
            #W vector (or a scalar if both ww and wk are scalars). 
            W = ww * wk

            #A_Dist: NB: this is just a huge I_(3*templateVertexes) scaled by W(w,r) !!! THIS example must be used to simplyty other constructions of I like matrixes.
            A_Dist = sparse.csc_matrix((np.repeat(W, 3), (np.arange(3*sourceMeshVerLen), np.arange(3*sourceMeshVerLen))), shape=(3*sourceMeshVerLen, 3*sourceMeshVerLen))
            
            #A matrix
            """
            print("Bobbobeo")
            print("a_alphar", sparse.linalg.norm(a_alphar))
            print("A_FEM", sparse.linalg.norm(A_FEM))
            print("a_alphar @ A_FEM", sparse.linalg.norm(a_alphar @ A_FEM))
            """
            A = sparse.vstack([a_alphar @ A_FEM, A_Dist], format="csc")#alp * A_FEM
            
            #B_f
            vFlatten = (source_mesh_vertices).flatten()
            uFlatten = (target_vertices).flatten()
            uTarget = np.repeat(W, 3) * (uFlatten - vFlatten) # Moltiplicare tutto per W inq uesto punto dovrebbe funzionare perché dovrebbero avere tutti la stessa lunghezza? Forse.

            row = []
            col = []
            data = []
            i = 0
            breaker = (3*sourceMeshElemLen + 3*(sourceMeshVerLen-imposed_displacements_Len))
            for U_component in uTarget:
                row.append(3*sourceMeshElemLen + i)
                col.append(0)
                data.append(U_component)
                i += 1
                if (3*sourceMeshElemLen + i) == breaker:
                    break
            B_f = sparse.csc_matrix((data, (row, col)), shape=(3*sourceMeshElemLen + 3*(sourceMeshVerLen-imposed_displacements_Len), 1))

            #A_ff
            A_ff = A[:A.get_shape()[0] - imposed_displacements_Len*3, :A.get_shape()[1] - imposed_displacements_Len*3]

            #A_fl
            A_fl = A[:A.get_shape()[0] - imposed_displacements_Len*3, A.get_shape()[1] - imposed_displacements_Len*3:]

            #U_l
            row = []
            col = []
            data = []
            i = 0
            imposed_displacements_flat = imposed_displacements.flatten()
            for component in imposed_displacements_flat:
                row.append(i)
                col.append(0)
                data.append(component)
                i += 1
            U_l = sparse.csc_matrix((data, (row, col)), shape=(3*imposed_displacements_Len, 1))

            #B_tilde
            Bilde = B_f - A_fl*U_l

            #SOLVE! linalg.lsmr() is suposed to be faster, but I'm not sure
            U, istop, itn, normt = sparse.linalg.lsqr(A_ff, Bilde.toarray().flatten(), atol=1e-6)[:4]
        
            UExp = U.reshape((sourceMeshVerLen-imposed_displacements_Len, 3))
            
            UVecExp = [] #this is used to compute the next template
            for point in UExp:
                UVecExp.append(rs.CreateVector(point[0], point[1], point[2]))
            UVecExp.extend(landmarkUvec)

            U_increment[1] = np.linalg.norm(U)
            if abs(U_increment[0]-U_increment[1]) < epsilonThreshold:
                U_increment[0] = 0
                U_increment[1] = 0
                print("")
                break

            if eps == epsilon-1:
                print("")
                U_increment[0] = 0
                U_increment[1] = 0
            else:
                U_increment[0] = U_increment[1]
        #PLOT Advancements - new row
        with open("StatusDebug.txt", "a") as stf:
            stf.write(f"DONE!\n")



    """-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^""" #END BUG




"""-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^"""
# # #                   POST PROCESSING                  # # #
"""-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^-_-^"""

Targets = [] #TEMPORARY DEBUG
for point in target_vertices:
    Targets.append(rs.CreatePoint(point[0], point[1], point[2]))

#FINAL TIME TAG
time_end = time.time()
"""
#DeBuGGoNe
USig_el = T_Global[:,:3*(sourceMeshVerLen-imposed_displacements_Len)] @ U
U_el = T_Global_1[:,:3*(sourceMeshVerLen-imposed_displacements_Len)] @ U

USig_el = USig_el.reshape((len(USig_el)//3, 3))
USig_elExp = []
for point in USig_el:
     USig_elExp.append(rs.CreateVector(point[0], point[1], point[2]))

U_el = U_el.reshape((len(U_el)//3, 3))
UExp = U.reshape((sourceMeshVerLen-imposed_displacements_Len, 3))
U_elExp = []
for point in U_el:
    U_elExp.append(rs.CreateVector(point[0], point[1], point[2]))
### ^ ???
"""
U_Strains = []
for vec in UVecExp: #This is silly. it was made like that for programming convenience, but it's obviouly bad for performance
    U_Strains.append(vec.X)
    U_Strains.append(vec.Y)
    U_Strains.append(vec.Z)
    
strains = A_FEM * U_Strains #this is what it looks, plus reshaping to have a neat data structure of v^T stacked vertically 
strains = strains.reshape((sourceMeshElemLen, 3))

strainTensor = [] #Just 2x2 plane strain tensor: [E_xx, 1/2 gamma_xy][1/2 gamma_xy, E_yy]
for vector in strains:
    strainTensor.append([[vector[0], 0.5*vector[2]],[0.5*vector[2],vector[1]]])

principalStrainLocal = [] #Here the eigen problem is solved to have principal strains as vectors (lambda*V) arranged as as column vectors as it should be.
eigenSign = []
for tensor in strainTensor:
    eigenval, eigenvec = linalg.eig(tensor) #magic *-*
    eigenval = eigenval.real #**** the imaginary part which is 0
    eigenSign.append(np.sign(eigenval[0]))
    eigenSign.append(np.sign(eigenval[1]))
    principalStrainLocal.append([[eigenval[0]*eigenvec[0,0],eigenval[1]*eigenvec[0,1],0],[eigenval[0]*eigenvec[1,0],eigenval[1]*eigenvec[1,1],0],[0,0,0]])
principalStrainLocal = np.asarray(principalStrainLocal)

principalStrainGlobal = [] #Those are arranged in a list of [[E_1, E_2],...,[E_1, E_2]] to be displayed on rhino on the centroids of elements.
for matrix, Texp in zip(principalStrainLocal, T_EXP):
    PSG = Texp.transpose() @ matrix
    principalStrainGlobal.append(rs.CreateVector(PSG[0,0], PSG[1,0], PSG[2,0]))
    principalStrainGlobal.append(rs.CreateVector(PSG[0,1], PSG[1,1], PSG[2,1])) #Here I'm outputting the vectors

#??? #TO BE SCRAPPED DEBUG
strainsVec = []
for elem in strains:
    strainsVec.append(rs.CreateVector(elem[0], elem[1], elem[2]))

print("Final solution! End reason:", istop, "number iterations:", itn, "norm(b - Ax) =", normt)
print("Total elapsed time:", round(time_end - time_start,2), "seconds")
print("Thanks fur the consideration, and all the fish. 🐬")
