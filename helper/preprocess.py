import trimesh
import numpy as np
import os
from tqdm import tqdm
import json
import scipy
# Load the mesh from file
dir_path = "/home/chethanc/reconstruction/data/mini_shapenet"
derived_dir_path = "/home/chethanc/reconstruction/data/mini_shapenet_mat"
def process_obj_files(dir_path):
    for folder in os.listdir(dir_path):
        print(f"Processing object id {folder}")
        folder_path = os.path.join(dir_path,folder)
        folder_path_derived = os.path.join(derived_dir_path, folder)
        if os.path.isdir(folder_path):
            os.makedirs(folder_path_derived, exist_ok=True)
            count = 0
            for shape_folder in tqdm(os.listdir(folder_path)):
                shape_folder_path = os.path.join(folder_path,shape_folder)  
                if os.path.isdir(shape_folder_path):
                    obj_file = os.path.join(shape_folder_path, 'models','model_normalized.obj')
                    sample_points_path = os.path.join(folder_path_derived, shape_folder+'.mat')
                    sample_points(obj_file, sample_points_path)
                    count+=1
                    if count>=5000:
                        break


def sample_points(obj_file, sample_points_path):
    # Load the mesh from file
    mesh = trimesh.load(obj_file,force='mesh')
    # Generate random points on the surface of the mesh
    points, face_ids = trimesh.sample.sample_surface(mesh, count=10000)
    normals  = mesh.face_normals[face_ids]
    sampled_mesh = {'v':points.tolist(), 'f':normals.tolist()}
    # Save the points to a file
    scipy.io.savemat (sample_points_path,sampled_mesh)

if __name__=="__main__":
    process_obj_files(dir_path)
