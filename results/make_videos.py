import os
import sys
import bpy
import numpy as np

from os import listdir, makedirs, system
from os.path import exists
from os.path import isdir

data_path = "./results/blender_files/"
directories = sorted([f for f in listdir(data_path) if isdir(data_path + f)])
for d in directories:
  files = sorted([f for f in listdir(data_path+d) if f.endswith(".blend") and
                  not f.startswith(".")])
  for f in files:
    dump_path = data_path + d
    bpy.ops.wm.open_mainfile(filepath=dump_path+"/"+f)
    inp_bvh = dump_path+"/"+f.split(".blend")[0]+"_inp.bvh"
    gt_bvh = dump_path+"/"+f.split(".blend")[0]+"_gt.bvh"
    ours_bvh = dump_path+"/"+f.split(".blend")[0]+".bvh"
    cpy_bvh = dump_path+"/"+f.split(".blend")[0]+"_cpy.bvh"

    try:
      for obj_name in bpy.data.objects.keys():
        if "from=" in obj_name:
          bpy.ops.object.select_all(action="DESELECT")
          bpy.data.objects[obj_name].select = True
          bpy.ops.object.delete() 
  
      bpy.ops.import_anim.bvh(filepath=inp_bvh, global_scale=0.01)
      bpy.ops.import_anim.bvh(filepath=gt_bvh, global_scale=0.01)
      bpy.ops.import_anim.bvh(filepath=ours_bvh, global_scale=0.01)
      bpy.ops.import_anim.bvh(filepath=cpy_bvh, global_scale=0.01)
    except Exception:
      continue

    bpy.data.objects[f.split(".blend")[0]+"_inp"].location = (-6,0,0)
    bpy.data.objects[f.split(".blend")[0]+"_gt"].location = (-2,0,0)
    bpy.data.objects[f.split(".blend")[0]].location = (2,0,0)
    bpy.data.objects[f.split(".blend")[0]+"_cpy"].location = (6,0,0)

    """ Target character skins """
    if "Liam" in f.split("to=")[-1]:
      parts = ["Body", "Bottoms", "Eyes", "Shoes", "Tops", "default"]
    elif "Malcolm" in f.split("to=")[-1]:
      parts = ["Body", "Bottoms", "Eyes", "Hair", "Hats", "Shoes",
               "Tops", "default"]
    elif "Warrok" in f.split("to=")[-1]:
      parts = ["Warrok"]
    elif "Mutant" in f.split("to=")[-1]:
      parts = ["MutantMesh"]
    else:
      pdb.set_trace()

    parent = bpy.data.objects[f.split(".blend")[0]+"_gt"]
    for part in parts:
      bpy.data.objects[part+".001"].location = (-2,0,0)
      bpy.ops.object.select_all(action="DESELECT")
      child = bpy.data.objects[part+".001"]
      parent.select = True
      child.select = True 
      bpy.context.scene.objects.active = parent
      bpy.ops.object.parent_set(type="ARMATURE")

    parent = bpy.data.objects[f.split(".blend")[0]]
    for part in parts:
      bpy.data.objects[part+".002"].location = (2,0,0)
      bpy.ops.object.select_all(action="DESELECT")
      child = bpy.data.objects[part+".002"]
      parent.select = True
      child.select = True 
      bpy.context.scene.objects.active = parent
      bpy.ops.object.parent_set(type="ARMATURE")

    parent = bpy.data.objects[f.split(".blend")[0]+"_cpy"]
    for part in parts:
      bpy.data.objects[part+".003"].location = (6,0,0)
      bpy.ops.object.select_all(action="DESELECT")
      child = bpy.data.objects[part+".003"]
      parent.select = True
      child.select = True 
      bpy.context.scene.objects.active = parent
      bpy.ops.object.parent_set(type="ARMATURE")


    """ Input character skins """
    if "Granny" in f.split("from=")[-1].split("_to")[0]:
      parts = ["Fitness_Grandma_BodyGeo", "Fitness_Grandma_BrowsAnimGeo",
               "Fitness_Grandma_EyesAnimGeo", "Fitness_Grandma_MouthAnimGeo"]
    elif "Peasant" in f.split("from=")[-1].split("_to")[0]:
      parts = ["Peasant_Man"]
    elif "AJ" in f.split("from=")[-1].split("_to")[0]:
      parts = ["Boy01_Body_Geo", "Boy01_Brows_Geo",
               "Boy01_Eyes_Geo", "h_Geo"]
    elif "Vegas" in f.split("from=")[-1].split("_to")[0]:
      parts = ["newVegas:Elvis_BodyGeo", "newVegas:Elvis_BrowsAnimGeo",
               "newVegas:Elvis_EyesAnimGeo", "newVegas:Elvis_MouthAnimGeo"]
    elif "Claire" in f.split("from=")[-1].split("_to")[0]:
      parts = ["Girl_Body_Geo", "Girl_Brows_Geo",
               "Girl_Brows_Geo", "Girl_Mouth_Geo", "Girl_Eyes_Geo"]
    elif "Kaya" in f.split("from=")[-1].split("_to")[0]:
      parts = ["OccupyGuy_Body_Mesh", "OccupyGuy_BrowsAnimGeo",
               "OccupyGuy_EyesAnimGeo", "OccupyGuy_MouthAnimGeo"]
    elif "Mutant" in f.split("from=")[-1].split("_to")[0]:
      parts = ["MutantMesh"]
    else:
      pdb.set_trace()

    parent = bpy.data.objects[f.split(".blend")[0]+"_inp"]
    for part in parts:
      bpy.data.objects[part].location = (-6,0,0)
      bpy.ops.object.select_all(action="DESELECT")
      child = bpy.data.objects[part]
      parent.select = True
      child.select = True 
      bpy.context.scene.objects.active = parent
      bpy.ops.object.parent_set(type="ARMATURE")

    bpy.ops.object.select_all(action="DESELECT")
    bpy.data.objects["Floor"].select = True
    bpy.ops.object.delete()
    bpy.ops.mesh.primitive_plane_add(location=(0,0,0))
    bpy.data.objects["Plane"].name = "Floor"
    mat = bpy.data.materials.new(name="Floor")
    bpy.data.objects["Floor"].data.materials.append(mat)
    bpy.context.scene.objects.active = bpy.data.objects["Floor"]
    bpy.context.object.active_material.diffuse_color = (0.154, 0.154, 0.154)
    bpy.data.objects["Floor"].scale = (8.0, 8.0, 8.0)

    bpy.ops.object.select_all(action="DESELECT")
    bpy.data.objects["Camera"].select = True
    bpy.ops.object.delete()
    cam = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam)
    bpy.context.scene.objects.link(cam_obj)
    bpy.context.scene.render.resolution_x = 2500
    bpy.context.scene.render.resolution_y = 800
#    bpy.data.objects["Camera"].location = (0.5, -17.0, 0.6)
    bpy.data.objects["Camera"].location = (0.0, -22.0, 1.0)
    bpy.data.objects["Camera"].rotation_euler = (np.pi*(95.0/180.0), 0.0, 0.0)
    bpy.data.cameras.values()[0].angle = 0.872665

    bpy.ops.object.text_add(location=(-6.0, 0.0, 4.0))
    bpy.data.objects["Text"].rotation_euler = (1.570796, 0.0, 0.0)
    mat = bpy.data.materials.new(name="Text")
    bpy.data.objects["Text"].data.materials.append(mat)
    bpy.context.scene.objects.active = bpy.data.objects["Text"]
    bpy.context.object.active_material.diffuse_color = (0.0, 0.0, 0.0)
    bpy.data.objects["Text"].data.align_x = "CENTER"
    bpy.data.objects["Text"].data.body = "Input"
    bpy.data.objects["Text"].data.size = 0.8

    bpy.ops.object.text_add(location=(-2.0, 0.0, 4.0))
    bpy.data.objects["Text.001"].rotation_euler = (1.570796, 0.0, 0.0)
    mat = bpy.data.materials.new(name="Text.001")
    bpy.data.objects["Text.001"].data.materials.append(mat)
    bpy.context.scene.objects.active = bpy.data.objects["Text.001"]
    bpy.context.object.active_material.diffuse_color = (0.0, 0.0, 0.0)
    bpy.data.objects["Text.001"].data.align_x = "CENTER"
    bpy.data.objects["Text.001"].data.body = "GT"
    bpy.data.objects["Text.001"].data.size = 0.8

    bpy.ops.object.text_add(location=(2.0, 0.0, 4.0))
    bpy.data.objects["Text.002"].rotation_euler = (1.623156, 0.0, 0.0)
    mat = bpy.data.materials.new(name="Text.002")
    bpy.data.objects["Text.002"].data.materials.append(mat)
    bpy.context.scene.objects.active = bpy.data.objects["Text.002"]
    bpy.context.object.active_material.diffuse_color = (0.0, 0.0, 0.0)
    bpy.data.objects["Text.002"].data.align_x = "CENTER"
    bpy.data.objects["Text.002"].data.body = "Ours"
    bpy.data.objects["Text.002"].data.size = 0.8

    bpy.ops.object.text_add(location=(6.0, 0.0, 4.0))
    bpy.data.objects["Text.003"].rotation_euler = (1.570796, 0.0, 0.0)
    mat = bpy.data.materials.new(name="Text.003")
    bpy.data.objects["Text.003"].data.materials.append(mat)
    bpy.context.scene.objects.active = bpy.data.objects["Text.003"]
    bpy.context.object.active_material.diffuse_color = (0.0, 0.0, 0.0)
    bpy.data.objects["Text.003"].data.align_x = "CENTER"
    bpy.data.objects["Text.003"].data.body = "Baseline"
    bpy.data.objects["Text.003"].data.size = 0.8


    constraint1 = bpy.data.objects["Camera"].constraints.new('COPY_LOCATION')
    constraint2 = bpy.data.objects["Camera"].constraints.new('COPY_LOCATION')
    constraint3 = bpy.data.objects["Text"].constraints.new('COPY_LOCATION')
    constraint4 = bpy.data.objects["Text.001"].constraints.new('COPY_LOCATION')
    constraint5 = bpy.data.objects["Text.002"].constraints.new('COPY_LOCATION')
    constraint6 = bpy.data.objects["Text.003"].constraints.new('COPY_LOCATION')


    constraint1.target = bpy.data.objects[f.split(".")[0]+"_inp"]
    constraint2.target = bpy.data.objects[f.split(".")[0]+"_cpy"]
    constraint3.target = bpy.data.objects[f.split(".")[0]+"_inp"]
    constraint4.target = bpy.data.objects[f.split(".")[0]+"_gt"]
    constraint5.target = bpy.data.objects[f.split(".")[0]]
    constraint6.target = bpy.data.objects[f.split(".")[0]+"_cpy"]

    inp_bones = bpy.data.objects[f.split(".")[0]+"_inp"].pose.bones.keys()
    gt_bones = bpy.data.objects[f.split(".")[0]+"_gt"].pose.bones.keys()
    ours_bones = bpy.data.objects[f.split(".")[0]].pose.bones.keys()
    cpy_bones = bpy.data.objects[f.split(".")[0]+"_cpy"].pose.bones.keys()

    for bone in inp_bones:
      if "Hip" in bone:
        constraint1.subtarget = bone
        constraint3.subtarget = bone
        break

    for bone in gt_bones:
      if "Hip" in bone:
        constraint4.subtarget = bone
        break

    for bone in ours_bones:
      if "Hip" in bone:
        constraint5.subtarget = bone
        break

    for bone in cpy_bones:
      if "Hip" in bone:
        constraint2.subtarget = bone
        constraint6.subtarget = bone
        break

#    constraint1.use_z = False
#    constraint2.use_z = False
#    constraint1.use_offset = True
#    constraint2.use_offset = True

    constraint1.use_x = True
    constraint1.use_y = False
    constraint1.use_z = True
    constraint1.invert_z = True
    constraint1.use_offset = True
    constraint1.target_space = "LOCAL"
    constraint1.owner_space = "LOCAL"

    constraint2.use_x = True
    constraint2.use_y = False
    constraint2.use_z = True
    constraint2.invert_z = True
    constraint2.use_offset = True
    constraint2.target_space = "LOCAL"
    constraint2.owner_space = "LOCAL"

    constraint3.use_x = True
    constraint3.use_y = True
    constraint3.use_z = False
    constraint3.use_offset = False

    constraint4.use_x = True
    constraint4.use_y = True
    constraint4.use_z = False
    constraint4.use_offset = False

    constraint5.use_x = True
    constraint5.use_y = True
    constraint5.use_z = False
    constraint5.use_offset = False

    constraint6.use_x = True
    constraint6.use_y = True
    constraint6.use_z = False
    constraint6.use_offset = False

    bpy.data.scenes["Scene"].render.filepath = (
        os.getcwd()+"/results/videos/"+d+"/"+f.split(".")[0]+".avi"
    )
    bpy.data.scenes["Scene"].render.image_settings.file_format = "AVI_JPEG"
    bpy.context.scene.camera = bpy.data.objects['Camera']
    bpy.ops.render.render(animation=True)
#    bpy.ops.wm.save_as_mainfile(filepath=dump_path+"/"+f[:-8]+".blend")
    print(data_path+d+"/"+f+" processed.")

