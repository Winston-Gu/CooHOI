import open3d as o3d
import numpy as np

def create_box_with_translation(size, translation, rotation=None):
    # 创建长方体
    box = o3d.geometry.TriangleMesh.create_box(width=size[0], height=size[1], depth=size[2])
    # 将长方体中心移动到指定位置
    box.translate(translation - np.array(box.get_center()))
    # 可选：将长方体旋转到指定角度
    if rotation is not None:
        R = o3d.geometry.get_rotation_matrix_from_xyz(rotation)
        box.rotate(R, center = box.get_center())
    return box

def create_cross_shape_complex():
    # 创建中心长方体
    center_box = create_box_with_translation(size=[0.5, 0.5, 0.5], translation=np.array([0, 0, 0]), rotation=np.array([0, 0, 0]))

    # 创建周围的四个长方体
    boxes = [center_box]
    translations = [
        np.array([0.625, 0, 0]),  # 右侧
        np.array([-0.625, 0, 0]), # 左侧
        np.array([0, 0.625, 0]),  # 上方
        np.array([0, -0.625, 0])  # 下方
    ]
    
    rotations = [
        np.array([0, 0, 0]),  # 右侧
        np.array([0, 0, 0]),  # 左侧
        np.array([0, 0, np.pi / 2]), # 上方
        np.array([0, 0, np.pi / 2])  # 下方
    ]

    for t, r in zip(translations, rotations):
        boxes.append(create_box_with_translation(size=[0.75, 0.5, 0.5], translation=t, rotation=r))

    # 合并所有长方体为一个网格
    cross_shape = o3d.geometry.TriangleMesh()
    for box in boxes:
        cross_shape += box

    # 简化网格：去除重复的顶点和面
    cross_shape = cross_shape.remove_duplicated_vertices()
    cross_shape = cross_shape.remove_duplicated_triangles()
    cross_shape = cross_shape.remove_degenerate_triangles()

    # 可选：重新计算法线
    # cross_shape.compute_vertex_normals()

    return cross_shape


def create_simple_box():
    center_box = create_box_with_translation(size=[2, 0.5, 0.5], translation=np.array([0, 0, 0]), rotation=np.array([0, 0, 0]))
    cross_shape = o3d.geometry.TriangleMesh()
    cross_shape += center_box
    return cross_shape

if __name__ == '__main__':

    # 创建并处理复杂的十字形长方体
    # cross_shape_complex = create_cross_shape_complex()
    simple_box = create_simple_box()

    # 保存为OBJ文件
    o3d.io.write_triangle_mesh("train/box/long_box.obj", simple_box, write_triangle_uvs=True)