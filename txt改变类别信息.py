import os
import glob

# 定义要修改的class_id映射规则
class_id_mapping = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 4,
    6: 4,
    7: 4,
}

def modify_class_id(file_path, mapping):
    # 读取文件内容
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 修改class_id
    modified_lines = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        if class_id in mapping:
            parts[0] = str(mapping[class_id])
        modified_lines.append(' '.join(parts))

    # 写回修改后的内容
    with open(file_path, 'w') as file:
        file.write('\n'.join(modified_lines) + '\n')

def main():
    # 指定标签文件所在的文件夹路径
    folder_path = r'C:\Users\LazyShark\Desktop\RZB\labels1'  # 请替换为你的文件夹路径

    # 查找文件夹中的所有.txt文件
    label_files = glob.glob(os.path.join(folder_path, '*.txt'))

    # 批量修改每个文件中的class_id
    for label_file in label_files:
        modify_class_id(label_file, class_id_mapping)
        print(f'Modified {label_file}')

if __name__ == '__main__':
    main()
