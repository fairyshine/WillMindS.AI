import os
import shutil
import subprocess

from .. import logger

def backup_files(src_dir, exclude_list, output_dir):
    logger.info("====== Backuping Files ======")
    logger.info("From: {}".format(src_dir))
    logger.info("To: {}".format(output_dir))
    logger.info("Exclude: {}".format(str(exclude_list)))

    os.makedirs(output_dir, exist_ok=True)
        
    # 遍历源目录
    for root, dirs, files in os.walk(src_dir):
        # 检查当前目录是否在排除列表中
        if any(os.path.commonpath([root, excl]) == excl for excl in exclude_list):
            continue
        
        for file in files:
            # 构建源文件路径
            src_file_path = os.path.join(root, file)

            if src_file_path not in exclude_list:
                # 构建目标文件路径
                rel_path = os.path.relpath(src_file_path, src_dir)
                dst_file_path = os.path.join(output_dir, rel_path)

                # 创建目标目录
                os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)

                # 复制文件
                shutil.copy2(src_file_path, dst_file_path)
                logger.info(f'Copied {src_file_path} to {dst_file_path}')
    logger.info("=============================")

def count_file_lines(file_path):
    # 使用subprocess.run执行wc -l命令
    result = subprocess.run(['wc', '-l', file_path], capture_output=True, text=True)
    # 提取行数
    line_count = result.stdout.split()[0]
    return int(line_count)+1

if __name__ == "__main__":
    src_dir = 'src/'  # 源文件夹路径
    exclude_list = ['exclude_folder1', 'exclude_folder2']  # 要排除的文件夹路径列表
    output_dir = 'backup/'  # 备份文件夹路径
    backup_files(src_dir, exclude_list, output_dir)
