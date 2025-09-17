import os
import shutil
import random

dst_TrainData = "../dataset/train/Data"
dst_TrainGround = "../dataset/train/Ground"
dst_ValData = "../dataset/val/Data"
dst_ValGround = "../dataset/val/Ground"
dst_TestData = "../dataset/test/Data"
dst_TestGround = "../dataset/test/Ground"


def collect_T1_name(patient_dir):
    ground_paths = list()
    inphase_paths = list()

    t1_datadir = os.path.join(patient_dir, "T1DUAL")
    ground_dir = os.path.join(t1_datadir, "Ground")
    # 检查Ground目录是否存在
    if not os.path.exists(ground_dir):
        return ground_paths, inphase_paths
    ground_names = os.listdir(ground_dir)
    nums_ground = len(ground_names)
    # 拼接Ground文件夹的文件，存入到ground_paths列表中
    for i in range(nums_ground):
        ground_paths.append(os.path.join(ground_dir, ground_names[i]))

    inphase_dir = os.path.join(t1_datadir, "DICOM_anon", "InPhase")
    # 检查InPhase目录是否存在
    if not os.path.exists(inphase_dir):
        return ground_paths, inphase_paths
    inphase_names = os.listdir(inphase_dir)
    nums_inphase = len(inphase_names)

    # 拼接inphase文件夹的文件，存入到inphase_paths列表中
    for i in range(nums_inphase):
        inphase_paths.append(os.path.join(inphase_dir, inphase_names[i]))

    return ground_paths, inphase_paths


if __name__ == '__main__':
    dataset_dir = os.path.join("../CHAOS_Train_Sets", "Train_Sets", "MR")
    # 修改比例为6:2:2
    train_pct = 0.7
    val_pct = 0.2
    test_pct = 0.1

    # 确保目标目录存在
    os.makedirs(dst_TrainData, exist_ok=True)
    os.makedirs(dst_TrainGround, exist_ok=True)
    os.makedirs(dst_ValData, exist_ok=True)
    os.makedirs(dst_ValGround, exist_ok=True)
    os.makedirs(dst_TestData, exist_ok=True)
    os.makedirs(dst_TestGround, exist_ok=True)

    for root, dirs, files in os.walk(dataset_dir):
        random.shuffle(dirs)
        dir_count = len(dirs)
        train_point = int(dir_count * train_pct)
        val_point = int(dir_count * (train_pct + val_pct))
        i = 0
        for sub_dir in dirs:  # sub_dir代表病人编号
            patient_dir = os.path.join(root, sub_dir)
            # 检查病人目录是否存在T1DUAL子目录
            t1_dir = os.path.join(patient_dir, "T1DUAL")
            if not os.path.exists(t1_dir):
                continue
                
            ground_paths, inphase_paths = collect_T1_name(patient_dir)
            
            # 确保有数据需要复制
            if len(ground_paths) == 0 or len(inphase_paths) == 0:
                print(f"Warning: No data found for patient {sub_dir}")
                continue
                
            if i < train_point:
                for num in range(len(ground_paths)):
                    dst_groundpath = os.path.join(dst_TrainGround, "T1_Patient%s_No%d.png" % (sub_dir, num))
                    shutil.copy(ground_paths[num], dst_groundpath)

                for num in range(len(inphase_paths)):
                    dst_inphasepath = os.path.join(dst_TrainData, "T1_Patient%s_No%d.dcm" % (sub_dir, num))
                    shutil.copy(inphase_paths[num], dst_inphasepath)

                i += 1
            elif i < val_point:
                for num in range(len(ground_paths)):
                    dst_groundpath = os.path.join(dst_ValGround, "T1_Patient%s_No%d.png" % (sub_dir, num))
                    shutil.copy(ground_paths[num], dst_groundpath)

                for num in range(len(inphase_paths)):
                    dst_inphasepath = os.path.join(dst_ValData, "T1_Patient%s_No%d.dcm" % (sub_dir, num))
                    shutil.copy(inphase_paths[num], dst_inphasepath)

                i += 1
            else:
                for num in range(len(ground_paths)):
                    dst_groundpath = os.path.join(dst_TestGround, "T1_Patient%s_No%d.png" % (sub_dir, num))
                    shutil.copy(ground_paths[num], dst_groundpath)

                for num in range(len(inphase_paths)):
                    dst_inphasepath = os.path.join(dst_TestData, "T1_Patient%s_No%d.dcm" % (sub_dir, num))
                    shutil.copy(inphase_paths[num], dst_inphasepath)

                i += 1
        # 跳出根目录的遍历，避免递归遍历其他子目录
        break