import os
import random


def CreateSplit(src_path, dst_path, train_class_list, val_class_list, test_class_list):
    video_count = 0
    count = 1
    train_txt_path = os.path.join(dst_path, "trainlist01.txt")
    val_txt_path = os.path.join(dst_path, "vallist01.txt")
    test_txt_path = os.path.join(dst_path, "testlist01.txt")

    while os.path.exists(train_txt_path) or os.path.exists(val_txt_path) or os.path.exists(test_txt_path):
        count += 1
        train_txt_path = os.path.join(dst_path, f"trainlist{count:02}.txt")
        val_txt_path = os.path.join(dst_path, f"vallist{count:02}.txt")
        test_txt_path = os.path.join(dst_path, f"testlist{count:02}.txt")

    with open(train_txt_path, "w") as file:
        for c in train_class_list:
            c_path = os.path.join(src_path, c)
            for video in os.listdir(c_path):
                file.write(f"{c}/{video}\n")
                video_count += 1

    with open(val_txt_path, "w") as file:
        for c in val_class_list:
            c_path = os.path.join(src_path, c)
            for video in os.listdir(c_path):
                file.write(f"{c}/{video}\n")
                video_count += 1

    with open(test_txt_path, "w") as file:
        for c in test_class_list:
            c_path = os.path.join(src_path, c)
            for video in os.listdir(c_path):
                file.write(f"{c}/{video}\n")
                video_count += 1
    print(f"Train Split created: {train_txt_path}")
    print(f"Validation Split created: {val_txt_path}")
    print(f"Test Split created: {test_txt_path}")
    print(f"Video Count: {video_count}")

def createRandomSplit(src_path, number_Train, number_Val, number_Test):
    class_list = []
    for folder in os.listdir(src_path):
        class_list.append(folder)
    
    if len(class_list) < (number_Train + number_Val + number_Test):
        print("Not enough classes...")
        return
    
    train_class_list = random.sample(class_list, number_Train)
    class_list = [elem for elem in class_list if elem not in train_class_list]
    val_class_list = random.sample(class_list, number_Val)
    class_list = [elem for elem in class_list if elem not in val_class_list]
    test_class_list = random.sample(class_list, number_Test)
    class_list = [elem for elem in class_list if elem not in test_class_list]

def createRandomSplit_fixedTestClass(src_path, number_Train, number_Val, Test_Class_Prefix):
    class_list = []
    for folder in os.listdir(src_path):
        class_list.append(folder)
    
    test_class_list = [elem for elem in class_list if elem.startswith(Test_Class_Prefix + "_")]
    class_list = [elem for elem in class_list if elem not in test_class_list]

    if len(class_list) < (number_Train + number_Val):
        print("Not enough classes...")
        return

    train_class_list = random.sample(class_list, number_Train)
    class_list = [elem for elem in class_list if elem not in train_class_list]
    val_class_list = random.sample(class_list, number_Val)
    class_list = [elem for elem in class_list if elem not in val_class_list]
    
    return train_class_list, val_class_list, test_class_list


if __name__ == "__main__":
    src_directory = "C:\\Users\\roibl\\OneDrive - stud.uni-stuttgart.de\\PythonCode\\Forschungsarbeit\\trx\\video_datasets\\data\\surgicalphasev1_Xx256"
    dst_directory = "C:\\Users\\roibl\\OneDrive - stud.uni-stuttgart.de\\PythonCode\\Forschungsarbeit\\trx\\video_datasets\\splits\\surgicalphasev1TrainTestlist"
    train_list, val_list, test_list = createRandomSplit_fixedTestClass(src_directory, 57, 0, "C80")
    CreateSplit(src_directory, dst_directory, train_list, val_list, test_list)