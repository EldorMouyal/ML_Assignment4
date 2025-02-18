import subprocess




def main():
    # Step 1: Data Preparation
    print("Running data preparation...")
    subprocess.run(["python", "data_prep.py"])

    # Step 2: Train VGG19
    print("Training VGG19 model...")
    subprocess.run(["python", "train_vgg19.py"])

    # Step 3: Train YOLOv5
    print("Training YOLOv5 model...")
    subprocess.run(["python", "train_yolov5.py"])

    print("All processes completed.")

if __name__ == '__main__':
    main()
