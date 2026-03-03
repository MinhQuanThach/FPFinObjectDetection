from types import SimpleNamespace

def get_arguments():
    args = SimpleNamespace()
    args.batch_size = 4
    args.iter_size = 1
    args.num_workers = 2
    args.data_dir = 'data'
    args.img_size = 640
    args.num_classes = 8  # Adjust based on your dataset
    args.num_steps = 1000
    args.random_seed = 1234
    args.snapshot_dir = '/content/drive/MyDrive/FIOD_dataset/snapshots'
    args.set = 'train'
    args.lambda_det = 0.1 # Detection loss weight
    args.lambda_fsm = 1000  # Fog style matching weight
    args.lambda_con = 1  # Consistency loss weight
    args.file_name = 'yolov8s_fpf'
    args.gpu = 0
    return args