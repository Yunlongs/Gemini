
## choose what binary you want to generate the dataset
version = ["openssl-101a","openssl-101f"]
arch = ["arm","x86","mips"]
compiler = ["gcc","clang"]
optimizer = ["O0","O1","O2","O3"]
dir_name  = "../dataset/extracted-acfg/"

## VulSeeker
vulseeker_rawdata_dir = "dataset/openssl/"
vulseeker_dataset_dir = "dataset/vulseeker/"
vulseeker_feature_size = 8
vulseeker_model_save_path = "output/vulseeker/vulseeker_model_weight"

### some details about dataset generation
max_nodes = 500
min_nodes_threshold = 3
Buffer_Size = 1000
mini_batch = 10



### some params about training the network
learning_rate  = 0.0001
epochs  = 100
step_per_epoch = 30000
valid_step_pre_epoch = 3800
test_step_pre_epoch = 38000
T = 5
embedding_size = 64
embedding_depth = 2

