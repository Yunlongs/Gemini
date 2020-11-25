
## choose what binary you want to generate the dataset
version = ["openssl-101a","openssl-101f"]
arch = ["arm","x86","mips"]
compiler = ["gcc"]
optimizer = ["O0","O1","O2","O3"]
dir_name  = "../dataset/extracted-acfg/"

## VulSeeker
vulseeker_rawdata_dir = "dataset/openssl/"
vulseeker_dataset_dir = "dataset/vulseeker/"
vulseeker_feature_size = 8
vulseeker_model_save_path = "output/vulseeker/vulseeker_model_weight"
vulseeker_figure_save_path = "output/vulseeker/"

## Gemini
Gemini_rawdata_dir = "../data/extracted-acfg"
Gemini_dataset_dir = "dataset/Gemini/"
Gemini_feature_size = 9 # （max_constant_1,max_constant_2,num of strings,....）
Gemini_model_save_path = "output/Gemini/Experiment_2/model_weight"
Gemini_figure_save_path = "output/Gemini/"

### some details about dataset generation
max_nodes = 500
min_nodes_threshold = 0
Buffer_Size = 1000
mini_batch = 10



### some params about training the network
learning_rate  = 0.0001
epochs  = 100
step_per_epoch = 15000
valid_step_pre_epoch = 3000
test_step_pre_epoch = 3000
T = 5
embedding_size = 64
embedding_depth = 2

