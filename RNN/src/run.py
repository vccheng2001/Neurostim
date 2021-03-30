import preprocessing
import rnn_train_only
import rnn_test_only

# params: <data> <apnea_type>, <timesteps>, <batch_size>, <threshold>
(program, data, timesteps, epochs, batch_size, threshold) = sys.argv
results = {} 
for i in range(1,9):
    apnea_type = f"\n\nosahs_excerpt{i}\n"
    print(f"**********************{apnea_type}*************************")
    print("Preprocessing\n")
    os.system(f"python3 preprocessing.py")#{data} {apnea_type} {timesteps}")
    print(f"# Positives, # Negatives: {preprocessing.num_files_per_label.values()}")
    # print(f"# Negatives: {preprocessing.num_files_per_label["negatives"]}\n")

    print("Training....\n")
    os.system(f"python3 rnn_train_only.py {data} {apnea_type} {timesteps} {epochs} {batch_size}")
    print("Testing....\n")
    os.system(f"python3 rnn_test_only.py {data} {apnea_type} {timesteps} {batch_size} {threshold}")
