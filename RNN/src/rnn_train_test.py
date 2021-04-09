from preprocessing import preprocess
import sys 

def main():
    file_map = preprocess()
    print(file_map)

if __name__ == "__main__":
    print(f'Args: {sys.argv}')
    (program, data, apnea_type, timesteps) = sys.argv
    main()