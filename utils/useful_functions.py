import os


# Print stats to file
def print_stats(stats_file, text, print_to_console=True):
    with open(stats_file, 'a') as f:
        if isinstance(text, list):
            for t in text:
                f.write(t + "\n")
                if print_to_console:
                    print(t)
        else:
            f.write(text + "\n")
            if print_to_console:
                print(text)
    f.close()


# Function to create a new folder if does not exists
def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)