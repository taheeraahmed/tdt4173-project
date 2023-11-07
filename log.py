import datetime

def write_to_file(mse: str, other=""):
    filename = "log_model.txt"
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with open(filename, 'a') as file:
        file.write('-------------------------------\n')
        file.write('current_time: ' +str(current_time) + '\n')
        file.write('mse: ', str(mse) + '\n')
        if other != "":
            file.write('other: ', str(other) + '\n')
        file.write('-------------------------------\n')