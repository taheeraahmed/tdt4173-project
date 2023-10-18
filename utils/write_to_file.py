import time

def write_to_file(logged_data, start_time, filename = "logging.txt"):
    end_time = time.time()  # <- End the timer
    elapsed_time = end_time - start_time  # <- Calculate elapsed time
    with open(filename, 'a') as file:
        file.write('-------------------------------\n')

        file.write('Name: ')
        file.write(str(logged_data['name']) + '\n')

        file.write('Run name: ')
        file.write(str(logged_data['run_name']) + '\n')

        file.write('Elapsed time: ')
        file.write(str(elapsed_time) + ' seconds \n')

        file.write('Metrics \n')
        file.write(str(logged_data['metrics']) + '\n')

        best_model = logged_data.get('best_model')
        if best_model:
            file.write('Best model \n')
            file.write(str(best_model) + '\n')

        best_params = logged_data.get('best_params')
        if best_params:
            file.write('Best params \n')
            file.write(str(best_params) + '\n')