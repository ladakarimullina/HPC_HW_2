# from matplotlib import pyplot as plt
# import numpy as np
# from argparse import ArgumentParser
# import os

# if __name__ == '__main__':
#     parser = ArgumentParser()
#     parser.add_argument('-p', type=str, required=True, help='Specify .txt file with the profile')
#     path = parser.parse_args().p


#     f_list = []
#     if path.endswith(".txt"):
#       with open(path, "r") as file1:
#           lines = file1.readlines()
#           for i, line in enumerate(lines):
#               line = line.strip()
#               single_list = [float(i) for i in line.split(" ")]
#               if i == 1:
#                   single_list = single_list[1:]
#               f_list.extend(single_list)
    
#     y_max, dy = 5, 1e-3
#     y = np.arange(-y_max, 2 * (y_max + dy), dy)
#     plt.plot(y, f_list, 'k')
#     img_name = path.split('/')[1].split('.txt')[0]
#     plt.savefig('images_experiment/' + img_name + '.png')

from matplotlib import pyplot as plt
import numpy as np
from argparse import ArgumentParser
import os

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-p', type=str, required=True, help='Specify .txt file with the profile')
    path = parser.parse_args().p

    if os.path.isdir(path):
        print(f"Skipping directory: {path}")
        exit(1)

    f_list = []
    with open(path, "r") as file1:
        lines = file1.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            single_list = [float(i) for i in line.split(" ")]
            if i == 1:
                single_list = single_list[1:]
            f_list.extend(single_list)
    
    y_max, dy = 5, 1e-3
    y = np.arange(-y_max, 2 * (y_max + dy), dy)
    plt.plot(y, f_list, 'k')
    
    img_name = os.path.basename(path).split('.txt')[0]
    output_dir = 'images_experiment'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(os.path.join(output_dir, img_name + '.png'))
    plt.close()


 
