import os

cpu_data_dict = {}
gnne_no_ptq_data_dict = {}
gnne_ptq_data_dict = {}
gt_data_dict = {}


def calculate(gt_res, cpu_res, gnne_no_ptq_res, gnne_ptq_res, case_name):
    with open(gt_res, 'r') as gt_f, open(cpu_res, 'r') as cpu_f, open(gnne_no_ptq_res, 'r') as gnne_no_ptq_f, open(gnne_ptq_res, 'r') as gnne_ptq_f:
        for line in cpu_f.readlines():
            cpu_data_dict[line.strip('\n').split(' ')[0]] = line.strip('\n').split(' ')[1]
        for line in gnne_no_ptq_f.readlines():
            gnne_no_ptq_data_dict[line.strip('\n').split(' ')[0]] = line.strip('\n').split(' ')[1]
        for line in gnne_ptq_f.readlines():
            gnne_ptq_data_dict[line.strip('\n').split(' ')[0]] = line.strip('\n').split(' ')[1]
        for line in gt_f.readlines():
            print(line)
            gt_data_dict[line.strip('\n').split(' ')[0]] = line.strip('\n').split(' ')[1]

    true_cpu = 0
    cpu_map = 0
    num_classes_flag = 1
    if case_name.split('_')[-1] in ['vgg16']:
        num_classes_flag = 0
    for key, value in cpu_data_dict.items():
        if (int(value) == int(gt_data_dict[key]) + num_classes_flag):
            true_cpu = true_cpu + 1
    cpu_map = true_cpu / len(cpu_data_dict)
    print('cpu_map is: ', cpu_map)

    true_gnne_no_ptq = 0
    gnne_no_ptq_map = 0
    for key, value in gnne_no_ptq_data_dict.items():
        if (int(value) == int(gt_data_dict[key]) + num_classes_flag):
            true_gnne_no_ptq = true_gnne_no_ptq + 1
    gnne_no_ptq_map = true_gnne_no_ptq / len(gnne_no_ptq_data_dict)
    print('gnne_no_ptq_map is: ', gnne_no_ptq_map)

    true_gnne_ptq = 0
    gnne_ptq_map = 0
    for key, value in gnne_ptq_data_dict.items():
        if (int(value) == int(gt_data_dict[key]) + num_classes_flag):
            true_gnne_ptq = true_gnne_ptq + 1
    gnne_ptq_map = true_gnne_ptq / len(gnne_ptq_data_dict)
    print('gnne_ptq_map is: ', gnne_ptq_map)
