import os

def get_raw_data_paths(group, subject, conditions, areas):
    if type(conditions) is not list:
        conditions = [conditions]
    
    if type(areas) is not list:
        areas = [areas]

    subject_path = os.path.join('./data', group, subject)
    raw_data_paths = {}
    for condition in conditions:
        files = [os.path.join(subject_path, condition, f) for f in os.listdir(os.path.join(subject_path, condition)) if (os.path.isfile(os.path.join(subject_path, condition, f)) and f != 'Raw-Info.mat')]
        raw_data_paths[condition] = {}
        for area in areas:
            raw_data_paths[condition][area] = [file for file in files if file.split('\\')[-1].find('_' + area + '_') != -1 ]

    return raw_data_paths

def get_subject_information(group, subject, condition=None, area=None, channel=None):
    subject_information = group + '_' + subject

    if condition is not None:
        subject_information = subject_information + '_' + condition.split('_')[0]

    if area is not None:
        subject_information = subject_information + '_' + area

    if channel is not None:
        subject_information = subject_information + '_Ch' + str(channel)

    return subject_information