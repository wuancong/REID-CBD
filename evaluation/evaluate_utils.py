import os


def prepare_test_data(dataset_name, txt_path):
    root_dir = os.path.dirname(txt_path)
    if dataset_name == 'MSMT17':
        root_dir = os.path.join(root_dir, 'test')
    parse_fun = {'REID-CBD': parse_reid_cbd, 'DukeMTMC': parse_dukemtmc, 'MSMT17': parse_msmt17}
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        img_path, pid, cid = parse_fun[dataset_name](line)
        data.append((os.path.join(root_dir, img_path), pid, cid, 0))
    return data


def parse_reid_cbd(line):
    mapping = {"2A": 1, "2B": 2, "3A": 3, "3B": 4, "4A": 5, "4B": 6}
    img_path, pid = line[:-1].split(' ')
    pid = int(pid)
    base = os.path.basename(img_path)
    assert pid == int(base.split('_')[0])
    camid = base.split('_')[2][1]
    site = base.split('_')[1][1]
    cid = mapping[site + camid]
    return img_path, pid, cid


def parse_dukemtmc(line):
    # file name: 0001_c2_f0046182.jpg
    img_path, pid = line[:-1].split(' ')
    pid = int(pid)
    base = os.path.basename(img_path)
    assert pid == int(base.split('_')[0])
    cid = int(base.split('_')[1][1])
    return img_path, pid, cid


def parse_msmt17(line):
    # file name: 0000_000_01_0303morning_0008_0.jpg
    img_path, pid = line[:-1].split(' ')
    pid = int(pid)
    base = os.path.basename(img_path)
    assert pid == int(base.split('_')[0])
    cid = int(base.split('_')[2])
    return img_path, pid, cid
