import argparse
import json
import xml.etree.ElementTree as ET
import wget
from wget import bar_thermometer
import os
import configparser
import logging
import tarfile
from shutil import move, rmtree
import zipfile

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger("DatasetPrepare")


def create_path(path: str):
    now_path = ''
    if path.startswith('/'):
        now_path = '/'
    for path_i in path.split('/'):
        now_path += path_i + '/'
        if not os.path.exists(now_path):
            os.mkdir(now_path)


def load_config(config_path: str):
    parser = configparser.ConfigParser()
    parser.read(config_path)
    return parser


def download(url: str):
    target_filename = url.split('/')[-1]
    target_path = 'tmp/' + target_filename
    if not os.path.exists(target_path):
        log.info(f'Download {url} to {target_path}')
        wget.download(url, target_path, bar=bar_thermometer)
    else:
        log.warning(f'File {url} is already exists on {target_path}')

    return target_path


def extract_only_path(tar: tarfile, path: str, destination: str):
    members = [tarinfo for tarinfo in tar.getmembers()
                      if tarinfo.name.startswith(path)]
    log.info(f"Extracting {len(members)} files from {path}")
    tar.extractall(destination, members=members)


def unpack_tar_dataset(filename: str, dest_dir: str, voc_version: str = None):
    with tarfile.open(filename) as tar:
        if voc_version is not None:
            dest_path = os.path.join(dest_dir, voc_version)
            create_path(dest_path)
            extract_only_path(tar, f'VOCdevkit/{voc_version}/Annotations/', dest_path)
            extract_only_path(tar, f'VOCdevkit/{voc_version}/JPEGImages/', dest_path)
            voc_to_root_folder(f'{dest_dir}/{voc_version}/VOCdevkit/{voc_version}', dest_dir)
            rmtree(f'{dest_dir}/{voc_version}')


def unpack_coco_dataset(filename: str, dest_dir: str):
    with zipfile.ZipFile(filename, 'r') as archive:
        dest_path = os.path.join(os.getcwd(), dest_dir)
        create_path(dest_path)
        archive.extractall(dest_path)


def voc_to_root_folder(voc_path: str, voc_root_folder: str):
    print(voc_path)
    directories = [os.path.join(voc_path, path) for path in os.listdir(voc_path)]
    for directory in directories:
        move(directory, os.path.join(voc_root_folder, directory.split('/')[-1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Tool for load datasets (VOC, COCO)")
    parser.add_argument('-v', '--voc', help='Load VOC dataset', dest='voc_dn', action='store_true', default=False)
    parser.add_argument('-c', '--coco', help='Load COCO dataset', dest='coco_dn', action='store_true', default=False)
    parser.add_argument('-t', '--transform', help='Transform dataset to one type', dest='transform', action='store_true', default=False)
    parser.add_argument('-f', '--cfg', dest='cfg_file', required=True)
    args = parser.parse_args()

    config = load_config(args.cfg_file)
    create_path('tmp')

    if args.voc_dn:
        create_path('voc')
        files = []
        for url in config['VOC']['data'][1:-1].split(' '):
            url = url.strip('\'')
            files.append(download(url))
        print(files)
        unpack_tar_dataset(files[0], 'voc', 'VOC2012')

    if args.coco_dn:
        create_path('coco')
        files = []
        for url in config['COCO']['data'][1:-1].split(' '):
            url = url.strip('\'')
            files.append(download(url))
        for url in config['COCO']['image'][1:-1].split(' '):
            url = url.strip('\'')
            files.append(download(url))
        print(files)
        for file in files:
            unpack_coco_dataset(file, 'coco')

    if args.transform:
        all_annotation_ = {}
        create_path('data/images')
        categories = {}
        image_annotation = {}
        if os.path.exists('coco'):
            for image_parts in os.listdir('coco'):
                if image_parts != "annotations":
                    annotation_ = json.load(open(f'coco/annotations/instances_{image_parts}.json'))
                    file_st_path = os.path.join('coco', image_parts)
                    for el in annotation_['categories']:
                        categories[int(el['id'])] = el['name']
                    images = {
                        el['id']: os.path.join("dataset", "coco", image_parts, el["file_name"]) for el in annotation_['images']
                    }
                    for el in annotation_["annotations"]:
                        if el["image_id"] in images:
                            if images[el["image_id"]] not in image_annotation:
                                image_annotation[images[el["image_id"]]] = []
                            image_annotation[images[el["image_id"]]].append({
                                "box": [int(float(el["bbox"][0])),
                                        int(float(el["bbox"][1])),
                                        int(float(el["bbox"][2])) + int(float(el["bbox"][0])),
                                        int(float(el["bbox"][1])) + int(float(el["bbox"][3]))],
                                "classname": categories[el["category_id"]]
                            })
        if os.path.exists('voc'):
            for image_parts in os.listdir('voc'):
                if image_parts == "Annotations":
                    for element in os.listdir('voc/Annotations'):
                        tree = ET.parse(os.path.join('voc/Annotations', element))
                        root = tree.getroot()
                        image_annotation[os.path.join(f'dataset/voc/JPEGImages/{element.replace("xml", "jpg")}')] = []
                        for object in root.findall('object'):
                            object_name = object.find('name').text
                            object_box = [int(float(object.find('bndbox').find(key).text)) for key in ['xmin', 'ymin', 'xmax', 'ymax']]
                            image_annotation[os.path.join(f'dataset/voc/JPEGImages/{element.replace("xml", "jpg")}')].append({
                                "box": object_box,
                                "classname": object_name
                            })
        with open('all_dataset.json', 'w+') as f:
            f.write(json.dumps({
                "images": image_annotation,
                "classnames": categories
            }))
