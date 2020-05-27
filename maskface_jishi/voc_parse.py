"""
reference: https://blog.csdn.net/weixin_41278720/article/details/84872064
"""
import os
import xml.etree.ElementTree as ET

CLASSES = ('background', 'nomask', 'mask')
LABEL_ENDSWITH = '.xml'


index_map = dict(zip(CLASSES, range(len(CLASSES))))
print(index_map)


debug_endswith_count = 0
debug_diffcult_count = 0
debug_notin_class_count = 0


def validate_label(xmin, ymin, xmax, ymax, width, height):
    """Validate labels."""
    assert 0 <= xmin < width, "xmin must in [0, {}), given {}".format(width, xmin)
    assert 0 <= ymin < height, "ymin must in [0, {}), given {}".format(height, ymin)
    assert xmin < xmax <= width, "xmax must in (xmin, {}], given {}".format(width, xmax)
    assert ymin < ymax <= height, "ymax must in (ymin, {}], given {}".format(height, ymax)


def parse_xml(anno_path):
    global debug_diffcult_count, debug_notin_class_count

    tree = ET.parse(anno_path)
    root = tree.getroot()
    filename = root.find('filename').text
    size = root.find('size')
    width = float(size.find('width').text)
    height = float(size.find('height').text)

    label = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult')
        if difficult is None:
            difficult = obj.find('Difficult')
            if difficult is None:
                debug_diffcult_count += 1
                print("{} not find difficult!".format(anno_path))
                continue
        difficult = int(difficult.text)
        cls_name = obj.find('name').text.strip().lower()
        if cls_name not in CLASSES:
            print("{} not in CLASSES!".format(anno_path))
            debug_notin_class_count += 1
            continue
        cls_id = index_map[cls_name]

        xml_box = obj.find('bndbox')
        xmin = float(xml_box.find('xmin').text)
        ymin = float(xml_box.find('ymin').text)
        xmax = float(xml_box.find('xmax').text)
        ymax = float(xml_box.find('ymax').text)

        try:
            validate_label(xmin, ymin, xmax, ymax, width, height)
        except AssertionError as e:
            text = "Invalid label at {}, {}".format(anno_path, e)
            print(text)
            # raise RuntimeError(text)
            continue
        label.append([xmin, ymin, xmax, ymax, cls_id, difficult])

    return filename, label


def parse_voc(data_dir):
    global debug_endswith_count

    images_dir = os.path.join(data_dir, "data")
    labels_dir = os.path.join(data_dir, "label")

    images_label = []
    label_list = os.listdir(labels_dir)
    for label_name in label_list:
        if not label_name.endswith(LABEL_ENDSWITH):
            print('{} not endswith {}'.format(label_name, LABEL_ENDSWITH))
            debug_endswith_count += 1
            continue
        anno_path = os.path.join(labels_dir, label_name)
        filename, label = parse_xml(anno_path)

        filename = os.path.join(images_dir, filename)
        if not os.path.exists(filename):
            if os.path.exists(filename + '.jpg'):
                filename += '.jpg'
            elif os.path.exists(filename + '.JPG'):
                filename += '.JPG'
            else:
                print('what image for {} ?'.format(filename))

        if len(label) > 0 and os.path.exists(filename):
            images_label.append((filename, label))

    return images_label


if __name__ == '__main__':
    images_path = '/home/xiajun/res/face/maskface/zhihu_akou/data'
    labels_path = '/home/xiajun/res/face/maskface/zhihu_akou/label'

    labels = parse_voc(labels_path)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("debug_endswith_count = {}".format(debug_endswith_count))
    print("debug_diffcult_count = {}".format(debug_diffcult_count))
    print("debug_notin_class_count = {}".format(debug_notin_class_count))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print('end')


