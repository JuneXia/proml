"""
将voc格式的数据按下面这样的格式存到 txt 文件中：
train_val_flag       图片路径                                                            bbox1, bbox1_class   bbox2, bbox2_class
0       /home/xiajun/res/VOC/VOCtrainval2007/VOCdevkit/VOC2007/JPEGImages/004976.jpg   311,108,476,221,14   128,193,500,338,10
"""

import os
import random
import xml.etree.ElementTree as ET

sets = [('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def convert_annotation(voc_path, voc_year, xml_name, xml_file, file_handle):
    in_file = open(xml_file)
    tree = ET.parse(in_file)
    root = tree.getroot()
    if root.find('object') == None:
        return

    image_path = os.path.join(voc_path, 'VOC%s/JPEGImages' % (voc_year), '%s.jpg' % (xml_name))
    file_handle.write(image_path)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
             int(xmlbox.find('ymax').text))
        file_handle.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

    file_handle.write('\n')


def xml_parse(voc_path, voc_year=2007, val_ratio=0.1, save_path='./', train_test_flag='train'):
    assert val_ratio >= 0
    assert train_test_flag in ['train', 'test']

    annotations = os.path.join(voc_path, 'VOC%s'%(voc_year), 'Annotations')
    temp_xml = os.listdir(annotations)
    random.shuffle(temp_xml)
    total_xml = []
    for xml in temp_xml:
        if xml.endswith(".xml"):
            total_xml.append(xml)

    def _traverse(xml_list, file_save_handle):
        for name in xml_list:
            xml_name = name.split('.')[0]
            xml_file = os.path.join(annotations, name)
            convert_annotation(voc_path, voc_year, xml_name, xml_file, file_save_handle)

    val_size = int(val_ratio * len(total_xml))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if train_test_flag == 'train':
        if val_ratio > 0:
            val_file = os.path.join(save_path, 'VOC%s_val.txt' % (voc_year))
            xml_list = total_xml[0:val_size]
            with open(val_file, 'w') as f:
                _traverse(xml_list, f)

            # 剩下的用作训练
            total_xml = total_xml[val_size:]

        train_file = os.path.join(save_path, 'VOC%s_train.txt' % (voc_year))
        with open(train_file, 'w') as f:
            _traverse(total_xml, f)
    else:
        test_file = os.path.join(save_path, 'VOC%s_test.txt' % (voc_year))
        with open(test_file, 'w') as f:
            _traverse(total_xml, f)


if __name__ == '__main__1':  # ubuntu
    xml_path = "/home/xiajun/res/VOC/VOCtrainval2007/VOCdevkit/"
    save_path = "/home/xiajun/dev/proml/object_detection/fasterrcnn2/VOCdevkit"
    xml_parse(xml_path, val_ratio=0.1, save_path=save_path, train_test_flag='train')
