import os
import sys
from scipy import misc
from face_detection import FaceDetector
import tools

fdetector = FaceDetector()


def youtube_align_face_for_one_user(people_path, people_save_path, align_height, align_width, margin):
    align_count = 0
    total_count = 0
    video_folder_list = os.listdir(people_path)
    for video_folder in video_folder_list:
        dst_image_path = os.path.join(people_save_path, video_folder)
        if os.path.exists(dst_image_path) is not True:
            os.makedirs(dst_image_path)

        video_file_list = os.listdir(os.path.join(people_path, video_folder))
        for video_file_name in video_file_list:
            src_image_file = os.path.join(people_path, video_folder, video_file_name)
            dst_image_file = os.path.join(dst_image_path, video_file_name)
            _, aligned_images = fdetector.align(src_image_file, (align_height, align_width), margin)
            if aligned_images is None:
                print('not detect face: ', src_image_file)
                continue
            if len(aligned_images) > 1:
                print('detected more than one people in ', src_image_file)
            for img in aligned_images:
                misc.imsave(dst_image_file, img)
            align_count += 1
        total_count += len(video_file_list)

    return total_count, align_count


def celeb_align_face_for_one_image(src_image_file, dst_image_file, align_height, align_width, margin):
    _, aligned_images = fdetector.align(src_image_file, (align_height, align_width), margin)
    if aligned_images is None:
        # print('not detect face: ', src_image_file)
        return 1, 0
    if len(aligned_images) > 1:
        print('detected more than two people in ', src_image_file)
    for img in aligned_images:
        misc.imsave(dst_image_file, img)

    return 1, 1


def vggface2_align_face_for_one_user(people_path, people_save_path, align_height, align_width, margin):
    if os.path.exists(people_save_path) is not True:
        os.makedirs(people_save_path)

    total_image = 0
    align_count = 0
    image_list = os.listdir(people_path)
    for imfile in image_list:
        src_imfile_path = os.path.join(people_path, imfile)
        dst_imfile_path = os.path.join(people_save_path, imfile)
        _, aligned_images = fdetector.align(src_imfile_path, (align_height, align_width), margin)
        if aligned_images is None:
            # print('not detect face: ', src_imfile_path)
            continue
        if len(aligned_images) > 1:
            print('detected more than one people in ', src_imfile_path)
        for img in aligned_images:
            misc.imsave(dst_imfile_path, img)
        align_count += 1
    total_image += len(image_list)

    return total_image, align_count


def align_batches(dataset_name, src_folder, dst_folder, align_height, align_width, margin):
    if os.path.exists(dst_folder) is not True:
        os.makedirs(dst_folder)

    total_picture = 0
    people = []
    people_list = os.listdir(src_folder)
    for peop in people_list:
        if '.txt' not in peop:
            people.append(peop)
    people_list = people

    for i, people_folder in enumerate(people_list):
        src_people_path = os.path.join(src_folder, people_folder)
        dst_people_path = os.path.join(dst_folder, people_folder)
        if dataset_name == 'youtube':
            image_num, aligned_num = youtube_align_face_for_one_user(src_people_path, dst_people_path, align_height, align_width, margin)
        elif dataset_name == 'celeb':
            image_num, aligned_num = celeb_align_face_for_one_image(src_people_path, dst_people_path, align_height, align_width, margin)
        elif dataset_name == 'vggface2':
            image_num, aligned_num = vggface2_align_face_for_one_user(src_people_path, dst_people_path, align_height, align_width, margin)
        total_picture += image_num

        tools.view_bar('aligning: ', i + 1, len(people_list))


if __name__ == "__main__":
    ''
    sys.argv = ['face_align.py',
                'vggface2',
                '/home/xiajun/res/face/VGGFace2/train',
                '/home/xiajun/res/face/VGGFace2/Experiment/train_mtcnn_align55x47_margin16',
                55, 47, 16]
    ''
    if len(sys.argv) < 7:
        print('Usage: python %s dataset_name src_folder dst_folder align_height align_width margin' % (sys.argv[0]))
        sys.exit()

    dataset_name = sys.argv[1]
    src_folder = sys.argv[2]
    dst_folder = sys.argv[3]
    align_height = int(sys.argv[4])
    align_width = int(sys.argv[5])
    margin = int(sys.argv[6])
    align_batches(dataset_name, src_folder, dst_folder, align_height, align_width, margin)
