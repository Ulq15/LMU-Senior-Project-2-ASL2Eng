import json
from math import ceil
import cv2
import os
import shutil
import csv

def convert_everything_to_mp4():
    cmd = 'bash data/scripts/swf2mp4.sh'
    os.system(cmd)


def video_to_frames(video_path, size=None):
    """
    video_path -> str, path to video.
    size -> (int, int), width, height.
    """

    videoSRC = cv2.VideoCapture(video_path)
    old_frames = []
    frames = []
    while True:
        ret, frame = videoSRC.read(old_frames)
        if ret:
            height, width, layers = frame.shape
            if size is None:
                scale = 256
                ratio = float(height)/scale
                size = (int(width/ratio), int(height/ratio))
            frame = cv2.resize(frame, size)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
        else:
            break
    videoSRC.release()
    return frames


def convert_frames_to_video(frame_array, path_out, size, fps=25):
    out = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


def extract_frame_as_video(src_video_path, start_frame, end_frame):
    frames = video_to_frames(src_video_path)
    return frames[start_frame: end_frame+1]


def extract_all_yt_instances(content):
    cnt = 1
    if not os.path.exists('data/videos'):
        os.mkdir('data/videos')
    for entry in content:
        instances = entry['instances']
        for inst in instances:
            url = inst['url']
            video_id = inst['video_id']
            if 'youtube' in url or 'youtu.be' in url:
                cnt += 1
                yt_identifier = url[-11:]
                src_video_path = os.path.join('data', 'raw_videos_mp4', yt_identifier + '.mp4')
                dst_video_path = os.path.join('data', 'videos', video_id + '.mp4')
                if not os.path.exists(src_video_path):
                    continue
                if os.path.exists(dst_video_path):
                    print('{} exists.'.format(dst_video_path))
                    continue
                # because the JSON file indexes from 1.
                start_frame = inst['frame_start'] - 1
                end_frame = inst['frame_end'] - 1
                if end_frame <= 0:
                    shutil.copyfile(src_video_path, dst_video_path)
                    continue
                selected_frames = extract_frame_as_video(src_video_path, start_frame, end_frame)
                # when OpenCV reads an image, it returns size in (h, w, c)
                # when OpenCV creates a writer, it requres size in (w, h).
                size = selected_frames[0].shape[:2][::-1]
                convert_frames_to_video(selected_frames, dst_video_path, size)
                print(cnt, dst_video_path)
            else:
                cnt += 1
                src_video_path = os.path.join('data', 'raw_videos_mp4', video_id + '.mp4')
                dst_video_path = os.path.join('data', 'videos', video_id + '.mp4')
                if os.path.exists(dst_video_path):
                    print('{} exists.'.format(dst_video_path))
                    continue
                if not os.path.exists(src_video_path):
                    continue
                print(cnt, dst_video_path)
                # shutil.copyfile(src_video_path, dst_video_path)
                frames = video_to_frames(src_video_path)
                convert_frames_to_video(frames, dst_video_path, frames[0].shape[:2][::-1])


def organizeFS(content, video_path, path_out):
    if not os.path.exists(path_out):
        os.mkdir(path_out)
        os.mkdir(os.path.join(path_out, 'train'))
        os.mkdir(os.path.join(path_out, 'test'))
        os.mkdir(os.path.join(path_out, 'val'))
    for entry in content:
        instances = entry['instances']
        word = entry['gloss']
        for inst in instances:
            video_id = inst['video_id']
            split = inst['split']
            src_path = os.path.join(video_path, video_id + '.mp4')
            if not os.path.exists(src_path):
                print(f'Did not find video id:{video_id} for word \'{word}\'.\n')
                continue
            dst_path = os.path.join(path_out, split, video_id + '.mp4')
            if os.path.exists(dst_path):
                print(f'Found video id:{video_id} for word \'{word}\' already in destination.\n')
                continue
            shutil.move(src_path, dst_path)
            with open(os.path.join(path_out, split+'.csv'), 'a+', encoding='UTF8', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([dst_path, word])
    if os.path.isdir(video_path) and len(os.listdir(video_path)) == 0:
        os.rmdir(video_path)


def train_test_split(src_path, dst_path, ratio):
    train_path = os.path.join(dst_path, 'train')
    test_path = os.path.join(dst_path, 'test')
    val_path = os.path.join(dst_path, 'val')
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
        os.mkdir(train_path)
        os.mkdir(test_path)
        os.mkdir(val_path)
    for folder in os.listdir(src_path):
        word = folder
        video_path = os.path.join(src_path, word)
        if not os.path.exists(os.path.join(train_path, word)):
            os.mkdir(os.path.join(train_path, word))
        if not os.path.exists(os.path.join(test_path, word)):
            os.mkdir(os.path.join(test_path, word))
        if not os.path.exists(os.path.join(val_path, word)):
            os.mkdir(os.path.join(val_path, word))
        videos = [f for f in os.listdir(video_path)]
        count = len(videos)
        num_Vidtest = ceil(count * ratio)
        for _ in range(num_Vidtest):
            v = videos.pop()
            shutil.copy(os.path.join(video_path, v), os.path.join(test_path, word, v))
        for v in videos:
            shutil.copy(os.path.join(video_path, v), os.path.join(train_path, word, v))


def createCSV(dir_path, csv_name):
    with open(os.path.join(dir_path, csv_name+'.csv'), 'w', encoding='UTF8', newline='') as file:
        writer = csv.writer(file)
        for word in os.listdir(os.path.join(dir_path, csv_name)):
            for vid in os.listdir(os.path.join(dir_path, csv_name, word)):
                writer.writerow([os.path.join(dir_path, csv_name, word, vid), word])


def main():
    # convert_everything_to_mp4()
    content = json.load(open('WLASL_v0.3.json'))
    extract_all_yt_instances(content)

    video_path = os.path.join('data','videos')
    # org_path = os.path.join('data','organized_vids')
    split_path = os.path.join('data','split_data')
    organizeFS(content, video_path, split_path)
    # train_test_split(org_path, split_path, 0.2)

    # createCSV(split_path, 'test')
    # createCSV(split_path, 'train')
    exit(0)

if __name__=="__main__":
    main()