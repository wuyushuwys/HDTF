import argparse
from click import FileError
import ffmpeg
import os
import logging
import cv2
import traceback

from pathlib import Path
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from facexlib.detection import init_detection_model
from utils import ImageFolder, get_largest_face, get_center_face, EMA

import h5py
import numpy as np
import warnings

warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True,
                    help='Dir containing youtube clips.')
parser.add_argument('--output_dir', type=str, required=True,
                    help='Location to dump outputs.')
parser.add_argument('--num_workers', type=int, default=4,
                    help='How many multiprocessing workers')
parser.add_argument('--max_frames', type=int, default=None,
                    help='Max frames extracted in total')
parser.add_argument('--resume_file', type=str, default=None,
                    help='resume processed file')
parser.add_argument('--failed_file', type=str, default=None,
                    help='only processed failed file')
parser.add_argument('--face_size', type=int, default=512,
                    help='face_size')
parser.add_argument('--cache', action='store_true',
                    help='whether cache file')
parser.add_argument('--landmark_alignment', '-la',action='store_true',
                    help='landmark alignment')
parser.add_argument('--ext', type=str, default='jpg', choices=['jpg', 'png'],
                    help='Extension for image frames')
args = parser.parse_args()

EXT = args.ext

index = int(time())

logging.basicConfig(filename=f'extract_av_{index}.log', filemode='w', level=logging.INFO, format='%(asctime)s::%(levelname)s::%(lineno)d::%(message)s')

record_logger = logging.getLogger("processed_file")
record_logger.setLevel(logging.INFO)
record_handler = logging.FileHandler(filename=f'processed_file_{index}.log', mode='w')
record_handler.setLevel(logging.INFO)
record_handler.setFormatter(logging.Formatter('%(message)s'))
record_logger.addHandler(record_handler)

fail_logger = logging.getLogger("failed_file")
fail_logger.setLevel(logging.INFO)
fail_handler = logging.FileHandler(filename=f'failed_file_{index}.log', mode='w')
fail_handler.setLevel(logging.INFO)
fail_handler.setFormatter(logging.Formatter('%(message)s'))
fail_logger.addHandler(fail_handler)

check_logger = logging.getLogger("check_file")
check_logger.setLevel(logging.INFO)
check_handler = logging.FileHandler(filename=f'check_file_{index}.log', mode='w')
check_handler.setLevel(logging.INFO)
check_handler.setFormatter(logging.Formatter('%(message)s'))
check_logger.addHandler(check_handler)

face_detectors = [init_detection_model(model_name='retinaface_resnet50', half=True, device=f'cuda:{idx}') for idx in range(args.num_workers)]
face_template = np.array([[192.98138, 239.94708],
                          [318.90277, 240.1936],
                          [256.63416, 314.01935],
                          [201.26117, 371.41043],
                          [313.08905, 371.15118]])
face_size = (args.face_size, args.face_size)
face_template = face_template * (face_size[0] / 512.0)

def video_extract(output_dir, file_path, job_id):

    if not os.path.isfile(file_path):
        logging.warning(f"{file_path} not found")
        return 
    
    if args.max_frames:
        current_files = len(glob(f"{output_dir}/**/*.{EXT}", recursive=True))
        logging.info(f"{file_path}: current files {current_files}/{args.max_frames}[{current_files/args.max_frames:.02%}]")
        if current_files > args.max_frames:
            return
    else:
        current_files = len(glob(f"{output_dir}/**/*.{EXT}", recursive=True))
        logging.info(f"{file_path}: current files {current_files}")
    pid = job_id

    
    filename = Path(file_path).stem
    
    try:
        probe = ffmpeg.probe(file_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        audio_info = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
        if not audio_info:
            logging.warning(f"{file_path} has No audio, skip.")
            return
        fps = eval(video_info['r_frame_rate'])
    except Exception as e:
        logging.error(f"{e.stderr} at probe {file_path}")
        fail_logger.info(file_path)
    else:
        assert fps == 25, f"Expected fps=25, but got {fps}"

    output_folder = os.path.join(output_dir, f'{filename}')
    os.makedirs(output_folder, exist_ok=True)
    
    streams = ffmpeg.input(file_path)
    logging.info(f"{output_folder} frame extraction")
    try:
        streams.video.output(os.path.join(output_folder, f'%06d.{EXT}'),
                             **{'qscale:v':0}).run(overwrite_output=True, quiet=True)
    except ffmpeg.Error as e:
        logging.error("Error in frame extraction")
        logging.error(e.stderr.decode('utf-8'))
        fail_logger.info(file_path)
        return
    try:
        streams.audio.output(os.path.join(output_folder, f'audio.wav'),
                             **{"qscale:a": 1, 'ar': 16000}).run(overwrite_output=True, quiet=True)
    except ffmpeg.Error as e:
        logging.error("Error in audio extraction")
        logging.error(e.stderr.decode('utf-8'))
        fail_logger.info(file_path)
        return
    ema_landmark = None
    try:
        logging.info(f"Caching images at {Path(output_folder) / 'cache.h5'}")
        logging.info(f"Write meta data to {Path(output_folder) / 'meta_data.txt'}")
        with h5py.File(Path(output_folder) / 'cache.h5', 'w', libver='latest') as h5_cache, open(Path(output_folder) / 'meta_data.txt', 'w') as meta_file:
            dataset = ImageFolder(output_folder, ext=EXT, output_mode='cv2')
            bsz = dataset.max_bsz_retinaface(pid)
            logging.info(f"BSZ {bsz} at {output_folder} for face crop")
            dataloader = DataLoader(dataset, batch_size=bsz, shuffle=False, num_workers=4)
            
            face_detector = face_detectors[pid]
            all_landmarks = []
            for data in dataloader:
                names, imgs = data['name'], data['img']
                try: 
                    with torch.no_grad():
                        batched_bboxes, batched_landmarks = face_detector.batched_detect_faces(imgs)
                except Exception as e:
                    logging.error(f"{output_folder} {e} at inference")
                    fail_logger.info(file_path)
                    return
                
                b, h, w, c = imgs.size()
                batched_det_faces = []
                batched_det_landmarks = []
                try:
                    for bboxes, landmarks in zip(batched_bboxes, batched_landmarks):
                        det_faces = []
                        det_landmarks = []
                        if len(bboxes) == 0:
                            batched_det_faces.append(None)
                            batched_det_landmarks.append(None)
                        else:
                            for bbox, landmark in zip(bboxes, landmarks):
                                det_faces.append(bbox[0:5])
                                det_landmarks.append(np.array(np.split(landmark, 5, axis=0)))
                            det_faces, face_idx = get_center_face(det_faces, h=h, w=w)
                            batched_det_faces.append(det_faces.astype(int).tolist()[:4])
                            batched_det_landmarks.append(det_landmarks[face_idx])
                except Exception as e:
                    logging.error(f"failed at {output_path} at bbox extraction")
                else:
                    all_landmarks.extend(batched_det_landmarks)
                
                
                for name, bbox, landmark, img in zip(names, batched_det_faces, batched_det_landmarks, imgs):
                    output_path = os.path.join(output_folder, f'{name}.{EXT}')  # override original image
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        if args.landmark_alignment:
                            if np.all(y1 < landmark[:, 0] < y2) and np.all(x1 < landmark[:, 1] < x2):
                                if ema_landmark is not None:
                                    ema_landmark.update(landmark)
                                else:
                                    ema_landmark = EMA(landmark)
                            else:    
                                if ema_landmark is None:
                                    logging.warning(f"{output_path} contains bad landmark. Please check or reprocess")
                                    check_logger.info(file_path)
                                    fail_logger.info(file_path)
                                    raise ValueError   
                                logging.warning(f"Bad landmark detection. Using ema landmark {output_path}")
                                check_logger.info(file_path)
                            
                        if x1 >= x2:
                            x1, x2 = 0, img.size(2)
                            logging.error(f"{output_path} Bad detection {img.shape} {(x1, y1, x2, y2)}")
                        if y1 >= y2:
                            y1, y2 = 0, img.size(1)
                            logging.error(f"{output_path} Bad detection {img.shape} {(x1, y1, x2, y2)}")
                        
                    else:
                        logging.error(f"{output_path} failed to find face. Please check videos")
                        fail_logger.info(file_path)
                        raise ValueError   
                    try:
                        if args.landmark_alignment:
                            border_mode = cv2.BORDER_CONSTANT
                            affine_matrix = cv2.estimateAffinePartial2D(landmark, face_template, method=cv2.LMEDS)[0]
                            cropped_face = cv2.warpAffine(img.numpy(), affine_matrix, face_size, borderMode=border_mode, borderValue=(135, 133, 132))
                            tensor_image = torch.tensor(cropped_face).permute(2, 0, 1).flip(0)
                        else:
                            cropped_face = img.numpy()[y1:y2, x1:x2, :]
                            tensor_image = img[y1:y2, x1:x2, :].permute(2, 0, 1).flip(0)
                        meta_file.write(name+'\n')
                        if args.cache:
                            h5_cache.create_dataset(
                                name, data=tensor_image, maxshape=tensor_image.shape,
                                compression='lzf', shuffle=True, track_times=False,)
                            os.remove(output_path)
                        else:
                            cv2.imwrite(output_path, cropped_face)
                    except Exception as e:
                        logging.error(f"{output_path} {e}")
                        fail_logger.info(file_path)
                        raise FileError(output_path)    
            # if args.landmark_alignment:
            #     normlized_landmark = np.array(all_landmarks)/face_size[0]
            #     if np.any(normlized_landmark.std(axis=0) > 0.1):
            #         check_logger.info(file_path)
            #         logging.warning(f"{output_path} landmarks might contain bad detection. Please check or reprocess")
        if not args.cache:
            os.remove(Path(output_folder) / 'cache.h5')
    except Exception as e:
        logging.error(f"{output_folder} {e}  at caching")
        fail_logger.info(file_path)
        return
    else:
        record_logger.info(file_path)
        logging.info(f"Caching {Path(output_folder) / 'cache.h5'} finished")

    # h5 = Hdf5(Path(output_folder) / 'cache.h5', overwrite=True)
    
    # with h5py.File(Path(output_folder) / 'cache.h5', 'w', libver='latest') as h5:
    #     imgs = glob(f'{output_folder}/*.{EXT}')
    #     with open(Path(output_folder) / 'meta_data.txt', 'a') as f:
    #         for img in imgs:
    #             name = Path(img).stem
    #             f.write(name+'\n')
    #             value = read_image(img)
    #             h5.create_dataset(
    #                 name, data=value, maxshape=value.shape,
    #                 compression='lzf', shuffle=True, track_times=False,
    #             )
    #             os.remove(img)


def mp_handler(job):
	output_dir, file_path, job_id = job
	try:
		video_extract(output_dir, file_path, job_id)
	except KeyboardInterrupt:
		exit(0)
	except:
		traceback.print_exc()

if __name__ == '__main__':
    os.makedirs(os.path.join(args.output_dir), exist_ok=True)
    if args.failed_file:
        filelist = []
        with open(args.failed_file, 'r') as f:
            for line in f.readlines():
                filelist.append(line.strip('\n'))
        logging.info(f"Only process from {args.failed_file}")
    else:
        filelist = sorted(glob(f"{args.input_dir}/*.mp4", recursive=True))

        if args.resume_file:
            previous_file = []
            with open(args.resume_file, 'r') as f:
                for line in f.readlines():
                    previous_file.append(line.strip('\n'))
            filelist = sorted(list(set(filelist).difference(set(previous_file))))
            logging.info(f"Resume from {args.resume_file}")

    
    if args.num_workers == 1:
        for file_path in filelist:
            video_extract(args.output_dir, file_path, 0)
    else:
        jobs = [(args.output_dir, fname, i%args.num_workers) for i, fname in enumerate(filelist)]
        p = ThreadPoolExecutor(args.num_workers)
        futures = [p.submit(mp_handler, j) for j in jobs]
        _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures), dynamic_ncols=True)]