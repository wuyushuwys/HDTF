"""
This file downloads almost all the videos from the HDTF dataset. Some videos are discarded for the following reasons:
- they do not contain cropping information because they are somewhat noisy (hand moving, background changing, etc.)
- they are not available on youtube anymore (at all or in the specified format)

The discarded videos constitute a small portion of the dataset, so you can try to re-download them manually on your own.

Usage:
```
$ python download.py --output_dir /tmp/data/hdtf --num_workers 8
```

You need tqdm and youtube-dl libraries to be installed for this script to work.
"""


import os, sys
import argparse
from typing import List, Dict
from multiprocessing import Pool
import subprocess
from subprocess import Popen, PIPE
from urllib import parse

from tqdm import tqdm
from pytube import YouTube
import ffmpeg
import logging

subsets = ["RD", "WDA", "WRA"]

logging.basicConfig(filename='download.log', filemode='w', level=logging.INFO, format='%(asctime)s::%(levelname)s::%(message)s')


def download_hdtf(source_dir: os.PathLike, output_dir: os.PathLike, num_workers: int, **process_video_kwargs):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, '_videos_raw'), exist_ok=True)

    download_queue = construct_download_queue(source_dir, output_dir)
    task_kwargs = [dict(
        video_data=vd,
        output_dir=output_dir,
        **process_video_kwargs,
     ) for vd in download_queue]
    pool = Pool(processes=num_workers)
    tqdm_kwargs = dict(total=len(task_kwargs), desc=f'Downloading videos into {output_dir} (note: without sound)', position=1, leave=False)

    for _ in tqdm(pool.imap_unordered(task_proxy, task_kwargs), **tqdm_kwargs):
        pass
    # for kwargs in tqdm(task_kwargs, **tqdm_kwargs):
    #     task_proxy(kwargs)


    logging.info('Download is finished, you can now (optionally) delete the following directories, since they are not needed anymore and occupy a lot of space:')
    logging.info(f" - {os.path.join(output_dir, '_videos_raw')}")


def construct_download_queue(source_dir: os.PathLike, output_dir: os.PathLike) -> List[Dict]:
    download_queue = []

    for subset in subsets:
        video_urls = read_file_as_space_separated_data(os.path.join(source_dir, f'{subset}_video_url.txt'))
        crops = read_file_as_space_separated_data(os.path.join(source_dir, f'{subset}_crop_wh.txt'))
        intervals = read_file_as_space_separated_data(os.path.join(source_dir, f'{subset}_annotion_time.txt'))
        resolutions = read_file_as_space_separated_data(os.path.join(source_dir, f'{subset}_resolution.txt'))

        for video_name, (video_url,) in video_urls.items():
            if not f'{video_name}.mp4' in intervals:
                logging.warning(f'Entire {subset}/{video_name} does not contain any clip intervals, hence is broken. Discarding it.')
                continue

            if not f'{video_name}.mp4' in resolutions or len(resolutions[f'{video_name}.mp4']) > 1:
                logging.warning(f'Entire {subset}/{video_name} does not contain the resolution (or it is in a bad format), hence is broken. Discarding it.')
                continue

            all_clips_intervals = [x.split('-') for x in intervals[f'{video_name}.mp4']]
            clips_crops = []
            clips_intervals = []

            for clip_idx, clip_interval in enumerate(all_clips_intervals):
                clip_name = f'{video_name}_{clip_idx}.mp4'
                if not clip_name in crops:
                    logging.warning(f'Clip {subset}/{clip_name} is not present in crops, hence is broken. Discarding it.')
                    continue
                clips_crops.append(crops[clip_name])
                clips_intervals.append(clip_interval)

            clips_crops = [list(map(int, cs)) for cs in clips_crops]

            if len(clips_crops) == 0:
                logging.warning(f'Entire {subset}/{video_name} does not contain any crops, hence is broken. Discarding it.')
                continue

            assert len(clips_intervals) == len(clips_crops)
            assert set([len(vi) for vi in clips_intervals]) == {2}, f"Broken time interval, {clips_intervals}"
            assert set([len(vc) for vc in clips_crops]) == {4}, f"Broken crops, {clips_crops}"
            assert all([vc[1] == vc[3] for vc in clips_crops]), f'Some crops are not square, {clips_crops}'

            download_queue.append({
                'name': f'{subset}_{video_name}',
                'id': parse.parse_qs(parse.urlparse(video_url).query)['v'][0],
                'intervals': clips_intervals,
                'crops': clips_crops,
                'output_dir': output_dir,
                'resolution': resolutions[f'{video_name}.mp4'][0]
            })

    return download_queue


def task_proxy(kwargs):
    return download_and_process_video(**kwargs)


def download_and_process_video(video_data: Dict, output_dir: str):
    """
    Downloads the video and cuts/crops it into several ones according to the provided time intervals
    """
    raw_download_path = os.path.join(output_dir, '_videos_raw', f"{video_data['name']}.mp4")
    raw_download_log_file = os.path.join(output_dir, '_videos_raw', f"{video_data['name']}_download_log.txt")
    download_result = download_video(video_data['id'], raw_download_path, resolution=video_data['resolution'], log_file=raw_download_log_file)

    if not download_result:
        logging.error(f'Failed to download {video_data}')
        logging.error(f'See {raw_download_log_file} for details')
        return

    # We do not know beforehand, what will be the resolution of the downloaded video
    # Youtube-dl selects a (presumably) highest one
    video_resolution = get_video_resolution(raw_download_path)
    # if not video_resolution != video_data['resolution']:
    #     logging.warning(f"Downloaded resolution is not correct for {video_data['name']}: {video_resolution} vs {video_data['name']}. Discarding this video.")
    #     return

    for clip_idx in range(len(video_data['intervals'])):
        start, end = video_data['intervals'][clip_idx]
        clip_name = f'{video_data["name"]}_{clip_idx:03d}'
        clip_path = os.path.join(output_dir, clip_name + '.mp4')
        try:
            probe = ffmpeg.probe(raw_download_path, select_streams='a')
            if not probe['streams']:
                logging.warning(f"{clip_path} does not have audio, skip")
                continue
        except Exception as e:
            logging.error(e.stderr)
        crop_success = cut_and_crop_video(raw_download_path, clip_path, start, end, video_data['crops'][clip_idx], video_resolution, video_data['resolution'])

        if not crop_success:
            logging.error(f'Failed to cut-and-crop clip #{clip_idx} {video_data}')
            continue


def read_file_as_space_separated_data(filepath: os.PathLike) -> Dict:
    """
    Reads a file as a space-separated dataframe, where the first column is the index
    """
    with open(filepath, 'r') as f:
        lines = f.read().splitlines()
        lines = [[v.strip() for v in l.strip().split(' ')] for l in lines]
        data = {l[0]: l[1:] for l in lines}

    return data


def download_video(video_id, download_path, resolution: int=None, video_format="mp4", log_file=None):
    """
    Download video from YouTube.
    :param video_id:        YouTube ID of the video.
    :param download_path:   Where to save the video.
    :param video_format:    Format to download.
    :param log_file:        Path to a log file for youtube-dl.
    :return:                Tuple: path to the downloaded video and a bool indicating success.

    Copy-pasted from https://github.com/ytdl-org/youtube-dl
    """
    # if os.path.isfile(download_path): return True # File already exists

    if log_file is None:
        stderr = subprocess.DEVNULL
    else:
        stderr = open(log_file, "w")
    try:
        yt = YouTube("https://youtube.com/watch?v={}".format(video_id), use_oauth=True, allow_oauth_cache=True)
        path_dir, filename = os.path.split(download_path)
        yt.streams.filter(progressive=True, 
                          file_extension=video_format).order_by('resolution').desc().first().download(filename=filename,
                                                                                                      output_path=path_dir)
        success = True
    except Exception as e:
        s = "Error {0}".format(str(e)) # string
        stderr.write(s)
        logging.error(f"{video_id} {s}")
        success = False

    if log_file is not None:
        stderr.close()
    
    return success and os.path.isfile(download_path)


def get_video_resolution(video_path: os.PathLike) -> int:
    # command = ' '.join([
    #     "ffprobe",
    #     "-v", "error",
    #     "-select_streams", "v:0", "-show_entries", "stream=height", "-of", "csv=p=0",
    #     video_path
    # ])

    # process = Popen(command, stdout=PIPE, shell=True)
    # (output, err) = process.communicate()
    # return_code = process.wait()
    # success = return_code == 0

    # if not success:
    #     print('Command failed:', command)
    #     return -1
    
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        height = int(video_stream['height'])
    except Exception as e:
        logging.error(f"{e.stderr}")

    

    return int(height)


def cut_and_crop_video(raw_video_path, output_path, start, end, crop: List[int], true_resolution, expected_resolution):
    if os.path.isfile(output_path) and os.path.getsize(output_path) > 0: return True # File already exists
    
    x, out_w, y, out_h = crop
    if eval(expected_resolution) != 0:
        scale = true_resolution / eval(expected_resolution)
        # print(eval(expected_resolution), true_resolution)
        adj_bb = org_bb = f"{x}, {out_w}, {y}, {out_h}"
        if scale != 1:
            x, out_w, y, out_h = int(x * scale), int(out_w * scale), int(y * scale), int(out_h * scale)
            adj_bb = f"{x}, {out_w}, {y}, {out_h}"
            logging.warning(f'{raw_video_path} :EXP->RET[{expected_resolution}->{true_resolution}] adjust {org_bb} -> {adj_bb}')
    else:
        logging.error(f"{raw_video_path} get {expected_resolution} resolution")
        return
    # command = ' '.join([
    #     "ffmpeg", "-i", raw_video_path,
    #     "-strict", "-2", # Some legacy arguments
    #     "-loglevel", "quiet", # Verbosity arguments
    #     "-qscale", "0", # Preserve the quality
    #     "-y", # Overwrite if the file exists
    #     "-ss", str(start), "-to", str(end), # Cut arguments
    #     "-filter:v", f'"crop={out_w}:{out_h}:{x}:{y}"', # Crop arguments
    #     output_path
    # ])
    ffmpeg_obj = ffmpeg.input(raw_video_path)
    video = ffmpeg_obj.video.crop(height=out_h, width=out_w, x=x, y=y).trim(start=start, end=end).filter('fps', fps=25, round='up').filter('setpts', 'PTS-STARTPTS')
    audio = ffmpeg_obj.audio.filter('atrim', start=start, end=end).filter('asetpts', 'PTS-STARTPTS')
    try:
        ffmpeg.output(video, audio, output_path, **{'qscale:v': 0, 'qscale:a': 0}).run(overwrite_output=True, quiet=True)
        success = True
    except ffmpeg.Error as e:
        logging.error(f"failed at crop {raw_video_path} EXP->RET[{expected_resolution}->{true_resolution}] {org_bb}->{adj_bb} {start}-{end}")
        logging.error(f"stderr: {e.stderr.decode('utf8')}")
        success = False
        if os.path.isfile(output_path) and os.path.getsize(output_path) == 0:
            logging.error('Remove error output', output_path)
            os.remove(output_path)
    
    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HDTF dataset")
    parser.add_argument('-s', '--source_dir', type=str, default='HDTF_dataset', help='Path to the directory with the dataset')
    parser.add_argument('-o', '--output_dir', type=str, help='Where to save the videos?')
    parser.add_argument('-w', '--num_workers', type=int, default=8, help='Number of workers for downloading')
    args = parser.parse_args()

    download_hdtf(
        args.source_dir,
        args.output_dir,
        args.num_workers,
    )
