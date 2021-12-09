import librosa
import soundfile as sf
import glob
import os
import torchaudio
import torch
import IPython.display as ipd
import matplotlib.pyplot as plt
from multiprocessing import Process


def createDirectory(directory): 
    try: 
        if not os.path.exists(directory): 
            os.makedirs(directory) 
    except OSError: 
        print("Error: Failed to create the directory.")

def list_chunk(lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]

def write_wav_file(wav_fn_list, index):
    for idx, wav_fn in enumerate(wav_fn_list):    
        print(f'process {index}' + str(((idx+1) / len(wav_fn_list))*100) + '%')    

        # 오디오 읽어오기
        audio, sr = librosa.load(wav_fn, sr = 16000, mono = True)
        clip = librosa.effects.trim(audio, top_db = 35)

        # 디렉토리가 없다면 생성
        createDirectory(file_path + wav_fn[47:51])
        clip_tensor = clip[0]

        # 오디오 길이가 20480이하일 경우 늘려주는 작업
        if len(clip[0]) <= 20480:
            clip_tensor = torch.from_numpy(clip[0]).unsqueeze(0)
            if 20480//len(clip[0]) == 1:
                clip_tensor = torch.cat((clip_tensor,clip_tensor),1)
            elif 20480//len(clip[0]) == 2:
                clip_tensor = torch.cat((clip_tensor,clip_tensor, clip_tensor),1)
            elif 20480//len(clip[0]) == 3:
                clip_tensor = torch.cat((clip_tensor,clip_tensor, clip_tensor, clip_tensor),1)

            clip_tensor = clip_tensor.squeeze(0).numpy()


        #파일 쓰기
        sf.write(file_path + wav_fn[47:], clip_tensor, sr)
        
wav_fn_list = glob.glob('/data/public_dataset/speaker_recognition/train/*/*.wav', recursive = True)
file_path = '/data/kdg_workspace/speaker_recognition/train_wav/'

wav_fn_list = list_chunk(wav_fn_list, 71622)

procs = []

for index, fn_list in enumerate(wav_fn_list):
    proc = Process(target = write_wav_file, args=(fn_list, index))
    procs.append(proc)
    proc.start()
        
#프로세스 종료 대기
for proc in procs:
    proc.join()
    

