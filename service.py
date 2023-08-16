from Model_deploy import TIMNET_Model
from fastapi import FastAPI, status, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pydub import AudioSegment
from loguru import logger
from natsort import natsorted
import numpy as np
import os
import io
import requests
import librosa
import time

logger.add("serving_{time}.log", level="INFO", rotation="5 MB", retention=2)
"""
模型配置开始
"""
CLASS_LABELS = ("angry", "fear", "happy", "neutral", "sad", "excited")
Model = TIMNET_Model(input_shape=(87, 39), class_label=CLASS_LABELS)
Model.create_model()
"""
模型配置结束
"""


class AudioItem(BaseModel):
    audio: str


def download_audio(audio):
    """
    根据URL请求下载音频
    """
    return requests.get(audio).content


def convert_audio_to_wav(binary_data, file_path):
    """
    ['mp3', 'wav', 'ogg', 'flv', 'mp4', 'aac']格式的二进制音频文件转为16khz,mono的wav保存到本地
    """
    byte_stream = io.BytesIO(binary_data)
    formats = ["mp3", "wav", "ogg", "flv", "mp4", "aac"]
    audio = None
    for format in formats:
        try:
            audio = AudioSegment.from_file(byte_stream, format=format)
            break
        except:
            pass
    if audio is None:
        raise ValueError("Invalid audio file format")

    if audio.channels > 1:
        audio = audio.set_channels(1)
    audio = audio.set_frame_rate(22050)
    output_file_path = file_path.rsplit(".", 1)[0] + ".wav"
    audio.export(output_file_path, format="wav")


def get_feature(file_path: str, mfcc_len: int = 39, mean_signal_length: int = 44100):
    """
    file_path: Speech signal folder
    mfcc_len: MFCC coefficient length
    mean_signal_length: MFCC feature average length
    """
    signal, fs = librosa.load(file_path)  # signal: speech signal, fs: sampling rate
    mfcc = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=mfcc_len)
    mfcc = mfcc.T
    feature = mfcc
    return feature


def split_audio(input_file, output_folder):
    """
    2s为单位切割并保存wav文件
    """
    audio = AudioSegment.from_wav(input_file)
    duration = len(audio)
    split_length = 2000  # 2秒

    start_time = 0
    end_time = split_length
    i = 0

    while end_time <= duration:
        split_audio = audio[start_time:end_time]
        output_file = os.path.join(output_folder, f"{i}.wav")
        split_audio.export(output_file, format="wav")
        start_time += split_length
        end_time += split_length
        i += 1

    if end_time > duration:
        pass


def get_all_features(directory: str, batch_size: int = 64):
    features = []
    file_names = natsorted(os.listdir(directory))
    for file_name in file_names:
        file_path = os.path.join(directory, file_name)
        feature = get_feature(file_path)
        features.append(feature)

    features = np.array(features)
    num_batches = len(features) // batch_size

    return np.array_split(features, num_batches)


app = FastAPI()


@app.post("/ser")
def get_prediction(items: AudioItem):
    decoded_data = download_audio(items.audio)
    try:
        convert_audio_to_wav(decoded_data, "tmp.wav")
        split_audio("tmp.wav", "output")
    except Exception as e:
        error_message = str(e)
        logger.error(f"{e}")
        return JSONResponse(status_code=400, content={"error": error_message})
    result = []
    features = get_all_features("/home/TIM-Net_SER/Code/output")
    for batch in features:
        predictions = Model.predict(batch)
        for prediction in predictions:
            prediction_label = CLASS_LABELS[np.argmax(np.array(prediction))]
            result.append(prediction_label)

    file_names = os.listdir("/home/TIM-Net_SER/Code/output")
    for file_name in file_names:
        file_path = os.path.join("/home/TIM-Net_SER/Code/output", file_name)
        os.remove(file_path)
    logger.info(f"{result}")
    return {"prediction": result}


@app.get("/health")
async def health_check():
    try:
        logger.debug("health 200")
        return status.HTTP_200_OK
    except:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)


@app.get("/health/inference")
async def health_check():
    try:
        logger.debug("health 200")
        return status.HTTP_200_OK

    except:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)
