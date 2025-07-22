# step 1.
import shlex
from pydantic import BaseModel
from typing import List
import bentoml
from bentoml.io import JSON, Image
from bentoml.exceptions import BentoMLException
import numpy as np
import pandas as pd
from PIL import Image as PILImage
from datetime import datetime
import subprocess
import logging
import asyncio
import warnings
warnings.filterwarnings('ignore')

# step 2.
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s\n\n', datefmt='%Y-%m-%d %H:%M:%S')
ch.setFormatter(formatter)
bentoml_logger = logging.getLogger("bentoml")
if not bentoml_logger.hasHandlers():
    bentoml_logger.addHandler(ch)
bentoml_logger.setLevel(logging.INFO)

def log_and_raise_error(message: str, exc: Exception):
    bentoml_logger.error(f"{message}: {exc}")
    raise BentoMLException(message)

# step 3.
class ModelTrainingParams(BaseModel):
    algorithm: str
    epochs: int

# step 4.
svc = bentoml.Service('ai_test')

# step 5.
@svc.api(input=JSON(), output=JSON(), route='/training')
async def training(params: dict):
    try:
        parsed_params = ModelTrainingParams(**params)
        bentoml_logger.info('입력 파라미터가 모두 유효합니다.')
        algorithm = parsed_params.algorithm
        epochs = parsed_params.epochs
        cmd = (
            f"python3 /mnt/working/kubeflow/pipelines/model_training.py "
            f"--algorithm={algorithm} "
            f"--epochs={epochs}"
        )
        proc = await asyncio.create_subprocess_exec(
            *shlex.split(cmd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            log_and_raise_error("모델 훈련 스크립트 실행 실패", stderr.decode())
        bentoml_logger.info('모델 훈련 스크립트 실행 완료')
        bentoml_logger.info('모델 훈련 컨테이너를 생성합니다.')
    except Exception as e:
        bentoml_logger.error(f"입력 파라미터 파싱 실패: {e}")
        raise BentoMLException('입력 파라미터가 유효하지 않습니다.')

# step 6.
@svc.api(input=JSON(), output=JSON(), route='/inference')
async def inference(img: PILImage.Image) -> PILImage.Image:
    name_tag = 'yolo_test'
    try:
        model_path = '/mnt/working/model_save/bentoml/train/weights/best.pt'
        with bentoml.models.create(
            name=name_tag,
            custom_objects={'pt_path': model_path},
            labels={'timestamp': datetime.now().isoformat()}
        ) as bento_model:
            bentoml_logger.info(f"BentoML 저장소에서 사용할 수 있는 추론 모델은 {bento_model.tag} 입니다.")
        models = bentoml.models.list()
        bentoml_logger.info(f"BentoML 모델 개수: {len(models)}")
        for model in models:
            bentoml_logger.info(f"BentoML 모델 이름: {model.tag}, 저장 경로: {model.path}")
        try:
            model_info = bentoml.models.get(name_tag)
            model_path = model_info.custom_objects['model_path']
            model = YOLO(model_path)
            bentoml_logger.info('BentoML 저장소에서 추론 모델을 성공적으로 로딩했습니다.')
        except Exception as e:
            bentoml_logger.error(f"BentoML 저장소에서 추론 모델 로딩 실패: {e}")
            raise BentoMLException('BentoML 저장소에서 추론 모델을 로딩하는데 실패했습니다.')
        img_np = np.array(img)
        try:
            result = model.predict(img_np)
            result_image = result[0].plot()
            bentoml_logger.info('모델 추론에 성공했습니다.')
        except Exception as e:
            bentoml_logger.error(f"모델 추론 실패: {e}")
            raise BentoMLException('모델 추론에 실패했습니다.')        
        return PILImage.fromarray(result_image)
    except Exception as e:
        bentoml_logger.info(f": BentoML 저장소에서 추론 모델 생성 실패: {e}")
        raise BentoMLException('BentoML 저장소에서 사용할 수 있는 추론 모델이 없습니다.')
