# step 1.
from functools import partial
from kfp.components import create_component_from_func
import kfp
from kfp import onprem
from kfp import compiler
from kfp import dsl
from kfp.dsl import component
import argparse
import requests
import asyncio
import bentoml
from pydantic import BaseModel
import typing as t
import requests
from kubernetes.client import V1EnvVar

# step 2.
@partial(create_component_from_func, base_image='dgkim1983/dlabflow:model-20250304')
def Training(algorithm: str, epochs:int):
    from ultralytics import YOLO
    import logging
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s\n\n', datefmt='%Y-%m-%d %H:%M:%S')
    ch.setFormatter(formatter)
    kubeflow_logger = logging.getLogger("bentoml")
    if not kubeflow_logger.hasHandlers():
        kubeflow_logger.addHandler(ch)
    kubeflow_logger.setLevel(logging.INFO)
    def model_training():
        model = YOLO(algorithm)
        results = model.train(data='coco8.yaml', epochs=epochs, project='/mnt/working', name='models', exist_ok=True)
    try:
        model_training()
        kubeflow_logger.info(f"{algorithm} 파일에서 YOLO 모델을 성공적으로 로딩했으며, 이 파일에는 사전 학습된 가중치가 포함되어 있습니다.")
    except Exception as e:
        kubeflow_logger.error(f'{algorithm} 파일에서 YOLO 모델 로딩 실패: {e}')
        raise

# step 3.
def pipelines():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--algorithm', type=str)
    parser.add_argument('--epochs', type=int)
    args = parser.parse_args()
    Training_task = Training(args.algorithm, args.epochs) \
        .set_display_name('Model Training') \
        .apply(onprem.mount_pvc('example-claim', volume_name='data', volume_mount_path='/mnt/working')) \
        .add_env_variable(V1EnvVar(name="CUDA_VISIBLE_DEVICES", value="0"))
    smh_vol = kfp.dsl.PipelineVolume(name = 'shm-vol', empty_dir = {'medium': 'Memory'})
    Training_task.add_pvolumes({'/dev/shm': smh_vol})

# step 4.
if __name__ == '__main__':
    pipeline_package_path = 'training_pipelines.zip'
    kfp.compiler.Compiler().compile(pipelines, pipeline_package_path)
    HOST = 'http://10.40.217.244:8080/'
    USERNAME = 'user@example.com'
    PASSWORD = '12341234'
    NAMESPACE = 'kubeflow-user-example-com'
    session = requests.Session()
    response = session.get(HOST)
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = {'login': USERNAME, 'password': PASSWORD}
    session.post(response.url, headers = headers, data = data)
    session_cookie = session.cookies.get_dict()['authservice_session']
    client = kfp.Client(host = f'{HOST}/pipeline', cookies = f'authservice_session={session_cookie}', namespace = NAMESPACE)
    experiment = client.create_experiment(name='Training')
    run = client.run_pipeline(experiment.id, 'Training pipelines', pipeline_package_path)

