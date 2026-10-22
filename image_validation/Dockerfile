FROM public.ecr.aws/lambda/python:3.12
WORKDIR ${LAMBDA_TASK_ROOT}
RUN dnf install -y git && dnf clean all
COPY requirements-lambda.txt .
RUN pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir setuptools==80.9.0 wheel==0.45.1 \
    && pip install --no-cache-dir --no-build-isolation -r requirements-lambda.txt
# Download CLIP during the build; Lambda's filesystem is read-only at runtime.
ENV HOME=/opt/models EXP32_DEVICE=cpu \
    OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN python -c "import clip; clip.load('ViT-B/32', device='cpu')" \
    && chmod -R a+rX /opt/models
COPY inference_service_exp32.py .
COPY lambda_handler.py .
COPY artifacts/exp03_2_clip_refined/reg_mlp/best.pt artifacts/exp03_2_clip_refined/reg_mlp/best.pt
CMD ["lambda_handler.handler"]
