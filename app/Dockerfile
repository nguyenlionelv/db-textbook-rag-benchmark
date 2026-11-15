FROM node:22-bookworm-slim AS frontend
WORKDIR /frontend
COPY app/frontend/package.json app/frontend/package-lock.json ./
RUN npm ci
COPY app/frontend/index.html app/frontend/tsconfig.json app/frontend/vite.config.ts ./
COPY app/frontend/src ./src
COPY app/frontend/public ./public
RUN npm run build

FROM python:3.13-slim
WORKDIR /project
COPY app/requirements.txt ./app/requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 \
    && pip install --no-cache-dir -r app/requirements.txt
COPY src ./src
COPY app/__init__.py ./app/__init__.py
COPY app/backend ./app/backend
COPY --from=frontend /frontend/dist ./app/frontend/dist
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "-m", "app.backend.docker_entrypoint"]
