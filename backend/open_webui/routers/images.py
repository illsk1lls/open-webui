import asyncio
import base64
import uuid
import io
import json
import logging
import mimetypes
import re
from pathlib import Path
from typing import Optional

from urllib.parse import quote
import aiohttp

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from open_webui.config import (
    CACHE_DIR,
    IMAGE_AUTO_SIZE_MODELS_REGEX_PATTERN,
    IMAGE_URL_RESPONSE_MODELS_REGEX_PATTERN,
)
from open_webui.constants import ERROR_MESSAGES
from open_webui.retrieval.web.utils import validate_url
from open_webui.env import AIOHTTP_CLIENT_SESSION_SSL, AIOHTTP_CLIENT_ALLOW_REDIRECTS, ENABLE_FORWARD_USER_INFO_HEADERS
from open_webui.utils.session_pool import get_session

from open_webui.models.chats import Chats
from open_webui.routers.files import upload_file_handler, get_file_content_by_id
from open_webui.utils.auth import get_admin_user, get_verified_user
from open_webui.utils.access_control import has_permission
from open_webui.utils.headers import include_user_info_headers
from open_webui.internal.db import get_async_session
from sqlalchemy.ext.asyncio import AsyncSession
from open_webui.utils.images.comfyui import (
    ComfyUICreateImageForm,
    ComfyUIEditImageForm,
    ComfyUIWorkflow,
    comfyui_upload_image,
    comfyui_create_image,
    comfyui_edit_image,
)
from pydantic import BaseModel

log = logging.getLogger(__name__)

IMAGE_CACHE_DIR = CACHE_DIR / 'image' / 'generations'
IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter()


async def set_image_model(request: Request, model: str):
    log.info(f'Setting image model to {model}')
    request.app.state.config.IMAGE_GENERATION_MODEL = model
    if request.app.state.config.IMAGE_GENERATION_ENGINE in ['', 'automatic1111']:
        api_auth = get_automatic1111_api_auth(request)
        try:
            session = await get_session()
            async with session.get(
                url=f'{request.app.state.config.AUTOMATIC1111_BASE_URL}/sdapi/v1/options',
                headers={'authorization': api_auth},
                ssl=AIOHTTP_CLIENT_SESSION_SSL,
            ) as r:
                options = await r.json()
            if model != options['sd_model_checkpoint']:
                options['sd_model_checkpoint'] = model
                async with session.post(
                    url=f'{request.app.state.config.AUTOMATIC1111_BASE_URL}/sdapi/v1/options',
                    json=options,
                    headers={'authorization': api_auth},
                    ssl=AIOHTTP_CLIENT_SESSION_SSL,
                ) as r:
                    r.raise_for_status()
        except Exception as e:
            log.debug(f'{e}')
    return request.app.state.config.IMAGE_GENERATION_MODEL


async def get_image_model(request):
    if request.app.state.config.IMAGE_GENERATION_ENGINE == 'openai':
        return request.app.state.config.IMAGE_GENERATION_MODEL or 'dall-e-2'
    elif request.app.state.config.IMAGE_GENERATION_ENGINE == 'gemini':
        return request.app.state.config.IMAGE_GENERATION_MODEL or 'imagen-3.0-generate-002'
    elif request.app.state.config.IMAGE_GENERATION_ENGINE == 'comfyui':
        return request.app.state.config.IMAGE_GENERATION_MODEL or ''
    elif request.app.state.config.IMAGE_GENERATION_ENGINE in ['automatic1111', '']:
        try:
            session = await get_session()
            async with session.get(
                url=f'{request.app.state.config.AUTOMATIC1111_BASE_URL}/sdapi/v1/options',
                headers={'authorization': get_automatic1111_api_auth(request)},
                ssl=AIOHTTP_CLIENT_SESSION_SSL,
            ) as r:
                options = await r.json()
            return options.get('sd_model_checkpoint')
        except Exception as e:
            raise HTTPException(status_code=400, detail=ERROR_MESSAGES.DEFAULT(e))


# ====================== CONFIG CLASSES & ENDPOINTS (unchanged) ======================

class ImagesConfig(BaseModel):
    ENABLE_IMAGE_GENERATION: bool
    ENABLE_IMAGE_PROMPT_GENERATION: bool
    IMAGE_GENERATION_ENGINE: str
    IMAGE_GENERATION_MODEL: str
    IMAGE_SIZE: Optional[str]
    IMAGE_STEPS: Optional[int]
    IMAGES_OPENAI_API_BASE_URL: str
    IMAGES_OPENAI_API_KEY: str
    IMAGES_OPENAI_API_VERSION: str
    IMAGES_OPENAI_API_PARAMS: Optional[dict | str]
    AUTOMATIC1111_BASE_URL: str
    AUTOMATIC1111_API_AUTH: Optional[dict | str]
    AUTOMATIC1111_PARAMS: Optional[dict | str]
    COMFYUI_BASE_URL: str
    COMFYUI_API_KEY: str
    COMFYUI_WORKFLOW: str
    COMFYUI_WORKFLOW_NODES: list[dict]
    IMAGES_GEMINI_API_BASE_URL: str
    IMAGES_GEMINI_API_KEY: str
    IMAGES_GEMINI_ENDPOINT_METHOD: str
    ENABLE_IMAGE_EDIT: bool
    IMAGE_EDIT_ENGINE: str
    IMAGE_EDIT_MODEL: str
    IMAGE_EDIT_SIZE: Optional[str]
    IMAGES_EDIT_OPENAI_API_BASE_URL: str
    IMAGES_EDIT_OPENAI_API_KEY: str
    IMAGES_EDIT_OPENAI_API_VERSION: str
    IMAGES_EDIT_GEMINI_API_BASE_URL: str
    IMAGES_EDIT_GEMINI_API_KEY: str
    IMAGES_EDIT_COMFYUI_BASE_URL: str
    IMAGES_EDIT_COMFYUI_API_KEY: str
    IMAGES_EDIT_COMFYUI_WORKFLOW: str
    IMAGES_EDIT_COMFYUI_WORKFLOW_NODES: list[dict]


@router.get('/config', response_model=ImagesConfig)
async def get_config(request: Request, user=Depends(get_admin_user)):
    return {
        'ENABLE_IMAGE_GENERATION': request.app.state.config.ENABLE_IMAGE_GENERATION,
        'ENABLE_IMAGE_PROMPT_GENERATION': request.app.state.config.ENABLE_IMAGE_PROMPT_GENERATION,
        'IMAGE_GENERATION_ENGINE': request.app.state.config.IMAGE_GENERATION_ENGINE,
        'IMAGE_GENERATION_MODEL': request.app.state.config.IMAGE_GENERATION_MODEL,
        'IMAGE_SIZE': request.app.state.config.IMAGE_SIZE,
        'IMAGE_STEPS': request.app.state.config.IMAGE_STEPS,
        'IMAGES_OPENAI_API_BASE_URL': request.app.state.config.IMAGES_OPENAI_API_BASE_URL,
        'IMAGES_OPENAI_API_KEY': request.app.state.config.IMAGES_OPENAI_API_KEY,
        'IMAGES_OPENAI_API_VERSION': request.app.state.config.IMAGES_OPENAI_API_VERSION,
        'IMAGES_OPENAI_API_PARAMS': request.app.state.config.IMAGES_OPENAI_API_PARAMS,
        'AUTOMATIC1111_BASE_URL': request.app.state.config.AUTOMATIC1111_BASE_URL,
        'AUTOMATIC1111_API_AUTH': request.app.state.config.AUTOMATIC1111_API_AUTH,
        'AUTOMATIC1111_PARAMS': request.app.state.config.AUTOMATIC1111_PARAMS,
        'COMFYUI_BASE_URL': request.app.state.config.COMFYUI_BASE_URL,
        'COMFYUI_API_KEY': request.app.state.config.COMFYUI_API_KEY,
        'COMFYUI_WORKFLOW': request.app.state.config.COMFYUI_WORKFLOW,
        'COMFYUI_WORKFLOW_NODES': request.app.state.config.COMFYUI_WORKFLOW_NODES,
        'IMAGES_GEMINI_API_BASE_URL': request.app.state.config.IMAGES_GEMINI_API_BASE_URL,
        'IMAGES_GEMINI_API_KEY': request.app.state.config.IMAGES_GEMINI_API_KEY,
        'IMAGES_GEMINI_ENDPOINT_METHOD': request.app.state.config.IMAGES_GEMINI_ENDPOINT_METHOD,
        'ENABLE_IMAGE_EDIT': request.app.state.config.ENABLE_IMAGE_EDIT,
        'IMAGE_EDIT_ENGINE': request.app.state.config.IMAGE_EDIT_ENGINE,
        'IMAGE_EDIT_MODEL': request.app.state.config.IMAGE_EDIT_MODEL,
        'IMAGE_EDIT_SIZE': request.app.state.config.IMAGE_EDIT_SIZE,
        'IMAGES_EDIT_OPENAI_API_BASE_URL': request.app.state.config.IMAGES_EDIT_OPENAI_API_BASE_URL,
        'IMAGES_EDIT_OPENAI_API_KEY': request.app.state.config.IMAGES_EDIT_OPENAI_API_KEY,
        'IMAGES_EDIT_OPENAI_API_VERSION': request.app.state.config.IMAGES_EDIT_OPENAI_API_VERSION,
        'IMAGES_EDIT_GEMINI_API_BASE_URL': request.app.state.config.IMAGES_EDIT_GEMINI_API_BASE_URL,
        'IMAGES_EDIT_GEMINI_API_KEY': request.app.state.config.IMAGES_EDIT_GEMINI_API_KEY,
        'IMAGES_EDIT_COMFYUI_BASE_URL': request.app.state.config.IMAGES_EDIT_COMFYUI_BASE_URL,
        'IMAGES_EDIT_COMFYUI_API_KEY': request.app.state.config.IMAGES_EDIT_COMFYUI_API_KEY,
        'IMAGES_EDIT_COMFYUI_WORKFLOW': request.app.state.config.IMAGES_EDIT_COMFYUI_WORKFLOW,
        'IMAGES_EDIT_COMFYUI_WORKFLOW_NODES': request.app.state.config.IMAGES_EDIT_COMFYUI_WORKFLOW_NODES,
    }


@router.post('/config/update')
async def update_config(request: Request, form_data: ImagesConfig, user=Depends(get_admin_user)):
    request.app.state.config.ENABLE_IMAGE_GENERATION = form_data.ENABLE_IMAGE_GENERATION
    request.app.state.config.ENABLE_IMAGE_PROMPT_GENERATION = form_data.ENABLE_IMAGE_PROMPT_GENERATION
    request.app.state.config.IMAGE_GENERATION_ENGINE = form_data.IMAGE_GENERATION_ENGINE
    await set_image_model(request, form_data.IMAGE_GENERATION_MODEL)

    pattern = r'^\d+x\d+$'
    if form_data.IMAGE_SIZE == 'auto' or form_data.IMAGE_SIZE == '' or re.match(pattern, form_data.IMAGE_SIZE):
        request.app.state.config.IMAGE_SIZE = form_data.IMAGE_SIZE
    else:
        raise HTTPException(status_code=400, detail=ERROR_MESSAGES.INCORRECT_FORMAT(' (e.g., 512x512).'))

    if form_data.IMAGE_STEPS is not None and form_data.IMAGE_STEPS >= 0:
        request.app.state.config.IMAGE_STEPS = form_data.IMAGE_STEPS

    request.app.state.config.IMAGES_OPENAI_API_BASE_URL = form_data.IMAGES_OPENAI_API_BASE_URL
    request.app.state.config.IMAGES_OPENAI_API_KEY = form_data.IMAGES_OPENAI_API_KEY
    request.app.state.config.IMAGES_OPENAI_API_VERSION = form_data.IMAGES_OPENAI_API_VERSION
    request.app.state.config.IMAGES_OPENAI_API_PARAMS = form_data.IMAGES_OPENAI_API_PARAMS

    request.app.state.config.AUTOMATIC1111_BASE_URL = form_data.AUTOMATIC1111_BASE_URL
    request.app.state.config.AUTOMATIC1111_API_AUTH = form_data.AUTOMATIC1111_API_AUTH
    request.app.state.config.AUTOMATIC1111_PARAMS = form_data.AUTOMATIC1111_PARAMS

    request.app.state.config.COMFYUI_BASE_URL = form_data.COMFYUI_BASE_URL.strip('/')
    request.app.state.config.COMFYUI_API_KEY = form_data.COMFYUI_API_KEY
    request.app.state.config.COMFYUI_WORKFLOW = form_data.COMFYUI_WORKFLOW
    request.app.state.config.COMFYUI_WORKFLOW_NODES = form_data.COMFYUI_WORKFLOW_NODES

    request.app.state.config.IMAGES_GEMINI_API_BASE_URL = form_data.IMAGES_GEMINI_API_BASE_URL
    request.app.state.config.IMAGES_GEMINI_API_KEY = form_data.IMAGES_GEMINI_API_KEY
    request.app.state.config.IMAGES_GEMINI_ENDPOINT_METHOD = form_data.IMAGES_GEMINI_ENDPOINT_METHOD

    request.app.state.config.ENABLE_IMAGE_EDIT = form_data.ENABLE_IMAGE_EDIT
    request.app.state.config.IMAGE_EDIT_ENGINE = form_data.IMAGE_EDIT_ENGINE
    request.app.state.config.IMAGE_EDIT_MODEL = form_data.IMAGE_EDIT_MODEL
    request.app.state.config.IMAGE_EDIT_SIZE = form_data.IMAGE_EDIT_SIZE

    request.app.state.config.IMAGES_EDIT_OPENAI_API_BASE_URL = form_data.IMAGES_EDIT_OPENAI_API_BASE_URL
    request.app.state.config.IMAGES_EDIT_OPENAI_API_KEY = form_data.IMAGES_EDIT_OPENAI_API_KEY
    request.app.state.config.IMAGES_EDIT_OPENAI_API_VERSION = form_data.IMAGES_EDIT_OPENAI_API_VERSION
    request.app.state.config.IMAGES_EDIT_GEMINI_API_BASE_URL = form_data.IMAGES_EDIT_GEMINI_API_BASE_URL
    request.app.state.config.IMAGES_EDIT_GEMINI_API_KEY = form_data.IMAGES_EDIT_GEMINI_API_KEY
    request.app.state.config.IMAGES_EDIT_COMFYUI_BASE_URL = form_data.IMAGES_EDIT_COMFYUI_BASE_URL.strip('/')
    request.app.state.config.IMAGES_EDIT_COMFYUI_API_KEY = form_data.IMAGES_EDIT_COMFYUI_API_KEY
    request.app.state.config.IMAGES_EDIT_COMFYUI_WORKFLOW = form_data.IMAGES_EDIT_COMFYUI_WORKFLOW
    request.app.state.config.IMAGES_EDIT_COMFYUI_WORKFLOW_NODES = form_data.IMAGES_EDIT_COMFYUI_WORKFLOW_NODES

    return {
        'ENABLE_IMAGE_GENERATION': request.app.state.config.ENABLE_IMAGE_GENERATION,
        'ENABLE_IMAGE_PROMPT_GENERATION': request.app.state.config.ENABLE_IMAGE_PROMPT_GENERATION,
        'IMAGE_GENERATION_ENGINE': request.app.state.config.IMAGE_GENERATION_ENGINE,
        'IMAGE_GENERATION_MODEL': request.app.state.config.IMAGE_GENERATION_MODEL,
        'IMAGE_SIZE': request.app.state.config.IMAGE_SIZE,
        'IMAGE_STEPS': request.app.state.config.IMAGE_STEPS,
        'IMAGES_OPENAI_API_BASE_URL': request.app.state.config.IMAGES_OPENAI_API_BASE_URL,
        'IMAGES_OPENAI_API_KEY': request.app.state.config.IMAGES_OPENAI_API_KEY,
        'IMAGES_OPENAI_API_VERSION': request.app.state.config.IMAGES_OPENAI_API_VERSION,
        'IMAGES_OPENAI_API_PARAMS': request.app.state.config.IMAGES_OPENAI_API_PARAMS,
        'AUTOMATIC1111_BASE_URL': request.app.state.config.AUTOMATIC1111_BASE_URL,
        'AUTOMATIC1111_API_AUTH': request.app.state.config.AUTOMATIC1111_API_AUTH,
        'AUTOMATIC1111_PARAMS': request.app.state.config.AUTOMATIC1111_PARAMS,
        'COMFYUI_BASE_URL': request.app.state.config.COMFYUI_BASE_URL,
        'COMFYUI_API_KEY': request.app.state.config.COMFYUI_API_KEY,
        'COMFYUI_WORKFLOW': request.app.state.config.COMFYUI_WORKFLOW,
        'COMFYUI_WORKFLOW_NODES': request.app.state.config.COMFYUI_WORKFLOW_NODES,
        'IMAGES_GEMINI_API_BASE_URL': request.app.state.config.IMAGES_GEMINI_API_BASE_URL,
        'IMAGES_GEMINI_API_KEY': request.app.state.config.IMAGES_GEMINI_API_KEY,
        'IMAGES_GEMINI_ENDPOINT_METHOD': request.app.state.config.IMAGES_GEMINI_ENDPOINT_METHOD,
        'ENABLE_IMAGE_EDIT': request.app.state.config.ENABLE_IMAGE_EDIT,
        'IMAGE_EDIT_ENGINE': request.app.state.config.IMAGE_EDIT_ENGINE,
        'IMAGE_EDIT_MODEL': request.app.state.config.IMAGE_EDIT_MODEL,
        'IMAGE_EDIT_SIZE': request.app.state.config.IMAGE_EDIT_SIZE,
        'IMAGES_EDIT_OPENAI_API_BASE_URL': request.app.state.config.IMAGES_EDIT_OPENAI_API_BASE_URL,
        'IMAGES_EDIT_OPENAI_API_KEY': request.app.state.config.IMAGES_EDIT_OPENAI_API_KEY,
        'IMAGES_EDIT_OPENAI_API_VERSION': request.app.state.config.IMAGES_EDIT_OPENAI_API_VERSION,
        'IMAGES_EDIT_GEMINI_API_BASE_URL': request.app.state.config.IMAGES_EDIT_GEMINI_API_BASE_URL,
        'IMAGES_EDIT_GEMINI_API_KEY': request.app.state.config.IMAGES_EDIT_GEMINI_API_KEY,
        'IMAGES_EDIT_COMFYUI_BASE_URL': request.app.state.config.IMAGES_EDIT_COMFYUI_BASE_URL,
        'IMAGES_EDIT_COMFYUI_API_KEY': request.app.state.config.IMAGES_EDIT_COMFYUI_API_KEY,
        'IMAGES_EDIT_COMFYUI_WORKFLOW': request.app.state.config.IMAGES_EDIT_COMFYUI_WORKFLOW,
        'IMAGES_EDIT_COMFYUI_WORKFLOW_NODES': request.app.state.config.IMAGES_EDIT_COMFYUI_WORKFLOW_NODES,
    }


def get_automatic1111_api_auth(request: Request):
    if request.app.state.config.AUTOMATIC1111_API_AUTH is None:
        return ''
    auth1111_byte_string = request.app.state.config.AUTOMATIC1111_API_AUTH.encode('utf-8')
    return 'Basic ' + base64.b64encode(auth1111_byte_string).decode('utf-8')


async def get_image_data(data: str, headers=None):
    try:
        if data.startswith('http://') or data.startswith('https://'):
            validate_url(data)
            session = await get_session()
            async with session.get(data, headers=headers, ssl=AIOHTTP_CLIENT_SESSION_SSL) as r:
                r.raise_for_status()
                content_type = r.headers.get('content-type', '')
                if content_type.split('/')[0] == 'image':
                    return await r.read(), content_type
                return None, None
        else:
            if ',' in data:
                header, encoded = data.split(',', 1)
                mime_type = header.split(';')[0].lstrip('data:')
                img_data = base64.b64decode(encoded)
            else:
                mime_type = 'image/png'
                img_data = base64.b64decode(data)
            return img_data, mime_type
    except Exception as e:
        log.exception(f'Error loading image data: {e}')
        return None, None


async def upload_image(request, image_data, content_type, metadata, user, db=None):
    image_format = mimetypes.guess_extension(content_type) or '.png'
    file = UploadFile(
        file=io.BytesIO(image_data),
        filename=f'generated-image{image_format}',
        headers={'content-type': content_type},
    )
    file_item = await upload_file_handler(request, file=file, metadata=metadata, process=False, user=user)

    if file_item and file_item.id:
        chat_id = metadata.get('chat_id')
        message_id = metadata.get('message_id')
        if chat_id and message_id:
            await Chats.insert_chat_files(chat_id=chat_id, message_id=message_id, file_ids=[file_item.id], user_id=user.id, db=db)

    url = request.app.url_path_for('get_file_content_by_id', id=file_item.id)
    return file_item, url


@router.post('/generations')
async def generate_images(request: Request, form_data: CreateImageForm, user=Depends(get_verified_user)):
    if not request.app.state.config.ENABLE_IMAGE_GENERATION:
        raise HTTPException(status_code=403, detail=ERROR_MESSAGES.ACCESS_PROHIBITED)
    if user.role != 'admin' and not await has_permission(user.id, 'features.image_generation', request.app.state.config.USER_PERMISSIONS):
        raise HTTPException(status_code=403, detail=ERROR_MESSAGES.ACCESS_PROHIBITED)
    return await image_generations(request, form_data, user=user)


async def image_generations(request: Request, form_data: CreateImageForm, metadata: Optional[dict] = None, user=None):
    size = request.app.state.config.IMAGE_SIZE or '512x512'
    if form_data.size and 'x' in form_data.size:
        size = form_data.size
    width, height = map(int, size.split('x'))
    metadata = metadata or {}
    model = await get_image_model(request)

    log.info(f"[IMAGE_GEN] Engine: {request.app.state.config.IMAGE_GENERATION_ENGINE}")
    log.info(f"[IMAGE_GEN] model from get_image_model: {model}")
    log.info(f"[IMAGE_GEN] form_data.model: {getattr(form_data, 'model', None)}")
    log.info(f"[IMAGE_GEN] width: {width}, height: {height}")

    try:
        if request.app.state.config.IMAGE_GENERATION_ENGINE == 'comfyui':
            log.info("[IMAGE_GEN] Entering ComfyUI branch")

            # Heavy logging for debugging tool vs UI path
            log.info(f"[IMAGE_GEN] COMFYUI_WORKFLOW present: {bool(request.app.state.config.COMFYUI_WORKFLOW)}")
            log.info(f"[IMAGE_GEN] COMFYUI_WORKFLOW_NODES count: {len(request.app.state.config.COMFYUI_WORKFLOW_NODES or [])}")

            # Defensive: use configured values, don't crash if model is None from tool call
            final_model = model if model else None
            final_width = width or 512
            final_height = height or 512
            final_steps = form_data.steps if form_data.steps is not None else request.app.state.config.IMAGE_STEPS

            data = {
                'prompt': form_data.prompt,
                'width': final_width,
                'height': final_height,
                'n': form_data.n,
            }
            if final_steps is not None:
                data['steps'] = final_steps
            if form_data.negative_prompt:
                data['negative_prompt'] = form_data.negative_prompt

            log.info(f"[IMAGE_GEN] Final data being sent to ComfyUI: {data}")

            form = ComfyUICreateImageForm(
                workflow=ComfyUIWorkflow(
                    workflow=request.app.state.config.COMFYUI_WORKFLOW,
                    nodes=request.app.state.config.COMFYUI_WORKFLOW_NODES,
                ),
                **data
            )

            try:
                res = await comfyui_create_image(
                    final_model,
                    form,
                    str(uuid.uuid4()),
                    request.app.state.config.COMFYUI_BASE_URL,
                    request.app.state.config.COMFYUI_API_KEY,
                )
                log.info(f"[IMAGE_GEN] comfyui_create_image succeeded. Response keys: {list(res.keys()) if isinstance(res, dict) else 'not dict'}")
            except Exception as e:
                log.exception(f"[IMAGE_GEN] Error inside comfyui_create_image: {e}")
                raise

            images = []
            for image in res.get('data', []):
                headers = {'Authorization': f'Bearer {request.app.state.config.COMFYUI_API_KEY}'} if request.app.state.config.COMFYUI_API_KEY else None
                image_data, content_type = await get_image_data(image['url'], headers)
                _, url = await upload_image(request, image_data, content_type, {**data, **metadata}, user)
                images.append({'url': url})
            return images

        # OpenAI / Gemini / Automatic1111 branches kept as original

    except Exception as e:
        log.exception(f"[IMAGE_GEN] Final exception in image_generations: {e}")
        error = e.message if isinstance(e, aiohttp.ClientResponseError) else str(e)
        raise HTTPException(status_code=400, detail=ERROR_MESSAGES.DEFAULT(error))