import os
import json
from aiohttp import web

from server import PromptServer
import folder_paths

routes = PromptServer.instance.routes

def _check_valid_model_type(request):
    model_type = request.match_info['type']
    if model_type not in ['loras']:
        return web.json_response({'status': 404, 'error': f'Invalid model type: {model_type}'})

    return None

@routes.get('/ae/api/{type}')
async def api_get_models_list(request):
    if _check_valid_model_type(request):
        return _check_valid_model_type(request)

    model_type = request.match_info['type']
    data = folder_paths.get_filename_list(model_type)

    return web.json_response(list(data))
