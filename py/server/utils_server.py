import os
from aiohttp import web

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_WEB = os.path.abspath(f'{THIS_DIR}/../../web/')

def get_param(request, param, default=None):
    return request.rel_url.query[param] if param in request.rel_url.query else default

def is_param_falsy(request, param):
    val = get_param(request, param)
    return val is not None and (val == "0" or val.upper() == "FALSE")

def is_param_truthy(request, param):
    val = get_param(request, param)
    return val is not None and not is_param_falsy(request, param)

def set_default_page_resources(path, routes):
    @routes.get(f'/rgthree/{path}/{{file}}')
    async def get_resource(request):
        return web.FileResponse(os.path.join(DIR_WEB, path, request.match_info['file']))

    @routes.get(f'/rgthree/{path}/{{subdir}}/{{file}}')
    async def get_resource_subdir(request):
        return web.FileResponse(os.path.join(DIR_WEB, path, request.match_info['subdir'], request.match_info['file']))

def set_default_page_routes(path, routes):
    @routes.get(f'/rgthree/{path}')
    async def get_path_redir(request):
        raise web.HTTPFound(f'{request.path}/')

    @routes.get(f'/rgthree/{path}/')
    async def get_path_index(request):
        html = ''
        with open(os.path.join(DIR_WEB, path, 'index.html'), 'r', encoding='UTF-8') as file:
            html = file.read()
        return web.Response(text=html, content_type='text/html')

    set_default_page_resources(path, routes)
