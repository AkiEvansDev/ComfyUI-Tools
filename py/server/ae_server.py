import os
from aiohttp import web
from server import PromptServer

from .utils_server import set_default_page_resources, set_default_page_routes
from .routes_model_info import *

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_WEB = os.path.abspath(f'{THIS_DIR}/../../web/')

routes = PromptServer.instance.routes

set_default_page_resources("comfyui", routes)
set_default_page_resources("common", routes)
set_default_page_resources("lib", routes)

set_default_page_routes("link_fixer", routes)
