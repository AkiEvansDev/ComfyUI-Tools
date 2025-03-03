from server import PromptServer
from aiohttp import web
import os
import folder_paths

dir = os.path.abspath(os.path.join(__file__, "../../autocomplete"))
file = os.path.join(dir, "autocomplete.txt")

if not os.path.exists(dir):
    os.mkdir(dir)

@PromptServer.instance.routes.get("/ae/autocomplete")
async def get_autocomplete(request):
    if os.path.isfile(file):
        return web.FileResponse(file)
    return web.Response(status=404)

@PromptServer.instance.routes.get("/ae/loras")
async def get_loras(request):
    loras = folder_paths.get_filename_list("loras")
    return web.json_response(list(loras))