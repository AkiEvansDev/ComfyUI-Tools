from server import PromptServer
from aiohttp import web
import os
import sys
import folder_paths

dir = os.path.abspath(os.path.join(__file__, "../../autocomplete"))
file = os.path.join(dir, "autocomplete.txt")

reset_registry = {}

@PromptServer.instance.routes.get("/ae/autocomplete")
async def get_autocomplete(request):
    if os.path.isfile(file):
        return web.FileResponse(file)
    return web.Response(status=404)

@PromptServer.instance.routes.get("/ae/loras")
async def get_loras(request):
    loras = folder_paths.get_filename_list("loras")
    return web.json_response(list(loras))

@PromptServer.instance.routes.get("/ae/reboot")
def restart(request):
    try:
        sys.stdout.close_log()
    except Exception:
        pass

    if '__COMFY_CLI_SESSION__' in os.environ:
        with open(os.path.join(os.environ['__COMFY_CLI_SESSION__'] + '.reboot'), 'w'):
            pass

        print("\nRestarting...\n\n")
        exit(0)

    print("\nRestarting... [Legacy Mode]\n\n") 

    sys_argv = sys.argv.copy()
    if '--windows-standalone-build' in sys_argv:
        sys_argv.remove('--windows-standalone-build')

    if sys.platform.startswith('win32'):
        cmds = ['"' + sys.executable + '"', '"' + sys_argv[0] + '"'] + sys_argv[1:]
    elif sys_argv[0].endswith("__main__.py"):  # this is a python module
        module_name = os.path.basename(os.path.dirname(sys_argv[0]))
        cmds = [sys.executable, '-m', module_name] + sys_argv[1:]
    else:
        cmds = [sys.executable] + sys_argv

    print(f"Command: {cmds}", flush=True)

    return os.execv(sys.executable, cmds)

@PromptServer.instance.routes.post("/ae/reset/{node_id}")
async def get_autocomplete(request):
    try:
        node_id = request.match_info["node_id"]
        print(reset_registry[node_id])
        if node_id in reset_registry:
            reset_registry[node_id] = True
            return web.json_response(status=200)
        else:
            return web.json_response(dict(error="Node not found"), status=404)
    except Exception as e:
        return web.json_response(dict(error=str(e)), status=500)