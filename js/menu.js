import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { $el } from "../../scripts/ui.js";

function showToast(message, duration = 3000) {
	const toast = $el("div.comfy-toast", { textContent: message });
	document.body.appendChild(toast);
	setTimeout(() => {
		toast.classList.add("comfy-toast-fadeout");
		setTimeout(() => toast.remove(), 500);
	}, duration);
}

function internalCustomConfirm(message, confirmMessage, cancelMessage) {
	return new Promise((resolve) => {
		const modalOverlay = document.createElement('div');
		modalOverlay.style.position = 'fixed';
		modalOverlay.style.top = 0;
		modalOverlay.style.left = 0;
		modalOverlay.style.width = '100%';
		modalOverlay.style.height = '100%';
		modalOverlay.style.backgroundColor = 'rgba(0, 0, 0, 0.8)';
		modalOverlay.style.display = 'flex';
		modalOverlay.style.alignItems = 'center';
		modalOverlay.style.justifyContent = 'center';
		modalOverlay.style.zIndex = '1101';

		const modalDialog = document.createElement('div');
		modalDialog.style.backgroundColor = '#333';
		modalDialog.style.padding = '20px';
		modalDialog.style.borderRadius = '4px';
		modalDialog.style.maxWidth = '400px';
		modalDialog.style.width = '80%';
		modalDialog.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.5)';
		modalDialog.style.color = '#fff';

		const modalMessage = document.createElement('p');
		modalMessage.textContent = message;
		modalMessage.style.margin = '0';
		modalMessage.style.padding = '0 0 20px';
		modalMessage.style.wordBreak = 'keep-all';

		const modalButtons = document.createElement('div');
		modalButtons.style.display = 'flex';
		modalButtons.style.justifyContent = 'flex-end';

		const confirmButton = document.createElement('button');
		if (confirmMessage)
			confirmButton.textContent = confirmMessage;
		else
			confirmButton.textContent = 'Confirm';
		confirmButton.style.marginLeft = '10px';
		confirmButton.style.backgroundColor = '#28a745';
		confirmButton.style.color = '#fff';
		confirmButton.style.border = 'none';
		confirmButton.style.padding = '6px 12px';
		confirmButton.style.borderRadius = '4px';
		confirmButton.style.cursor = 'pointer';
		confirmButton.style.fontWeight = 'bold';

		const cancelButton = document.createElement('button');
		if (cancelMessage)
			cancelButton.textContent = cancelMessage;
		else
			cancelButton.textContent = 'Cancel';

		cancelButton.style.marginLeft = '10px';
		cancelButton.style.backgroundColor = '#dc3545';
		cancelButton.style.color = '#fff';
		cancelButton.style.border = 'none';
		cancelButton.style.padding = '6px 12px';
		cancelButton.style.borderRadius = '4px';
		cancelButton.style.cursor = 'pointer';
		cancelButton.style.fontWeight = 'bold';

		const closeModal = () => {
			document.body.removeChild(modalOverlay);
		};

		confirmButton.addEventListener('click', () => {
			closeModal();
			resolve(true);
		});

		cancelButton.addEventListener('click', () => {
			closeModal();
			resolve(false);
		});

		modalButtons.appendChild(confirmButton);
		modalButtons.appendChild(cancelButton);
		modalDialog.appendChild(modalMessage);
		modalDialog.appendChild(modalButtons);
		modalOverlay.appendChild(modalDialog);
		document.body.appendChild(modalOverlay);
	});
}

async function customConfirm(message) {
	try {
		let res = await
			window['app'].extensionManager.dialog
				.confirm({
					title: 'Confirm',
					message: message
				});

		return res;
	}
	catch {
		let res = await internalCustomConfirm(message);
		return res;
	}
}

function rebootAPI() {
	if ('electronAPI' in window) {
		window.electronAPI.restartApp();
		return true;
	}

	customConfirm("Are you sure you'd like to reboot the server?").then((isConfirmed) => {
		if (isConfirmed) {
			try {
				api.fetchApi("/ae/reboot");
			}
			catch (exception) { }
		}
	});

	return false;
}

async function free_models(free_execution_cache) {
	try {
		let mode = "";
		if (free_execution_cache) {
			mode = '{"unload_models": true, "free_memory": true}';
		}
		else {
			mode = '{"unload_models": true}';
		}

		let res = await api.fetchApi(`/free`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: mode
		});

		if (res.status == 200) {
			if (free_execution_cache) {
				showToast("'Models' and 'Execution Cache' have been cleared.", 3000);
			}
			else {
				showToast("Models' have been unloaded.", 3000);
			}
		} else {
			showToast('Unloading of models failed. Installed ComfyUI may be an outdated version.', 5000);
		}
	} catch (error) {
		showToast('An error occurred while trying to unload models.', 5000);
	}
}

app.registerExtension({
	name: "AE.Menu",
	async setup() {
		const menu = document.querySelector(".comfy-menu");
		const separator = document.createElement("hr");

		separator.style.margin = "20px 0";
		separator.style.width = "100%";
		menu.append(separator);

		try {
			let cmGroup = new (await import("../../scripts/ui/components/buttonGroup.js")).ComfyButtonGroup(
				new (await import("../../scripts/ui/components/button.js")).ComfyButton({
					icon: "reload",
					action: () => {
						rebootAPI();
					},
					tooltip: "Restart Server",
					content: "Restart",
					classList: "comfyui-button comfyui-menu-mobile-collapse primary"
				}).element,
				new (await import("../../scripts/ui/components/button.js")).ComfyButton({
					icon: "vacuum-outline",
					action: () => {
						free_models();
					},
					tooltip: "Unload Models"
				}).element,
				new (await import("../../scripts/ui/components/button.js")).ComfyButton({
					icon: "vacuum",
					action: () => {
						free_models(true);
					},
					tooltip: "Free model and node cache"
				}).element
			);

			app.menu?.settingsGroup.element.before(cmGroup.element);
		}
		catch (exception) {
			console.log('ComfyUI is outdated. New style menu based features are disabled.');
		}
	},
});