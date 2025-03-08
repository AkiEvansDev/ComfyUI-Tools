import { app } from "../../scripts/app.js";
import { aeApi } from "./api.js"

app.registerExtension({
	name: "AE.Seed",
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData.name === "AE.Seed" || nodeData.name === "AE.SamplerConfig" || nodeData.name === "AE.SDXLConfig") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated ? onNodeCreated.apply(this) : undefined;

				const widget = this.widgets.find((w) => w.name === "seed_value");
				if (widget) {
					widget.node = this;
					widget.callback = function (v) {
						aeApi.resetNode(this.node.id);
					};
				}
			};

			const onExecuted = nodeType.prototype.onExecuted;
			nodeType.prototype.onExecuted = function ({ seed }) {
				const r = onExecuted ? onExecuted.apply(this, arguments) : undefined;

				const widget = this.widgets.find((w) => w.name === "seed_value");
				if (widget) {
					widget.value = seed[0];
				}
			};
		}
	},
});