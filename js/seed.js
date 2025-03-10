import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { aeApi } from "./api.js"

function aeSeedNodeFeedbackHandler(event) {
	let nodes = app.graph._nodes_by_id;
	let node = nodes[event.detail.node_id];
	if (node) {
		const widget = node.widgets.find((w) => w.name === "seed_value");
		if (widget) {
			widget.value = event.detail.seed;
		}
	}
}

api.addEventListener("ae-seed-node-feedback", aeSeedNodeFeedbackHandler);

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
		}
	},
});