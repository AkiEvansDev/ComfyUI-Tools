import { api } from "../../scripts/api.js";

function aeSeedNodeFeedbackHandler(event) {
	let nodes = app.graph._nodes_by_id;
	let node = nodes[event.detail.node_id];
	if (node) {
		const widget = node.widgets.find((w) => w.name === "value");
		if (widget) {
			widget.value = event.detail.seed;
		}
	}
}

api.addEventListener("ae-seed-node-feedback", aeSeedNodeFeedbackHandler);