import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function aeRangeNodeFeedbackHandler(event) {
	let nodes = app.graph._nodes_by_id;
	let node = nodes[event.detail.node_id];
	if (node) {
		const widget = node.widgets.find((w) => w.name === "current");
		if (widget) {
			widget.value = event.detail.current;
		}
	}
}

function aeXYRangeNodeFeedbackHandler(event) {
	let nodes = app.graph._nodes_by_id;
	let node = nodes[event.detail.node_id];
	if (node) {
		const widgetX = node.widgets.find((w) => w.name === "x");
		if (widgetX) {
			widgetX.value = event.detail.x;
		}
		const widgetY = node.widgets.find((w) => w.name === "y");
		if (widgetY) {
			widgetY.value = event.detail.y;
		}
	}
}

function aeRangeNodeUpdateHandler(event) {
	let nodes = app.graph._nodes_by_id;
	let node = nodes[event.detail.node_id];
	if (node) {
		const widgetStart = node.widgets.find((w) => w.name === "start");
		if (widgetStart) {
			widgetStart.value = event.detail.start;
		}
		const widgetEnd = node.widgets.find((w) => w.name === "end");
		if (widgetEnd) {
			widgetEnd.value = event.detail.end;
		}
	}
}

function aeXYRangeNodeUpdateHandler(event) {
	let nodes = app.graph._nodes_by_id;
	let node = nodes[event.detail.node_id];
	if (node) {
		const widgetStartX = node.widgets.find((w) => w.name === "x_start");
		if (widgetStartX) {
			widgetStartX.value = event.detail.x_start;
		}
		const widgetEndX = node.widgets.find((w) => w.name === "x_end");
		if (widgetEndX) {
			widgetEndX.value = event.detail.x_end;
		}
		const widgetStartY = node.widgets.find((w) => w.name === "y_start");
		if (widgetStartY) {
			widgetStartY.value = event.detail.y_start;
		}
		const widgetEndY = node.widgets.find((w) => w.name === "y_end");
		if (widgetEndY) {
			widgetEndY.value = event.detail.y_end;
		}
	}
}

api.addEventListener("ae-range-node-feedback", aeRangeNodeFeedbackHandler);
api.addEventListener("ae-xyrange-node-feedback", aeXYRangeNodeFeedbackHandler);

api.addEventListener("ae-range-node-update", aeRangeNodeUpdateHandler);
api.addEventListener("ae-xyrange-node-update", aeXYRangeNodeUpdateHandler);


function reset(type, node) {
	if (type === "AE.Range") {
		const current = node.widgets.find((w) => w.name === "current");
		const start = node.widgets.find((w) => w.name === "start");

		current.value = start.value - 1;
	}
	else if (type === "AE.XYRange") {
		const x = node.widgets.find((w) => w.name === "x");
		const startX = node.widgets.find((w) => w.name === "x_start");
		const y = node.widgets.find((w) => w.name === "y");
		const startY = node.widgets.find((w) => w.name === "y_start");

		x.value = startX.value - 1;
		y.value = startY.value - 1;
	}
}

function queue(type, node) {
	reset(type, node)

	let batch_size = 0;

	if (type === "AE.Range") {
		const start = node.widgets.find((w) => w.name === "start");
		const end = node.widgets.find((w) => w.name === "end");

		batch_size = end.value - start.value + 1;
	}
	else if (type === "AE.XYRange") {
		const startX = node.widgets.find((w) => w.name === "x_start");
		const endX = node.widgets.find((w) => w.name === "x_end");
		const startY = node.widgets.find((w) => w.name === "y_start");
		const endY = node.widgets.find((w) => w.name === "y_end");

		batch_size = (endX.value - startX.value + 1) * (endY.value - startY.value + 1);
	}

	app.queuePrompt(0, batch_size);
}

app.registerExtension({
    name: "AE.Range",
    async beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData.name === "AE.Range" || nodeData.name === "AE.XYRange") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this) : undefined;

				this.addWidget("button", "Queue Full", "QueueButton", (source, canvas, node, pos, event) => {
					queue(nodeData.name, node);
				});

				this.addWidget("button", "Reset Range", "ResetButton", (source, canvas, node, pos, event) => {
					reset(nodeData.name, node);
				});
            };
		}
    },
});