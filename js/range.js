import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { aeApi } from "./api.js"

function aeRangeNodeUpdate(event) {
	let nodes = app.graph._nodes_by_id;
	let node = nodes[event.detail.node_id];
	if (node) {
		const widget = node.widgets.find((w) => w.name === "current");
		if (widget) {
			widget.value = event.detail.current;
		}
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

function aeXYRangeNodeUpdate(event) {
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

api.addEventListener("ae-range-node-feedback", aeRangeNodeUpdate);
api.addEventListener("ae-xy-range-node-feedback", aeXYRangeNodeUpdate);

function reset(type, node) {
	aeApi.resetNode(node.id);

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
		if (nodeData.name === "AE.Range" || nodeData.name === "AE.XYRange" || nodeData.name === "AE.RangeList") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated ? onNodeCreated.apply(this) : undefined;

				if (nodeData.name === "AE.Range" || nodeData.name === "AE.RangeList") {
					const widget = this.widgets.find((w) => w.name === "current");
					if (widget) {
						widget.node = this;
						widget.callback = function (v) {
							aeApi.resetNode(this.node.id);
						};
					}
				}
				else if (nodeData.name === "AE.XYRange") {
					const widgetX = this.widgets.find((w) => w.name === "x");
					if (widgetX) {
						widgetX.node = this;
						widgetX.callback = function (v) {
							aeApi.resetNode(this.node.id);
						};
					}
					const widgetY = this.widgets.find((w) => w.name === "y");
					if (widgetY) {
						widgetY.node = this;
						widgetY.callback = function (v) {
							aeApi.resetNode(this.node.id);
						};
					}
				}

				if (nodeData.name != "AE.RangeList") {
					this.addWidget("button", "Queue Full", "QueueButton", (source, canvas, node, pos, event) => {
						queue(nodeData.name, node);
					});

					this.addWidget("button", "Reset Range", "ResetButton", (source, canvas, node, pos, event) => {
						reset(nodeData.name, node);
					});
				}
			};
		}
    },
});