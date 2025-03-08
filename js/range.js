import { app } from "../../scripts/app.js";
import { aeApi } from "./api.js"

function aeRangeNodeUpdate(node, pos, range) {
	if (node) {
		const widget = node.widgets.find((w) => w.name === "current");
		if (widget) {
			widget.value = pos[0];
		}
		const widgetStart = node.widgets.find((w) => w.name === "start");
		if (widgetStart) {
			widgetStart.value = range[0];
		}
		const widgetEnd = node.widgets.find((w) => w.name === "end");
		if (widgetEnd) {
			widgetEnd.value = range[1];
		}
	}
}

function aeXYRangeNodeUpdate(node, pos, range) {
	if (node) {
		const widgetX = node.widgets.find((w) => w.name === "x");
		if (widgetX) {
			widgetX.value = pos[0];
		}
		const widgetY = node.widgets.find((w) => w.name === "y");
		if (widgetY) {
			widgetY.value = pos[1];
		}
		const widgetStartX = node.widgets.find((w) => w.name === "x_start");
		if (widgetStartX) {
			widgetStartX.value = range[0];
		}
		const widgetEndX = node.widgets.find((w) => w.name === "x_end");
		if (widgetEndX) {
			widgetEndX.value = range[1];
		}
		const widgetStartY = node.widgets.find((w) => w.name === "y_start");
		if (widgetStartY) {
			widgetStartY.value = range[2];
		}
		const widgetEndY = node.widgets.find((w) => w.name === "y_end");
		if (widgetEndY) {
			widgetEndY.value = range[3];
		}
	}
}

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
		if (nodeData.name === "AE.Range" || nodeData.name === "AE.XYRange") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated ? onNodeCreated.apply(this) : undefined;

				if (nodeData.name === "AE.Range") {
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

				this.addWidget("button", "Queue Full", "QueueButton", (source, canvas, node, pos, event) => {
					queue(nodeData.name, node);
				});

				this.addWidget("button", "Reset Range", "ResetButton", (source, canvas, node, pos, event) => {
					reset(nodeData.name, node);
				});
			};

			const onExecuted = nodeType.prototype.onExecuted;
			nodeType.prototype.onExecuted = function ({ pos, range }) {
				const r = onExecuted ? onExecuted.apply(this, arguments) : undefined;

				if (nodeData.name === "AE.Range") {
					aeRangeNodeUpdate(this, pos, range);
				}
				else if (nodeData.name === "AE.XYRange") {
					aeXYRangeNodeUpdate(this, pos, range);
				}
			};
		}
    },
});