import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "AE.Text",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "AE.Text") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this) : undefined;
                const input = this.widgets.find((w) => w.name === "value").element;

                input.style.border = "0.5px solid " + LiteGraph.WIDGET_OUTLINE_COLOR;
                input.style.borderRadius = "4px";
            };
        }
    },
});