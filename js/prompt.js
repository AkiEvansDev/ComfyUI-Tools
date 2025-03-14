import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "AE.SDXLPrompt",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "AE.SDXLPrompt" || nodeData.name === "AE.SDXLPromptWithHires" || nodeData.name === "AE.SDXLRegionalPrompt" || nodeData.name === "AE.SDXLRegionalPromptWithHires") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this) : undefined;

                const targetNames = ["positive", "posture", "background", "style", "positive_style", "negative", "negative_style"];

                targetNames.forEach((name) => {
                    const widget = this.widgets.find((w) => w.name === name);
                    if (widget && widget.element) {
                        widget.element.style.border = "0.5px solid " + LiteGraph.WIDGET_OUTLINE_COLOR;
                        widget.element.style.borderRadius = "4px";
                    }
                });
            };
        }
    },
});