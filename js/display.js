import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
    name: "AE.DisplayAny",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "AE.DisplayAny") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this) : undefined;

                this.showValueWidget = ComfyWidgets["STRING"](this, "output", ["STRING", { multiline: true }], app).widget;
                this.showValueWidget.inputEl.readOnly = true;
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                const r = onExecuted ? onExecuted.apply(this, message) : undefined;
                this.showValueWidget.value = message.text[0];
            };
        }
    },
});
