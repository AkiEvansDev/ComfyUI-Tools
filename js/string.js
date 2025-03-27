import { app } from "../../scripts/app.js";
import { removeUnusedInputsFromEnd } from "./base.js";

function addAnyInput(node, num = 1) {
    for (let i = 0; i < num; i++) {
        node.addInput(`${String(node.inputs.filter((i) => i.type == "*").length + 1).padStart(2, "0")}`, "*");
    }
}

app.registerExtension({
    name: "AE.ToStringConcat",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "AE.ToStringConcat") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this) : undefined;

                removeUnusedInputsFromEnd(this, 1, /^\d+$/);
                addAnyInput(this, 2);

                const onConnectionsChange = this.onConnectionsChange;
                this.onConnectionsChange = function (type, index, connected, link_info) {
                    const r = onConnectionsChange ? onConnectionsChange.apply(this, type, index, connected, link_info) : undefined;

                    const oldWidth = this.size[0];

                    removeUnusedInputsFromEnd(this, 1, /^\d+$/);
                    addAnyInput(this);

                    this.setSize([
                        Math.max(oldWidth, this.size[0]),
                        this.size[1]
                    ]);
                };
            };
        }
    },
});