import { app } from "../../scripts/app.js";
import { followConnectionUntilType, removeUnusedInputsFromEnd } from "./base.js";

function addAnyInput(node, num = 1, type = "*") {
    for (let i = 0; i < num; i++) {
        node.addInput(`${String(node.inputs.length + 1).padStart(2, "0")}`, type);
    }
}

app.registerExtension({
    name: "AE.AnySwitch",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "AE.AnySwitch" || nodeData.name === "AE.AnyTypeSwitch") {
            if (nodeData.name === "AE.AnySwitch") {
                const onNodeCreated = nodeType.prototype.onNodeCreated;
                nodeType.prototype.onNodeCreated = function () {
                    const r = onNodeCreated ? onNodeCreated.apply(this) : undefined;
                    removeUnusedInputsFromEnd(this, 1);
                    addAnyInput(this, 2);
                };
            }

            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
                const r = onConnectionsChange ? onConnectionsChange.apply(this, type, index, connected, link_info) : undefined;

                let connectedType = followConnectionUntilType(this, "INPUT", undefined, true);
                if (!connectedType) {
                    connectedType = followConnectionUntilType(this, "OUTPUT", undefined, true);
                }

                if (nodeData.name === "AE.AnySwitch") {
                    removeUnusedInputsFromEnd(this, 1);
                    addAnyInput(this, 1, connectedType);
                }

                this.nodeType = (connectedType === null || connectedType === void 0 ? void 0 : connectedType.type) || "*";

                for (const input of this.inputs) {
                    input.type = this.nodeType;

                    if (input.link) {
                        const link = app.graph.links.get(input.link);
                        link.type = this.nodeType;
                    }
                }

                for (const output of this.outputs) {
                    output.type = this.nodeType;
                    output.label = Array.isArray(this.nodeType) || this.nodeType.includes(",")
                        ? (connectedType === null || connectedType === void 0 ? void 0 : connectedType.label) || String(this.nodeType)
                        : String(this.nodeType);

                    if (output.link) {
                        const link = app.graph.links.get(output.link);
                        link.type = this.nodeType;
                    }
                }
            };
        }
    },
});