import { app } from "../../scripts/app.js";
import { BaseVirtualNode, isValidConnection } from "./base.js";
import { addMenuItem } from "./draw.js";
import { getWidgetConfig, mergeIfValid, setWidgetConfig, } from "../../extensions/core/widgetInputs.js";

const CONFIG_WIDTH = 100;
const CONFIG_HEIGHT = 30;

const configDefaultSize = [CONFIG_WIDTH, CONFIG_HEIGHT];
const configResizable = false;

class RerouteService {
    constructor() {
        this.isFastLinking = false;
        this.handledNewRerouteKeypress = false;
        this.connectingData = null;
        this.fastReroutesHistory = [];
    }
    getConnectingData() {
        var _a, _b;
        const oldCanvas = app.canvas;
        if (oldCanvas.connecting_node && oldCanvas.connecting_slot != null && ((_a = oldCanvas.connecting_pos) === null || _a === void 0 ? void 0 : _a.length)) {
            return {
                node: oldCanvas.connecting_node,
                input: oldCanvas.connecting_input,
                output: oldCanvas.connecting_output,
                slot: oldCanvas.connecting_slot,
                pos: [...oldCanvas.connecting_pos],
            };
        }
        const canvas = app.canvas;
        if ((_b = canvas.connecting_links) === null || _b === void 0 ? void 0 : _b.length) {
            const link = canvas.connecting_links[0];
            return {
                node: link.node,
                input: link.input,
                output: link.output,
                slot: link.slot,
                pos: [...link.pos],
            };
        }
        throw new Error("Error, handling linking keydown, but there's no link.");
    }
    setCanvasConnectingData(ctx) {
        var _a, _b;
        const oldCanvas = app.canvas;
        if (oldCanvas.connecting_node && oldCanvas.connecting_slot != null && ((_a = oldCanvas.connecting_pos) === null || _a === void 0 ? void 0 : _a.length)) {
            oldCanvas.connecting_node = ctx.node;
            oldCanvas.connecting_input = ctx.input;
            oldCanvas.connecting_output = ctx.output;
            oldCanvas.connecting_slot = ctx.slot;
            oldCanvas.connecting_pos = ctx.pos;
        }
        const canvas = app.canvas;
        if ((_b = canvas.connecting_links) === null || _b === void 0 ? void 0 : _b.length) {
            const link = canvas.connecting_links[0];
            link.node = ctx.node;
            link.input = ctx.input;
            link.output = ctx.output;
            link.slot = ctx.slot;
            link.pos = ctx.pos;
        }
    }
    handleMoveOrResizeNodeMaybeWhileDragging(node) {
        const data = this.connectingData;
        if (this.isFastLinking && node === (data === null || data === void 0 ? void 0 : data.node)) {
            const entry = this.fastReroutesHistory[this.fastReroutesHistory.length - 1];
            if (entry) {
                data.pos = entry.node.getConnectionPos(!!data.input, 0);
                this.setCanvasConnectingData(data);
            }
        }
    }
    handleRemovedNodeMaybeWhileDragging(node) {
        const currentEntry = this.fastReroutesHistory[this.fastReroutesHistory.length - 1];
        if ((currentEntry === null || currentEntry === void 0 ? void 0 : currentEntry.node) === node) {
            this.setCanvasConnectingData(currentEntry.previous);
            this.fastReroutesHistory.splice(this.fastReroutesHistory.length - 1, 1);
            if (currentEntry.previous.node) {
                app.canvas.selectNode(currentEntry.previous.node);
            }
        }
    }
}
const SERVICE = new RerouteService();

class RerouteNode extends BaseVirtualNode {
    constructor(title = "Reroute") {
        super(title);
        this.title = "Reroute";
        this.isVirtualNode = true;
        this.hideSlotLabels = true;
        this.schedulePromise = null;
        this.properties["showLabel"] = true;
        this.shape = 1;
        this.color = "#353535";
        this.bgcolor = "#353535";
        this.onConstructed();
    }
    onConstructed() {
        var _a;
        this.setResizable((_a = this.properties["resizable"]) !== null && _a !== void 0 ? _a : configResizable);
        this.size = RerouteNode.size;
        this.addInput("", "*");
        this.addOutput("", "*");
        setTimeout(() => this.applyNodeSize(), 20);
    }
    configure(info) {
        var _a, _b, _c;
        if ((_a = info.outputs) === null || _a === void 0 ? void 0 : _a.length) {
            info.outputs.length = 1;
        }
        if ((_b = info.inputs) === null || _b === void 0 ? void 0 : _b.length) {
            info.inputs.length = 1;
        }
        super.configure(info);
        this.configuring = true;
        this.setResizable((_c = this.properties["resizable"]) !== null && _c !== void 0 ? _c : configResizable);
        this.applyNodeSize();
        this.configuring = false;
    }
    setResizable(resizable) {
        this.properties["resizable"] = !!resizable;
        this.resizable = this.properties["resizable"];
    }
    clone() {
        const cloned = super.clone();
        cloned.inputs[0].type = "*";
        cloned.outputs[0].type = "*";
        return cloned;
    }
    onConnectionsChange(type, _slotIndex, connected, _link_info, _ioSlot) {
        if (connected && type === LiteGraph.OUTPUT) {
            const types = new Set(this.outputs[0].links.map((l) => app.graph.links[l].type).filter((t) => t !== "*"));
            if (types.size > 1) {
                const linksToDisconnect = [];
                for (let i = 0; i < this.outputs[0].links.length - 1; i++) {
                    const linkId = this.outputs[0].links[i];
                    const link = app.graph.links[linkId];
                    linksToDisconnect.push(link);
                }
                for (const link of linksToDisconnect) {
                    const node = app.graph.getNodeById(link.target_id);
                    node.disconnectInput(link.target_slot);
                }
            }
        }
        this.scheduleStabilize();
    }
    onDrawForeground(ctx, canvas) {
        var _a, _b, _c, _d;
        if ((_a = this.properties) === null || _a === void 0 ? void 0 : _a["showLabel"]) {
            const low_quality = ((_b = canvas === null || canvas === void 0 ? void 0 : canvas.ds) === null || _b === void 0 ? void 0 : _b.scale) && canvas.ds.scale < 0.6;
            if (low_quality || this.size[0] <= 10) {
                return;
            }
            const fontSize = Math.min(10, (this.size[1] * 0.65) | 0);
            ctx.save();
            ctx.fillStyle = "#888";
            ctx.font = `${fontSize}px Arial`;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(String(this.title && this.title !== RerouteNode.title
                ? this.title
                : ((_d = (_c = this.outputs) === null || _c === void 0 ? void 0 : _c[0]) === null || _d === void 0 ? void 0 : _d.type) || ""), this.size[0] / 2, this.size[1] / 2, this.size[0] - 30);
            ctx.restore();
        }
    }
    findInputSlot(name) {
        return 0;
    }
    findOutputSlot(name) {
        return 0;
    }
    disconnectOutput(slot, targetNode) {
        return super.disconnectOutput(slot, targetNode);
    }
    disconnectInput(slot) {
        return super.disconnectInput(slot);
    }
    scheduleStabilize(ms = 64) {
        if (!this.schedulePromise) {
            this.schedulePromise = new Promise((resolve) => {
                setTimeout(() => {
                    this.schedulePromise = null;
                    this.stabilize();
                    resolve();
                }, ms);
            });
        }
        return this.schedulePromise;
    }
    stabilize() {
        var _a, _b, _c, _d, _e, _f, _g, _h, _j, _k, _l;
        if (this.configuring) {
            return;
        }
        let currentNode = this;
        let updateNodes = [];
        let input = null;
        let inputType = null;
        let inputNode = null;
        let inputNodeOutputSlot = null;
        while (currentNode) {
            updateNodes.unshift(currentNode);
            const linkId = currentNode.inputs[0].link;
            if (linkId !== null) {
                const link = app.graph.links[linkId];
                const node = app.graph.getNodeById(link.origin_id);
                if (!node) {
                    app.graph.removeLink(linkId);
                    currentNode = null;
                    break;
                }
                const type = node.constructor.type;
                if (type === null || type === void 0 ? void 0 : type.includes("Reroute")) {
                    if (node === this) {
                        currentNode.disconnectInput(link.target_slot);
                        currentNode = null;
                    }
                    else {
                        currentNode = node;
                    }
                }
                else {
                    inputNode = node;
                    inputNodeOutputSlot = link.origin_slot;
                    input = (_a = node.outputs[inputNodeOutputSlot]) !== null && _a !== void 0 ? _a : null;
                    inputType = (_b = input === null || input === void 0 ? void 0 : input.type) !== null && _b !== void 0 ? _b : null;
                    break;
                }
            }
            else {
                currentNode = null;
                break;
            }
        }
        const nodes = [this];
        let outputNode = null;
        let outputType = null;
        let outputWidgetConfig = null;
        let outputWidget = null;
        while (nodes.length) {
            currentNode = nodes.pop();
            const outputs = (currentNode.outputs ? currentNode.outputs[0].links : []) || [];
            if (outputs.length) {
                for (const linkId of outputs) {
                    const link = app.graph.links[linkId];
                    if (!link)
                        continue;
                    const node = app.graph.getNodeById(link.target_id);
                    if (!node)
                        continue;
                    const type = node.constructor.type;
                    if (type === null || type === void 0 ? void 0 : type.includes("Reroute")) {
                        nodes.push(node);
                        updateNodes.push(node);
                    }
                    else {
                        const output = (_d = (_c = node.inputs) === null || _c === void 0 ? void 0 : _c[link.target_slot]) !== null && _d !== void 0 ? _d : null;
                        const nodeOutType = output === null || output === void 0 ? void 0 : output.type;
                        if (nodeOutType == null) {
                            console.warn(`Reroute - Connected node ${node.id} does not have type information for ` +
                                `slot ${link.target_slot}. Skipping connection enforcement, but something is odd ` +
                                `with that node.`);
                        }
                        else if (inputType &&
                            inputType !== "*" &&
                            nodeOutType !== "*" &&
                            !isValidConnection(input, output)) {
                            console.warn(`Reroute - Disconnecting connected node's input (${node.id}.${link.target_slot}) (${node.type}) because its type (${String(nodeOutType)}) does not match the reroute type (${String(inputType)})`);
                            node.disconnectInput(link.target_slot);
                        }
                        else {
                            outputType = nodeOutType;
                            outputNode = node;
                            outputWidgetConfig = null;
                            outputWidget = null;
                            if (output === null || output === void 0 ? void 0 : output.widget) {
                                try {
                                    const config = getWidgetConfig(output);
                                    if (!outputWidgetConfig && config) {
                                        outputWidgetConfig = (_e = config[1]) !== null && _e !== void 0 ? _e : {};
                                        outputType = config[0];
                                        if (!outputWidget) {
                                            outputWidget = (_f = outputNode.widgets) === null || _f === void 0 ? void 0 : _f.find((w) => { var _a; return w.name === ((_a = output === null || output === void 0 ? void 0 : output.widget) === null || _a === void 0 ? void 0 : _a.name); });
                                        }
                                        const merged = mergeIfValid(output, [config[0], outputWidgetConfig]);
                                        if (merged.customConfig) {
                                            outputWidgetConfig = merged.customConfig;
                                        }
                                    }
                                }
                                catch (e) {
                                    console.error("Could not propagate widget infor for reroute; maybe ComfyUI updated?");
                                    outputWidgetConfig = null;
                                    outputWidget = null;
                                }
                            }
                        }
                    }
                }
            }
            else {
            }
        }
        const displayType = inputType || outputType || "*";
        const color = LGraphCanvas.link_type_colors[displayType];
        for (const node of updateNodes) {
            node.outputs[0].type = inputType || "*";
            node.__outputType = displayType;
            //node.outputs[0].name = (input === null || input === void 0 ? void 0 : input.name) || "";
            node.size = node.computeSize();
            (_h = (_g = node).applyNodeSize) === null || _h === void 0 ? void 0 : _h.call(_g);
            for (const l of node.outputs[0].links || []) {
                const link = app.graph.links[l];
                if (link) {
                    link.color = color;
                }
            }
            try {
                if (outputWidgetConfig && outputWidget && outputType) {
                    node.inputs[0].widget = { name: "value" };
                    setWidgetConfig(node.inputs[0], [outputType !== null && outputType !== void 0 ? outputType : displayType, outputWidgetConfig], outputWidget);
                }
                else {
                    setWidgetConfig(node.inputs[0], null);
                }
            }
            catch (e) {
                console.error("Could not set widget config for reroute; maybe ComfyUI updated?");
                outputWidgetConfig = null;
                outputWidget = null;
                if ((_j = node.inputs[0]) === null || _j === void 0 ? void 0 : _j.widget) {
                    delete node.inputs[0].widget;
                }
            }
        }
        if (inputNode && inputNodeOutputSlot != null) {
            const links = inputNode.outputs[inputNodeOutputSlot].links;
            for (const l of links || []) {
                const link = app.graph.links[l];
                if (link) {
                    link.color = color;
                }
            }
        }
        (_k = inputNode === null || inputNode === void 0 ? void 0 : inputNode.onConnectionsChainChange) === null || _k === void 0 ? void 0 : _k.call(inputNode);
        (_l = outputNode === null || outputNode === void 0 ? void 0 : outputNode.onConnectionsChainChange) === null || _l === void 0 ? void 0 : _l.call(outputNode);
        app.graph.setDirtyCanvas(true, true);
    }
    setSize(size) {
        const oldSize = [...this.size];
        const newSize = [...size];
        super.setSize(newSize);
        this.properties["size"] = [...this.size];
        this.stabilizeLayout(oldSize, newSize);
    }
    stabilizeLayout(oldSize, newSize) {
        if (newSize[0] === 10 || newSize[1] === 10) {
            const props = this.properties;
            props["connections_layout"] = props["connections_layout"] || ["Left", "Right"];
            const layout = props["connections_layout"];
            props["connections_dir"] = props["connections_dir"] || [-1, -1];
            const dir = props["connections_dir"];
            if (oldSize[0] > 10 && newSize[0] === 10) {
                dir[0] = LiteGraph.DOWN;
                dir[1] = LiteGraph.UP;
                if (layout[0] === "Bottom") {
                    layout[1] = "Top";
                }
                else if (layout[1] === "Top") {
                    layout[0] = "Bottom";
                }
                else {
                    layout[0] = "Top";
                    layout[1] = "Bottom";
                    dir[0] = LiteGraph.UP;
                    dir[1] = LiteGraph.DOWN;
                }
                this.setDirtyCanvas(true, true);
            }
            else if (oldSize[1] > 10 && newSize[1] === 10) {
                dir[0] = LiteGraph.RIGHT;
                dir[1] = LiteGraph.LEFT;
                if (layout[0] === "Right") {
                    layout[1] = "Left";
                }
                else if (layout[1] === "Left") {
                    layout[0] = "Right";
                }
                else {
                    layout[0] = "Left";
                    layout[1] = "Right";
                    dir[0] = LiteGraph.LEFT;
                    dir[1] = LiteGraph.RIGHT;
                }
                this.setDirtyCanvas(true, true);
            }
        }
        SERVICE.handleMoveOrResizeNodeMaybeWhileDragging(this);
    }
    applyNodeSize() {
        this.properties["size"] = this.properties["size"] || RerouteNode.size;
        this.properties["size"] = [
            Number(this.properties["size"][0]),
            Number(this.properties["size"][1]),
        ];
        this.size = this.properties["size"];
        app.graph.setDirtyCanvas(true, true);
    }
    rotate(degrees) {
        const w = this.size[0];
        const h = this.size[1];
        this.properties["connections_layout"] =
            this.properties["connections_layout"] || this.defaultConnectionsLayout;
        const inputDirIndex = LAYOUT_CLOCKWISE.indexOf(this.properties["connections_layout"][0]);
        const outputDirIndex = LAYOUT_CLOCKWISE.indexOf(this.properties["connections_layout"][1]);
        if (degrees == 90 || degrees === -90) {
            if (degrees === -90) {
                this.properties["connections_layout"][0] =
                    LAYOUT_CLOCKWISE[(((inputDirIndex - 1) % 4) + 4) % 4];
                this.properties["connections_layout"][1] =
                    LAYOUT_CLOCKWISE[(((outputDirIndex - 1) % 4) + 4) % 4];
            }
            else {
                this.properties["connections_layout"][0] =
                    LAYOUT_CLOCKWISE[(((inputDirIndex + 1) % 4) + 4) % 4];
                this.properties["connections_layout"][1] =
                    LAYOUT_CLOCKWISE[(((outputDirIndex + 1) % 4) + 4) % 4];
            }
        }
        else if (degrees === 180) {
            this.properties["connections_layout"][0] =
                LAYOUT_CLOCKWISE[(((inputDirIndex + 2) % 4) + 4) % 4];
            this.properties["connections_layout"][1] =
                LAYOUT_CLOCKWISE[(((outputDirIndex + 2) % 4) + 4) % 4];
        }
        this.setSize([h, w]);
    }
    manuallyHandleMove(event) {
        const shortcut = this.shortcuts.move;
        if (shortcut.state) {
            const diffX = Math.round((event.clientX - shortcut.initialMousePos[0]) / 10) * 10;
            const diffY = Math.round((event.clientY - shortcut.initialMousePos[1]) / 10) * 10;
            this.pos[0] = shortcut.initialNodePos[0] + diffX;
            this.pos[1] = shortcut.initialNodePos[1] + diffY;
            this.setDirtyCanvas(true, true);
            SERVICE.handleMoveOrResizeNodeMaybeWhileDragging(this);
        }
    }
    manuallyHandleResize(event) {
        const shortcut = this.shortcuts.resize;
        if (shortcut.state) {
            let diffX = Math.round((event.clientX - shortcut.initialMousePos[0]) / 10) * 10;
            let diffY = Math.round((event.clientY - shortcut.initialMousePos[1]) / 10) * 10;
            diffX *= shortcut.resizeOnSide[0] === LiteGraph.LEFT ? -1 : 1;
            diffY *= shortcut.resizeOnSide[1] === LiteGraph.UP ? -1 : 1;
            const oldSize = [...this.size];
            this.setSize([
                Math.max(10, shortcut.initialNodeSize[0] + diffX),
                Math.max(10, shortcut.initialNodeSize[1] + diffY),
            ]);
            if (shortcut.resizeOnSide[0] === LiteGraph.LEFT && oldSize[0] > 10) {
                this.pos[0] = shortcut.initialNodePos[0] - diffX;
            }
            if (shortcut.resizeOnSide[1] === LiteGraph.UP && oldSize[1] > 10) {
                this.pos[1] = shortcut.initialNodePos[1] - diffY;
            }
            this.setDirtyCanvas(true, true);
        }
    }
    cycleConnection(ioDir) {
        var _a, _b;
        const props = this.properties;
        props["connections_layout"] = props["connections_layout"] || ["Left", "Right"];
        const propIdx = ioDir == IoDirection.INPUT ? 0 : 1;
        const oppositeIdx = propIdx ? 0 : 1;
        let currentLayout = props["connections_layout"][propIdx];
        let oppositeLayout = props["connections_layout"][oppositeIdx];
        if (this.size[0] === 10 || this.size[1] === 10) {
            props["connections_dir"] = props["connections_dir"] || [-1, -1];
            let currentDir = props["connections_dir"][propIdx];
            const options = this.size[0] === 10
                ? currentLayout === "Bottom"
                    ? [LiteGraph.DOWN, LiteGraph.RIGHT, LiteGraph.LEFT]
                    : [LiteGraph.UP, LiteGraph.LEFT, LiteGraph.RIGHT]
                : currentLayout === "Right"
                    ? [LiteGraph.RIGHT, LiteGraph.DOWN, LiteGraph.UP]
                    : [LiteGraph.LEFT, LiteGraph.UP, LiteGraph.DOWN];
            let idx = options.indexOf(currentDir);
            let next = (_a = options[idx + 1]) !== null && _a !== void 0 ? _a : options[0];
            this.properties["connections_dir"][propIdx] = next;
            return;
        }
        let next = currentLayout;
        do {
            let idx = LAYOUT_CLOCKWISE.indexOf(next);
            next = (_b = LAYOUT_CLOCKWISE[idx + 1]) !== null && _b !== void 0 ? _b : LAYOUT_CLOCKWISE[0];
        } while (next === oppositeLayout);
        this.properties["connections_layout"][propIdx] = next;
        this.setDirtyCanvas(true, true);
    }
    onDeselected() {
        var _a;
        (_a = super.onDeselected) === null || _a === void 0 ? void 0 : _a.call(this);
    }
    onRemoved() {
        var _a;
        (_a = super.onRemoved) === null || _a === void 0 ? void 0 : _a.call(this);
        setTimeout(() => {
            SERVICE.handleRemovedNodeMaybeWhileDragging(this);
        }, 32);
    }
}

addMenuItem(RerouteNode, app, {
    name: (node) => { var _a; return `${((_a = node.properties) === null || _a === void 0 ? void 0 : _a["showLabel"]) ? "Hide" : "Show"} Label/Title`; },
    property: "showLabel",
    callback: async (node, value) => {
        app.graph.setDirtyCanvas(true, true);
    },
});
addMenuItem(RerouteNode, app, {
    name: (node) => `${node.resizable ? "No" : "Allow"} Resizing`,
    callback: (node) => {
        node.setResizable(!node.resizable);
        node.size[0] = Math.max(40, node.size[0]);
        node.size[1] = Math.max(30, node.size[1]);
        node.applyNodeSize();
    },
});
addMenuItem(RerouteNode, app, {
    name: "Static Width",
    property: "size",
    subMenuOptions: (() => {
        const options = [];
        for (let w = 8; w > 0; w--) {
            options.push(`${w * 10}`);
        }
        return options;
    })(),
    prepareValue: (value, node) => [Number(value), node.size[1]],
    callback: (node) => {
        node.setResizable(false);
        node.applyNodeSize();
    },
});
addMenuItem(RerouteNode, app, {
    name: "Static Height",
    property: "size",
    subMenuOptions: (() => {
        const options = [];
        for (let w = 8; w > 0; w--) {
            options.push(`${w * 10}`);
        }
        return options;
    })(),
    prepareValue: (value, node) => [node.size[0], Number(value)],
    callback: (node) => {
        node.setResizable(false);
        node.applyNodeSize();
    },
});

app.registerExtension({
    name: "AE.Reroute",
    registerCustomNodes() {
        LiteGraph.registerNodeType("AE.Reroute", RerouteNode);
        RerouteNode.category = "AE.Tools";
        RerouteNode.comfyClass = "AE.Reroute";
        RerouteNode.title_mode = LiteGraph.NO_TITLE;
        RerouteNode.collapsable = false;
        RerouteNode.size = configDefaultSize;
    }
});
