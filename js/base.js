import { app } from "../../scripts/app.js";

const adjustMouseEvent = LGraphCanvas.prototype.adjustMouseEvent;
LGraphCanvas.prototype.adjustMouseEvent = function (e) {
    adjustMouseEvent.apply(this, [...arguments]);
    app.lastAdjustedMouseEvent = e;
};

export function getSlotLinks(inputOrOutput) {
    var _a;
    const links = [];
    if (!inputOrOutput) {
        return links;
    }
    if ((_a = inputOrOutput.links) === null || _a === void 0 ? void 0 : _a.length) {
        const output = inputOrOutput;
        for (const linkId of output.links || []) {
            const link = app.graph.links[linkId];
            if (link) {
                links.push({ id: linkId, link: link });
            }
        }
    }
    if (inputOrOutput.link) {
        const input = inputOrOutput;
        const link = app.graph.links[input.link];
        if (link) {
            links.push({ id: input.link, link: link });
        }
    }
    return links;
}

function getTypeFromSlot(slot, dir, skipSelf = false) {
    let graph = app.graph;
    let type = slot === null || slot === void 0 ? void 0 : slot.type;
    if (!skipSelf && type != null && type != "*") {
        return { type: type, label: slot === null || slot === void 0 ? void 0 : slot.label, name: slot === null || slot === void 0 ? void 0 : slot.name };
    }
    const links = getSlotLinks(slot);
    for (const link of links) {
        const connectedId = dir == "OUTPUT" ? link.link.target_id : link.link.origin_id;
        const connectedSlotNum = dir == "OUTPUT" ? link.link.target_slot : link.link.origin_slot;
        const connectedNode = graph.getNodeById(connectedId);
        const connectedSlots = dir === "OUTPUT" ? connectedNode.inputs : connectedNode.outputs;
        let connectedSlot = connectedSlots[connectedSlotNum];
        if ((connectedSlot === null || connectedSlot === void 0 ? void 0 : connectedSlot.type) != null && (connectedSlot === null || connectedSlot === void 0 ? void 0 : connectedSlot.type) != "*") {
            return {
                type: connectedSlot.type,
                label: connectedSlot === null || connectedSlot === void 0 ? void 0 : connectedSlot.label,
                name: connectedSlot === null || connectedSlot === void 0 ? void 0 : connectedSlot.name,
            };
        }
        else if ((connectedSlot === null || connectedSlot === void 0 ? void 0 : connectedSlot.type) == "*") {
            return followConnectionUntilType(connectedNode, dir);
        }
    }
    return null;
}

export function followConnectionUntilType(node, dir, slotNum, skipSelf = false) {
    const slots = dir === "OUTPUT" ? node.outputs : node.inputs;
    if (!slots || !slots.length) {
        return null;
    }
    let type = null;
    if (slotNum) {
        if (!slots[slotNum]) {
            return null;
        }
        type = getTypeFromSlot(slots[slotNum], dir, skipSelf);
    }
    else {
        for (const slot of slots) {
            type = getTypeFromSlot(slot, dir, skipSelf);
            if (type) {
                break;
            }
        }
    }
    return type;
}

export function removeUnusedInputsFromEnd(node, minNumber = 1, nameMatch) {
    var _a;
    for (let i = node.inputs.length - 1; i >= minNumber; i--) {
        if (!((_a = node.inputs[i]) === null || _a === void 0 ? void 0 : _a.link)) {
            if (!nameMatch || nameMatch.test(node.inputs[i].name)) {
                node.removeInput(i);
            }
            continue;
        }
        break;
    }
}

export function moveArrayItem(arr, itemOrFrom, to) {
    const from = typeof itemOrFrom === "number" ? itemOrFrom : arr.indexOf(itemOrFrom);
    arr.splice(to, 0, arr.splice(from, 1)[0]);
}

export function removeArrayItem(arr, itemOrIndex) {
    const index = typeof itemOrIndex === "number" ? itemOrIndex : arr.indexOf(itemOrIndex);
    arr.splice(index, 1);
}

export class BaseWidget {
    constructor(name) {
        this.last_y = 0;
        this.mouseDowned = null;
        this.isMouseDownedAndOver = false;
        this.hitAreas = {};
        this.downedHitAreasForMove = [];
        this.name = name;
    }
    clickWasWithinBounds(pos, bounds) {
        let xStart = bounds[0];
        let xEnd = xStart + (bounds.length > 2 ? bounds[2] : bounds[1]);
        const clickedX = pos[0] >= xStart && pos[0] <= xEnd;
        if (bounds.length === 2) {
            return clickedX;
        }
        return clickedX && pos[1] >= bounds[1] && pos[1] <= bounds[1] + bounds[3];
    }
    mouse(event, pos, node) {
        var _a, _b, _c;
        const canvas = app.canvas;
        if (event.type == "pointerdown") {
            this.mouseDowned = [...pos];
            this.isMouseDownedAndOver = true;
            this.downedHitAreasForMove.length = 0;
            let anyHandled = false;
            for (const part of Object.values(this.hitAreas)) {
                if ((part.onDown || part.onMove) && this.clickWasWithinBounds(pos, part.bounds)) {
                    if (part.onMove) {
                        this.downedHitAreasForMove.push(part);
                    }
                    if (part.onDown) {
                        const thisHandled = part.onDown.apply(this, [event, pos, node, part]);
                        anyHandled = anyHandled || thisHandled == true;
                    }
                }
            }
            return (_a = this.onMouseDown(event, pos, node)) !== null && _a !== void 0 ? _a : anyHandled;
        }
        if (event.type == "pointerup") {
            if (!this.mouseDowned)
                return true;
            this.downedHitAreasForMove.length = 0;
            this.cancelMouseDown();
            let anyHandled = false;
            for (const part of Object.values(this.hitAreas)) {
                if (part.onUp && this.clickWasWithinBounds(pos, part.bounds)) {
                    const thisHandled = part.onUp.apply(this, [event, pos, node, part]);
                    anyHandled = anyHandled || thisHandled == true;
                }
            }
            return (_b = this.onMouseUp(event, pos, node)) !== null && _b !== void 0 ? _b : anyHandled;
        }
        if (event.type == "pointermove") {
            this.isMouseDownedAndOver = !!this.mouseDowned;
            if (this.mouseDowned &&
                (pos[0] < 15 ||
                    pos[0] > node.size[0] - 15 ||
                    pos[1] < this.last_y ||
                    pos[1] > this.last_y + LiteGraph.NODE_WIDGET_HEIGHT)) {
                this.isMouseDownedAndOver = false;
            }
            for (const part of this.downedHitAreasForMove) {
                part.onMove.apply(this, [event, pos, node, part]);
            }
            return (_c = this.onMouseMove(event, pos, node)) !== null && _c !== void 0 ? _c : true;
        }
        return false;
    }
    cancelMouseDown() {
        this.mouseDowned = null;
        this.isMouseDownedAndOver = false;
        this.downedHitAreasForMove.length = 0;
    }
    onMouseDown(event, pos, node) {
        return;
    }
    onMouseUp(event, pos, node) {
        return;
    }
    onMouseMove(event, pos, node) {
        return;
    }
}

app.registerExtension({
    name: "AE",
    nodeCreated(node) {
        node.shape = 1;
        node.color = "#353535";
        node.bgcolor = "#212121";
    }
});