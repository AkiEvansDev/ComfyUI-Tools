//import { app } from "../../scripts/app.js";
//import { $el } from "../../scripts/ui.js";

//const DIRECT_ATTRIBUTE_MAP = {
//    cellpadding: 'cellPadding',
//    cellspacing: 'cellSpacing',
//    colspan: 'colSpan',
//    frameborder: 'frameBorder',
//    height: 'height',
//    maxlength: 'maxLength',
//    nonce: 'nonce',
//    role: 'role',
//    rowspan: 'rowSpan',
//    type: 'type',
//    usemap: 'useMap',
//    valign: 'vAlign',
//    width: 'width',
//};

//const RGX_STRING_VALID = '[a-z0-9_-]';
//const RGX_TAG = new RegExp(`^([a-z]${RGX_STRING_VALID}*)(\\.|\\[|\\#|$)`, 'i');
//const RGX_ATTR_ID = new RegExp(`#(${RGX_STRING_VALID}+)`, 'gi');
//const RGX_ATTR_CLASS = new RegExp(`(^|\\S)\\.([a-z0-9_\\-\\.]+)`, 'gi');
//const RGX_STRING_CONTENT_TO_SQUARES = '(.*?)(\\[|\\])';
//const RGX_ATTRS_MAYBE_OPEN = new RegExp(`\\[${RGX_STRING_CONTENT_TO_SQUARES}`, 'gi');

//const adjustMouseEvent = LGraphCanvas.prototype.adjustMouseEvent;
//LGraphCanvas.prototype.adjustMouseEvent = function (e) {
//    adjustMouseEvent.apply(this, [...arguments]);
//    app.lastAdjustedMouseEvent = e;
//};

//export function getSlotLinks(inputOrOutput) {
//    var _a;
//    const links = [];
//    if (!inputOrOutput) {
//        return links;
//    }
//    if ((_a = inputOrOutput.links) === null || _a === void 0 ? void 0 : _a.length) {
//        const output = inputOrOutput;
//        for (const linkId of output.links || []) {
//            const link = app.graph.links[linkId];
//            if (link) {
//                links.push({ id: linkId, link: link });
//            }
//        }
//    }
//    if (inputOrOutput.link) {
//        const input = inputOrOutput;
//        const link = app.graph.links[input.link];
//        if (link) {
//            links.push({ id: input.link, link: link });
//        }
//    }
//    return links;
//}

//function getTypeFromSlot(slot, dir, skipSelf = false) {
//    let graph = app.graph;
//    let type = slot === null || slot === void 0 ? void 0 : slot.type;
//    if (!skipSelf && type != null && type != "*") {
//        return { type: type, label: slot === null || slot === void 0 ? void 0 : slot.label, name: slot === null || slot === void 0 ? void 0 : slot.name };
//    }
//    const links = getSlotLinks(slot);
//    for (const link of links) {
//        const connectedId = dir == "OUTPUT" ? link.link.target_id : link.link.origin_id;
//        const connectedSlotNum = dir == "OUTPUT" ? link.link.target_slot : link.link.origin_slot;
//        const connectedNode = graph.getNodeById(connectedId);
//        const connectedSlots = dir === "OUTPUT" ? connectedNode.inputs : connectedNode.outputs;
//        let connectedSlot = connectedSlots[connectedSlotNum];
//        if ((connectedSlot === null || connectedSlot === void 0 ? void 0 : connectedSlot.type) != null && (connectedSlot === null || connectedSlot === void 0 ? void 0 : connectedSlot.type) != "*") {
//            return {
//                type: connectedSlot.type,
//                label: connectedSlot === null || connectedSlot === void 0 ? void 0 : connectedSlot.label,
//                name: connectedSlot === null || connectedSlot === void 0 ? void 0 : connectedSlot.name,
//            };
//        }
//        else if ((connectedSlot === null || connectedSlot === void 0 ? void 0 : connectedSlot.type) == "*") {
//            return followConnectionUntilType(connectedNode, dir);
//        }
//    }
//    return null;
//}

//export function followConnectionUntilType(node, dir, slotNum, skipSelf = false) {
//    const slots = dir === "OUTPUT" ? node.outputs : node.inputs;
//    if (!slots || !slots.length) {
//        return null;
//    }
//    let type = null;
//    if (slotNum) {
//        if (!slots[slotNum]) {
//            return null;
//        }
//        type = getTypeFromSlot(slots[slotNum], dir, skipSelf);
//    }
//    else {
//        for (const slot of slots) {
//            type = getTypeFromSlot(slot, dir, skipSelf);
//            if (type) {
//                break;
//            }
//        }
//    }
//    return type;
//}

//export function removeUnusedInputsFromEnd(node, minNumber = 1, nameMatch) {
//    var _a;
//    for (let i = node.inputs.length - 1; i >= minNumber; i--) {
//        if (!((_a = node.inputs[i]) === null || _a === void 0 ? void 0 : _a.link)) {
//            if (!nameMatch || nameMatch.test(node.inputs[i].name)) {
//                node.removeInput(i);
//            }
//            continue;
//        }
//        break;
//    }
//}

//export function moveArrayItem(arr, itemOrFrom, to) {
//    const from = typeof itemOrFrom === "number" ? itemOrFrom : arr.indexOf(itemOrFrom);
//    arr.splice(to, 0, arr.splice(from, 1)[0]);
//}

//export function removeArrayItem(arr, itemOrIndex) {
//    const index = typeof itemOrIndex === "number" ? itemOrIndex : arr.indexOf(itemOrIndex);
//    arr.splice(index, 1);
//}

//export function dec2hex(dec) {
//    return dec.toString(16).padStart(2, "0");
//}

//export function generateId(length) {
//    const arr = new Uint8Array(length / 2);
//    crypto.getRandomValues(arr);
//    return Array.from(arr, dec2hex).join("");
//}

//export function getResolver(timeout = 5000) {
//    const resolver = {};
//    resolver.id = generateId(8);
//    resolver.completed = false;
//    resolver.resolved = false;
//    resolver.rejected = false;
//    resolver.promise = new Promise((resolve, reject) => {
//        resolver.reject = (e) => {
//            resolver.completed = true;
//            resolver.rejected = true;
//            reject(e);
//        };
//        resolver.resolve = (data) => {
//            resolver.completed = true;
//            resolver.resolved = true;
//            resolve(data);
//        };
//    });
//    resolver.timeout = setTimeout(() => {
//        if (!resolver.completed) {
//            resolver.reject();
//        }
//    }, timeout);
//    return resolver;
//}

//function tryMatch(str, rgx, index = 1) {
//    var _a;
//    let found = '';
//    try {
//        found = ((_a = str.match(rgx)) === null || _a === void 0 ? void 0 : _a[index]) || '';
//    }
//    catch (e) {
//        found = '';
//    }
//    return found;
//}

//export function getSelectorTag(str) {
//    return tryMatch(str, RGX_TAG);
//}

//export function getSelectorAttributes(selector) {
//    RGX_ATTRS_MAYBE_OPEN.lastIndex = 0;
//    let attrs = [];
//    let result;
//    while (result = RGX_ATTRS_MAYBE_OPEN.exec(selector)) {
//        let attr = result[0];
//        if (attr.endsWith(']')) {
//            attrs.push(attr);
//        }
//        else {
//            attr = result[0]
//                + getOpenAttributesRecursive(selector.substr(RGX_ATTRS_MAYBE_OPEN.lastIndex), 2);
//            RGX_ATTRS_MAYBE_OPEN.lastIndex += (attr.length - result[0].length);
//            attrs.push(attr);
//        }
//    }
//    return attrs;
//}

//export function localAssertNotFalsy(input, errorMsg = `Input is not of type.`) {
//    if (input == null) {
//        throw new Error(errorMsg);
//    }
//    return input;
//}

//export function setAttribute(element, attribute, value) {
//    let isRemoving = value == null;
//    if (attribute === 'default') {
//        attribute = RGX_DEFAULT_VALUE_PROP.test(element.nodeName) ? 'value' : 'text';
//    }
//    if (attribute === 'text') {
//        empty(element).appendChild(createText(value != null ? String(value) : ''));
//    }
//    else if (attribute === 'html') {
//        empty(element).innerHTML += value != null ? String(value) : '';
//    }
//    else if (attribute == 'style') {
//        if (typeof value === 'string') {
//            element.style.cssText = isRemoving ? '' : (value != null ? String(value) : '');
//        }
//        else {
//            for (const [styleKey, styleValue] of Object.entries(value)) {
//                element.style[styleKey] = styleValue;
//            }
//        }
//    }
//    else if (attribute == 'events') {
//        for (const [key, fn] of Object.entries(value)) {
//            addEvent(element, key, fn);
//        }
//    }
//    else if (attribute === 'parent') {
//        value.appendChild(element);
//    }
//    else if (attribute === 'child' || attribute === 'children') {
//        if (typeof value === 'string' && /^\[[^\[\]]+\]$/.test(value)) {
//            const parseable = value.replace(/^\[([^\[\]]+)\]$/, '["$1"]').replace(/,/g, '","');
//            try {
//                const parsed = JSON.parse(parseable);
//                value = parsed;
//            }
//            catch (e) {
//                console.error(e);
//            }
//        }
//        if (attribute === 'children') {
//            empty(element);
//        }
//        let children = value instanceof Array ? value : [value];
//        for (let child of children) {
//            child = getChild(child);
//            if (child instanceof Node) {
//                if (element instanceof HTMLTemplateElement) {
//                    element.content.appendChild(child);
//                }
//                else {
//                    element.appendChild(child);
//                }
//            }
//        }
//    }
//    else if (attribute == 'for') {
//        element.htmlFor = value != null ? String(value) : '';
//        if (isRemoving) {
//            element.removeAttribute('for');
//        }
//    }
//    else if (attribute === 'class' || attribute === 'className' || attribute === 'classes') {
//        element.className = isRemoving ? '' : Array.isArray(value) ? value.join(' ') : String(value);
//    }
//    else if (attribute === 'dataset') {
//        if (typeof value !== 'object') {
//            console.error('Expecting an object for dataset');
//            return;
//        }
//        for (const [key, val] of Object.entries(value)) {
//            element.dataset[key] = String(val);
//        }
//    }
//    else if (attribute.startsWith('on') && typeof value === 'function') {
//        element.addEventListener(attribute.substring(2), value);
//    }
//    else if (['checked', 'disabled', 'readonly', 'required', 'selected'].includes(attribute)) {
//        element[attribute] = !!value;
//        if (!value) {
//            element.removeAttribute(attribute);
//        }
//        else {
//            element.setAttribute(attribute, attribute);
//        }
//    }
//    else if (DIRECT_ATTRIBUTE_MAP.hasOwnProperty(attribute)) {
//        if (isRemoving) {
//            element.removeAttribute(DIRECT_ATTRIBUTE_MAP[attribute]);
//        }
//        else {
//            element.setAttribute(DIRECT_ATTRIBUTE_MAP[attribute], String(value));
//        }
//    }
//    else if (isRemoving) {
//        element.removeAttribute(attribute);
//    }
//    else {
//        let oldVal = element.getAttribute(attribute);
//        if (oldVal !== value) {
//            element.setAttribute(attribute, String(value));
//        }
//    }
//}

//export function setAttributes(element, data) {
//    let attr;
//    for (attr in data) {
//        if (data.hasOwnProperty(attr)) {
//            setAttribute(element, attr, data[attr]);
//        }
//    }
//}

//export function getHtmlFragment(value) {
//    if (value.match(/^\s*<.*?>[\s\S]*<\/[a-z0-9]+>\s*$/)) {
//        return document.createRange().createContextualFragment(value.trim());
//    }
//    return null;
//}

//export function createText(text) {
//    return document.createTextNode(text);
//}

//export function empty(element) {
//    while (element.firstChild) {
//        element.removeChild(element.firstChild);
//    }
//    return element;
//}

//export function createElement(selectorOrMarkup, attrs) {
//    const frag = getHtmlFragment(selectorOrMarkup);
//    let element = frag === null || frag === void 0 ? void 0 : frag.firstElementChild;
//    let selector = "";
//    if (!element) {
//        selector = selectorOrMarkup.replace(/[\r\n]\s*/g, "");
//        const tag = getSelectorTag(selector) || "div";
//        element = document.createElement(tag);
//        selector = selector.replace(RGX_TAG, "$2");
//        selector = selector.replace(RGX_ATTR_ID, '[id="$1"]');
//        selector = selector.replace(RGX_ATTR_CLASS, (match, p1, p2) => `${p1}[class="${p2.replace(/\./g, " ")}"]`);
//    }
//    const selectorAttrs = getSelectorAttributes(selector);
//    if (selectorAttrs) {
//        for (const attr of selectorAttrs) {
//            let matches = attr.substring(1, attr.length - 1).split("=");
//            let key = localAssertNotFalsy(matches.shift());
//            let value = matches.join("=");
//            if (value === undefined) {
//                setAttribute(element, key, true);
//            }
//            else {
//                value = value.replace(/^['"](.*)['"]$/, "$1");
//                setAttribute(element, key, value);
//            }
//        }
//    }
//    if (attrs) {
//        setAttributes(element, attrs);
//    }
//    return element;
//}

//export function queryOne(selectors, parent = document) {
//    var _a;
//    return (_a = parent.querySelector(selectors)) !== null && _a !== void 0 ? _a : null;
//}

//export function getUrl(path, baseUrl) {
//    if (baseUrl) {
//        return new URL(path, baseUrl).toString();
//    } else {
//        return new URL("../" + path, import.meta.url).toString();
//    }
//}

//export function addStylesheet(url) {
//    if (url.endsWith(".js")) {
//        url = url.substr(0, url.length - 2) + "css";
//    }
//    $el("link", {
//        parent: document.head,
//        rel: "stylesheet",
//        type: "text/css",
//        href: url.startsWith("http") ? url : getUrl(url),
//    });
//}

//export function isValidConnection(ioA, ioB) {
//    if (!ioA || !ioB) {
//        return false;
//    }
//    const typeA = String(ioA.type);
//    const typeB = String(ioB.type);
//    let isValid = LiteGraph.isValidConnection(typeA, typeB);

//    if (!isValid) {
//        let areCombos =
//            (typeA.includes(",") && typeB === "COMBO") || (typeA === "COMBO" && typeB.includes(","));
//        if (areCombos) {
//            const nameA = ioA.name.toUpperCase().replace("_NAME", "").replace("CKPT", "MODEL");
//            const nameB = ioB.name.toUpperCase().replace("_NAME", "").replace("CKPT", "MODEL");
//            isValid = nameA.includes(nameB) || nameB.includes(nameA);
//        }
//    }
//    return isValid;
//}

//const oldIsValidConnection = LiteGraph.isValidConnection;
//LiteGraph.isValidConnection = function (typeA, typeB) {
//    let isValid = oldIsValidConnection.call(LiteGraph, typeA, typeB);
//    if (!isValid) {
//        typeA = String(typeA);
//        typeB = String(typeB);

//        let areCombos =
//            (typeA.includes(",") && typeB === "COMBO") || (typeA === "COMBO" && typeB.includes(","));
//        isValid = areCombos;
//    }
//    return isValid;
//};

//export class BaseWidget {
//    constructor(name) {
//        this.last_y = 0;
//        this.mouseDowned = null;
//        this.isMouseDownedAndOver = false;
//        this.hitAreas = {};
//        this.downedHitAreasForMove = [];
//        this.name = name;
//    }
//    clickWasWithinBounds(pos, bounds) {
//        let xStart = bounds[0];
//        let xEnd = xStart + (bounds.length > 2 ? bounds[2] : bounds[1]);
//        const clickedX = pos[0] >= xStart && pos[0] <= xEnd;
//        if (bounds.length === 2) {
//            return clickedX;
//        }
//        return clickedX && pos[1] >= bounds[1] && pos[1] <= bounds[1] + bounds[3];
//    }
//    mouse(event, pos, node) {
//        var _a, _b, _c;
//        const canvas = app.canvas;
//        if (event.type == "pointerdown") {
//            this.mouseDowned = [...pos];
//            this.isMouseDownedAndOver = true;
//            this.downedHitAreasForMove.length = 0;
//            let anyHandled = false;
//            for (const part of Object.values(this.hitAreas)) {
//                if ((part.onDown || part.onMove) && this.clickWasWithinBounds(pos, part.bounds)) {
//                    if (part.onMove) {
//                        this.downedHitAreasForMove.push(part);
//                    }
//                    if (part.onDown) {
//                        const thisHandled = part.onDown.apply(this, [event, pos, node, part]);
//                        anyHandled = anyHandled || thisHandled == true;
//                    }
//                }
//            }
//            return (_a = this.onMouseDown(event, pos, node)) !== null && _a !== void 0 ? _a : anyHandled;
//        }
//        if (event.type == "pointerup") {
//            if (!this.mouseDowned)
//                return true;
//            this.downedHitAreasForMove.length = 0;
//            this.cancelMouseDown();
//            let anyHandled = false;
//            for (const part of Object.values(this.hitAreas)) {
//                if (part.onUp && this.clickWasWithinBounds(pos, part.bounds)) {
//                    const thisHandled = part.onUp.apply(this, [event, pos, node, part]);
//                    anyHandled = anyHandled || thisHandled == true;
//                }
//            }
//            return (_b = this.onMouseUp(event, pos, node)) !== null && _b !== void 0 ? _b : anyHandled;
//        }
//        if (event.type == "pointermove") {
//            this.isMouseDownedAndOver = !!this.mouseDowned;
//            if (this.mouseDowned &&
//                (pos[0] < 15 ||
//                    pos[0] > node.size[0] - 15 ||
//                    pos[1] < this.last_y ||
//                    pos[1] > this.last_y + LiteGraph.NODE_WIDGET_HEIGHT)) {
//                this.isMouseDownedAndOver = false;
//            }
//            for (const part of this.downedHitAreasForMove) {
//                part.onMove.apply(this, [event, pos, node, part]);
//            }
//            return (_c = this.onMouseMove(event, pos, node)) !== null && _c !== void 0 ? _c : true;
//        }
//        return false;
//    }
//    cancelMouseDown() {
//        this.mouseDowned = null;
//        this.isMouseDownedAndOver = false;
//        this.downedHitAreasForMove.length = 0;
//    }
//    onMouseDown(event, pos, node) {
//        return;
//    }
//    onMouseUp(event, pos, node) {
//        return;
//    }
//    onMouseMove(event, pos, node) {
//        return;
//    }
//}

//export class BaseNode extends LGraphNode {
//    constructor(title) {
//        super(title);
//        this.isVirtualNode = false;
//        this.removed = false;
//        this.configuring = false;
//        this._tempWidth = 0;
//        this.widgets = this.widgets || [];
//        this.properties = this.properties || {};
//    }
//    configure(info) {
//        this.configuring = true;
//        super.configure(info);
//        for (const w of this.widgets || []) {
//            w.last_y = w.last_y || 0;
//        }
//        this.configuring = false;
//    }
//    async handleAction(action) {
//        action;
//    }
//    removeWidget(widgetOrSlot) {
//        if (typeof widgetOrSlot === "number") {
//            this.widgets.splice(widgetOrSlot, 1);
//        }
//        else if (widgetOrSlot) {
//            const index = this.widgets.indexOf(widgetOrSlot);
//            if (index > -1) {
//                this.widgets.splice(index, 1);
//            }
//        }
//    }
//    onRemoved() {
//        var _a;
//        (_a = super.onRemoved) === null || _a === void 0 ? void 0 : _a.call(this);
//        this.removed = true;
//    }
//}

//export class BaseVirtualNode extends BaseNode {
//    constructor(title) {
//        super(title);
//        this.isVirtualNode = true;
//    }
//}

//app.registerExtension({
//    name: "AE",
//    nodeCreated(node) {
//        node.shape = 1;
//        node.color = "#353535";
//        node.bgcolor = "#212121";
//    }
//});