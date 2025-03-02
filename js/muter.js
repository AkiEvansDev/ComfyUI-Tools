import { app } from "../../scripts/app.js";
import { BaseVirtualNode } from "./base.js";
import { SERVICE as FAST_GROUPS_SERVICE } from "./services/fast_groups_service.js";
import { fitString, drawNodeWidget } from "./draw.js";

const PROPERTY_SORT = "sort";
const PROPERTY_SORT_CUSTOM_ALPHA = "customSortAlphabet";
const PROPERTY_MATCH_TITLE = "matchTitle";
const PROPERTY_SHOW_NAV = "showNav";
const PROPERTY_RESTRICTION = "toggleRestriction";

export class BaseFastGroupsModeChanger extends BaseVirtualNode {
    constructor(title) {
        super(title);
        this.debouncerTempWidth = 0;
        this.tempSize = null;
        this.serialize_widgets = false;
        this.properties[PROPERTY_MATCH_TITLE] = "";
        this.properties[PROPERTY_SHOW_NAV] = true;
        this.properties[PROPERTY_SORT] = "position";
        this.properties[PROPERTY_SORT_CUSTOM_ALPHA] = "";
        this.properties[PROPERTY_RESTRICTION] = "default";
    }
    configure(info) {
        var _a;
        if ((_a = info.outputs) === null || _a === void 0 ? void 0 : _a.length) {
            info.outputs.length = 1;
        }
        super.configure(info);
    }
    onAdded(graph) {
        FAST_GROUPS_SERVICE.addFastGroupNode(this);
    }
    onRemoved() {
        FAST_GROUPS_SERVICE.removeFastGroupNode(this);
    }
    refreshWidgets() {
        var _a, _b, _c, _d, _e, _f, _g, _h;
        const canvas = app.canvas;
        let sort = ((_a = this.properties) === null || _a === void 0 ? void 0 : _a[PROPERTY_SORT]) || "position";
        let customAlphabet = null;
        if (sort === "custom alphabet") {
            const customAlphaStr = (_c = (_b = this.properties) === null || _b === void 0 ? void 0 : _b[PROPERTY_SORT_CUSTOM_ALPHA]) === null || _c === void 0 ? void 0 : _c.replace(/\n/g, "");
            if (customAlphaStr && customAlphaStr.trim()) {
                customAlphabet = customAlphaStr.includes(",")
                    ? customAlphaStr.toLocaleLowerCase().split(",")
                    : customAlphaStr.toLocaleLowerCase().trim().split("");
            }
            if (!(customAlphabet === null || customAlphabet === void 0 ? void 0 : customAlphabet.length)) {
                sort = "alphanumeric";
                customAlphabet = null;
            }
        }
        const groups = [...FAST_GROUPS_SERVICE.getGroups(sort)];
        if (customAlphabet === null || customAlphabet === void 0 ? void 0 : customAlphabet.length) {
            groups.sort((a, b) => {
                let aIndex = -1;
                let bIndex = -1;
                for (const [index, alpha] of customAlphabet.entries()) {
                    aIndex =
                        aIndex < 0 ? (a.title.toLocaleLowerCase().startsWith(alpha) ? index : -1) : aIndex;
                    bIndex =
                        bIndex < 0 ? (b.title.toLocaleLowerCase().startsWith(alpha) ? index : -1) : bIndex;
                    if (aIndex > -1 && bIndex > -1) {
                        break;
                    }
                }
                if (aIndex > -1 && bIndex > -1) {
                    const ret = aIndex - bIndex;
                    if (ret === 0) {
                        return a.title.localeCompare(b.title);
                    }
                    return ret;
                }
                else if (aIndex > -1) {
                    return -1;
                }
                else if (bIndex > -1) {
                    return 1;
                }
                return a.title.localeCompare(b.title);
            });
        }
        let index = 0;
        for (const group of groups) {
            if ((_h = (_g = this.properties) === null || _g === void 0 ? void 0 : _g[PROPERTY_MATCH_TITLE]) === null || _h === void 0 ? void 0 : _h.trim()) {
                try {
                    if (!new RegExp(this.properties[PROPERTY_MATCH_TITLE], "i").exec(group.title)) {
                        continue;
                    }
                }
                catch (e) {
                    console.error(e);
                    continue;
                }
            }
            const widgetName = `Enable ${group.title}`;
            let widget = this.widgets.find((w) => w.name === widgetName);
            if (!widget) {
                this.tempSize = [...this.size];
                widget = this.addCustomWidget({
                    name: "TOGGLE_AND_NAV",
                    label: "",
                    value: false,
                    disabled: false,
                    options: { on: "yes", off: "no" },
                    draw: function (ctx, node, width, posY, height) {
                        var _a;
                        posY = posY + 10;
                        const widgetData = drawNodeWidget(ctx, {
                            width,
                            height,
                            posY,
                        });
                        const showNav = ((_a = node.properties) === null || _a === void 0 ? void 0 : _a[PROPERTY_SHOW_NAV]) !== false;
                        let currentX = widgetData.width - widgetData.margin;
                        if (!widgetData.lowQuality && showNav) {
                            currentX -= 7;
                            const midY = widgetData.posY + widgetData.height * 0.5;
                            ctx.fillStyle = ctx.strokeStyle = "#89A";
                            ctx.lineJoin = "round";
                            ctx.lineCap = "round";
                            const arrow = new Path2D(`M${currentX} ${midY} l -7 6 v -3 h -7 v -6 h 7 v -3 z`);
                            ctx.fill(arrow);
                            ctx.stroke(arrow);
                            currentX -= 14;
                            currentX -= 7;
                            ctx.strokeStyle = widgetData.colorOutline;
                            ctx.stroke(new Path2D(`M ${currentX} ${widgetData.posY} v ${widgetData.height}`));
                        }
                        else if (widgetData.lowQuality && showNav) {
                            currentX -= 28;
                        }
                        currentX -= 7;
                        ctx.fillStyle = this.value ? "#89A" : "#333";
                        ctx.beginPath();
                        const toggleRadius = height * 0.36;
                        ctx.arc(currentX - toggleRadius, posY + height * 0.5, toggleRadius, 0, Math.PI * 2);
                        ctx.fill();
                        currentX -= toggleRadius * 2;
                        if (!widgetData.lowQuality) {
                            currentX -= 4;
                            ctx.textAlign = "right";
                            ctx.fillStyle = this.value ? widgetData.colorText : widgetData.colorTextSecondary;
                            const label = this.label || this.name;
                            const toggleLabelOn = this.options.on || "true";
                            const toggleLabelOff = this.options.off || "false";
                            ctx.fillText(this.value ? toggleLabelOn : toggleLabelOff, currentX, posY + height * 0.7);
                            currentX -= Math.max(ctx.measureText(toggleLabelOn).width, ctx.measureText(toggleLabelOff).width);
                            currentX -= 7;
                            ctx.textAlign = "left";
                            let maxLabelWidth = widgetData.width - widgetData.margin - 10 - (widgetData.width - currentX);
                            if (label != null) {
                                ctx.fillText(fitString(ctx, label, maxLabelWidth), widgetData.margin + 10, posY + height * 0.7);
                            }
                        }
                    },
                    serializeValue(serializedNode, widgetIndex) {
                        return this.value;
                    },
                    mouse(event, pos, node) {
                        var _a, _b, _c;
                        if (event.type == "pointerdown") {
                            if (((_a = node.properties) === null || _a === void 0 ? void 0 : _a[PROPERTY_SHOW_NAV]) !== false &&
                                pos[0] >= node.size[0] - 15 - 28 - 1) {
                                const canvas = app.canvas;
                                const lowQuality = (((_b = canvas.ds) === null || _b === void 0 ? void 0 : _b.scale) || 1) <= 0.5;
                                if (!lowQuality) {
                                    canvas.centerOnNode(group);
                                    const zoomCurrent = ((_c = canvas.ds) === null || _c === void 0 ? void 0 : _c.scale) || 1;
                                    const zoomX = canvas.canvas.width / group._size[0] - 0.02;
                                    const zoomY = canvas.canvas.height / group._size[1] - 0.02;
                                    canvas.setZoom(Math.min(zoomCurrent, zoomX, zoomY), [
                                        canvas.canvas.width / 2,
                                        canvas.canvas.height / 2,
                                    ]);
                                    canvas.setDirty(true, true);
                                }
                            }
                            else {
                                this.value = !this.value;
                                setTimeout(() => {
                                    var _a;
                                    (_a = this.callback) === null || _a === void 0 ? void 0 : _a.call(this, this.value, app.canvas, node, pos, event);
                                }, 20);
                            }
                        }
                        return true;
                    },
                });
                widget.doModeChange = (force, skipOtherNodeCheck) => {
                    var _a, _b, _c;
                    group.recomputeInsideNodes();
                    const hasAnyActiveNodes = group._nodes.some((n) => n.mode === LiteGraph.ALWAYS);
                    let newValue = force != null ? force : !hasAnyActiveNodes;
                    if (skipOtherNodeCheck !== true) {
                        if (newValue && ((_b = (_a = this.properties) === null || _a === void 0 ? void 0 : _a[PROPERTY_RESTRICTION]) === null || _b === void 0 ? void 0 : _b.includes(" one"))) {
                            for (const widget of this.widgets) {
                                widget.doModeChange(false, true);
                            }
                        }
                        else if (!newValue && ((_c = this.properties) === null || _c === void 0 ? void 0 : _c[PROPERTY_RESTRICTION]) === "always one") {
                            newValue = this.widgets.every((w) => !w.value || w === widget);
                        }
                    }
                    for (const node of group._nodes) {
                        node.mode = (newValue ? this.modeOn : this.modeOff);
                    }
                    group._rgthreeHasAnyActiveNode = newValue;
                    widget.value = newValue;
                    app.graph.setDirtyCanvas(true, false);
                };
                widget.callback = () => {
                    widget.doModeChange();
                };
                this.setSize(this.computeSize());
            }
            if (widget.name != widgetName) {
                widget.name = widgetName;
                this.setDirtyCanvas(true, false);
            }
            if (widget.value != group._rgthreeHasAnyActiveNode) {
                widget.value = group._rgthreeHasAnyActiveNode;
                this.setDirtyCanvas(true, false);
            }
            if (this.widgets[index] !== widget) {
                const oldIndex = this.widgets.findIndex((w) => w === widget);
                this.widgets.splice(index, 0, this.widgets.splice(oldIndex, 1)[0]);
                this.setDirtyCanvas(true, false);
            }
            index++;
        }
        while ((this.widgets || [])[index]) {
            this.removeWidget(index++);
        }
    }
    computeSize(out) {
        let size = super.computeSize(out);
        if (this.tempSize) {
            size[0] = Math.max(this.tempSize[0], size[0]);
            size[1] = Math.max(this.tempSize[1], size[1]);
            this.debouncerTempWidth && clearTimeout(this.debouncerTempWidth);
            this.debouncerTempWidth = setTimeout(() => {
                this.tempSize = null;
            }, 32);
        }
        setTimeout(() => {
            app.graph.setDirtyCanvas(true, true);
        }, 16);
        return size;
    }
}

BaseFastGroupsModeChanger["@matchTitle"] = { type: "string" };
BaseFastGroupsModeChanger["@showNav"] = { type: "boolean" };
BaseFastGroupsModeChanger["@sort"] = {
    type: "combo",
    values: ["position", "alphanumeric", "custom alphabet"],
};
BaseFastGroupsModeChanger["@customSortAlphabet"] = { type: "string" };
BaseFastGroupsModeChanger["@toggleRestriction"] = {
    type: "combo",
    values: ["default", "max one", "always one"],
};

export class FastGroupsMuter extends BaseFastGroupsModeChanger {
    constructor(title = "Fast Groups Muter") {
        super(title);
        this.title = "Fast Groups Muter";
        this.type = "AE.FastGroupsMuter";
        this.comfyClass = "AE.FastGroupsMuter";
        this.modeOn = LiteGraph.ALWAYS;
        this.modeOff = LiteGraph.NEVER;
        this.shape = 1;
        this.color = "#353535";
        this.bgcolor = "#212121";
    }
}

app.registerExtension({
    name: "AE.FastGroupsMuter",
    registerCustomNodes() {
        LiteGraph.registerNodeType("AE.FastGroupsMuter", FastGroupsMuter);
        FastGroupsMuter.category = "AE.Tools";
        FastGroupsMuter.comfyClass = "AE.FastGroupsMuter";
    }
});