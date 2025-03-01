import { app } from "../../scripts/app.js";
import { aeApi } from "./api.js";
import { BaseWidget, moveArrayItem, removeArrayItem } from "./base.js"
import { isLowQuality, fitString, drawTogglePart, drawRoundedRectangle, drawNumberWidgetPart } from "./draw.js"

const PROP_LABEL_SHOW_STRENGTHS = "Show Strengths";
const PROP_LABEL_SHOW_STRENGTHS_STATIC = `@${PROP_LABEL_SHOW_STRENGTHS}`;
const PROP_VALUE_SHOW_STRENGTHS_SINGLE = "Single Strength";
const PROP_VALUE_SHOW_STRENGTHS_SEPARATE = "Separate Model & Clip";

const DEFAULT_LORA_WIDGET_DATA = {
    on: true,
    lora: null,
    strength: 1,
    strengthTwo: null,
};

async function showLoraChooser(event, callback, parentMenu, loras) {
    var _a, _b;
    const canvas = app.canvas;
    if (!loras) {
        loras = ["None", ...(await aeApi.getLoras())];
    }
    new LiteGraph.ContextMenu(loras, {
        event: event,
        parentMenu,
        title: "Choose Lora",
        scale: Math.max(1, (_b = (_a = canvas.ds) === null || _a === void 0 ? void 0 : _a.scale) !== null && _b !== void 0 ? _b : 1),
        className: "dark",
        callback,
    });
}

function removeWidget(node, widgetOrSlot) {
    if (typeof widgetOrSlot === "number") {
        node.widgets.splice(widgetOrSlot, 1);
    }
    else if (widgetOrSlot) {
        const index = node.widgets.indexOf(widgetOrSlot);
        if (index > -1) {
            node.widgets.splice(index, 1);
        }
    }
}

function addNewLoraWidget(node, lora) {
    node.loraWidgetsCounter++;
    const widget = node.addCustomWidget(new LoraLoaderWidget("lora_" + node.loraWidgetsCounter));

    if (lora)
        widget.setLora(lora);

    if (node.addLoraWidget)
        moveArrayItem(node.widgets, widget, node.widgets.indexOf(node.addLoraWidget));

    return widget;
}

function addNonLoraWidgets(node) {
    node.addLoraWidget = node.addWidget("button", "Add Lora", "AddLoraButton", (source, canvas, node, pos, event) => {
        aeApi.getLoras().then((loras) => {
            showLoraChooser(event, (value) => {
                if (typeof value === "string") {
                    if (value !== "NONE") {
                        addNewLoraWidget(node, value);
                        const computed = node.computeSize();
                        node.size[1] = Math.max(node.size[1], computed[1]);
                        node.setDirtyCanvas(true, true);
                    }
                }
            }, null, [...loras]);
        });
    });
}

class LoraLoaderWidget extends BaseWidget {
    constructor(name) {
        super(name);
        this.haveMouseMovedStrength = false;
        this.loraInfoPromise = null;
        this.loraInfo = null;
        this.showModelAndClip = null;
        this.hitAreas = {
            toggle: { bounds: [0, 0], onDown: this.onToggleDown },
            lora: { bounds: [0, 0], onDown: this.onLoraDown },
            strengthDec: { bounds: [0, 0], onDown: this.onStrengthDecDown },
            strengthVal: { bounds: [0, 0], onUp: this.onStrengthValUp },
            strengthInc: { bounds: [0, 0], onDown: this.onStrengthIncDown },
            strengthAny: { bounds: [0, 0], onMove: this.onStrengthAnyMove },
            strengthTwoDec: { bounds: [0, 0], onDown: this.onStrengthTwoDecDown },
            strengthTwoVal: { bounds: [0, 0], onUp: this.onStrengthTwoValUp },
            strengthTwoInc: { bounds: [0, 0], onDown: this.onStrengthTwoIncDown },
            strengthTwoAny: { bounds: [0, 0], onMove: this.onStrengthTwoAnyMove },
        };
        this._value = {
            on: true,
            lora: null,
            strength: 1,
            strengthTwo: null,
        };
    }
    set value(v) {
        this._value = v;
        if (typeof this._value !== "object") {
            this._value = { ...DEFAULT_LORA_WIDGET_DATA };
            if (this.showModelAndClip) {
                this._value.strengthTwo = this._value.strength;
            }
        }
    }
    get value() {
        return this._value;
    }
    setLora(lora) {
        this._value.lora = lora;
    }
    draw(ctx, node, w, posY, height) {
        var _b, _c, _d, _e, _f, _g, _h, _j, _k, _l, _m, _o, _p;
        let currentShowModelAndClip = node.properties[PROP_LABEL_SHOW_STRENGTHS] === PROP_VALUE_SHOW_STRENGTHS_SEPARATE;
        if (this.showModelAndClip !== currentShowModelAndClip) {
            let oldShowModelAndClip = this.showModelAndClip;
            this.showModelAndClip = currentShowModelAndClip;
            if (this.showModelAndClip) {
                if (oldShowModelAndClip != null) {
                    this.value.strengthTwo = (_b = this.value.strength) !== null && _b !== void 0 ? _b : 1;
                }
            }
            else {
                this.value.strengthTwo = null;
                this.hitAreas.strengthTwoDec.bounds = [0, -1];
                this.hitAreas.strengthTwoVal.bounds = [0, -1];
                this.hitAreas.strengthTwoInc.bounds = [0, -1];
                this.hitAreas.strengthTwoAny.bounds = [0, -1];
            }
        }
        ctx.save();
        const margin = 10;
        const innerMargin = margin * 0.33;
        const lowQuality = isLowQuality();
        const midY = posY + height * 0.5;
        let posX = margin;
        drawRoundedRectangle(ctx, { posX, posY, height, width: node.size[0] - margin * 2 });
        this.hitAreas.toggle.bounds = drawTogglePart(ctx, { posX, posY, height, value: this.value.on });
        posX += this.hitAreas.toggle.bounds[1] + innerMargin;
        if (lowQuality) {
            ctx.restore();
            return;
        }
        if (!this.value.on) {
            ctx.globalAlpha = app.canvas.editor_alpha * 0.4;
        }
        ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
        let rposX = node.size[0] - margin - innerMargin - innerMargin;
        const strengthValue = this.showModelAndClip
            ? (_c = this.value.strengthTwo) !== null && _c !== void 0 ? _c : 1
            : (_d = this.value.strength) !== null && _d !== void 0 ? _d : 1;
        let textColor = undefined;
        if (((_e = this.loraInfo) === null || _e === void 0 ? void 0 : _e.strengthMax) != null && strengthValue > ((_f = this.loraInfo) === null || _f === void 0 ? void 0 : _f.strengthMax)) {
            textColor = "#c66";
        }
        else if (((_g = this.loraInfo) === null || _g === void 0 ? void 0 : _g.strengthMin) != null && strengthValue < ((_h = this.loraInfo) === null || _h === void 0 ? void 0 : _h.strengthMin)) {
            textColor = "#c66";
        }
        const [leftArrow, text, rightArrow] = drawNumberWidgetPart(ctx, {
            posX: node.size[0] - margin - innerMargin - innerMargin,
            posY,
            height,
            value: strengthValue,
            direction: -1,
            textColor,
        });
        this.hitAreas.strengthDec.bounds = leftArrow;
        this.hitAreas.strengthVal.bounds = text;
        this.hitAreas.strengthInc.bounds = rightArrow;
        this.hitAreas.strengthAny.bounds = [leftArrow[0], rightArrow[0] + rightArrow[1] - leftArrow[0]];
        rposX = leftArrow[0] - innerMargin;
        if (this.showModelAndClip) {
            rposX -= innerMargin;
            this.hitAreas.strengthTwoDec.bounds = this.hitAreas.strengthDec.bounds;
            this.hitAreas.strengthTwoVal.bounds = this.hitAreas.strengthVal.bounds;
            this.hitAreas.strengthTwoInc.bounds = this.hitAreas.strengthInc.bounds;
            this.hitAreas.strengthTwoAny.bounds = this.hitAreas.strengthAny.bounds;
            let textColor = undefined;
            if (((_j = this.loraInfo) === null || _j === void 0 ? void 0 : _j.strengthMax) != null && this.value.strength > ((_k = this.loraInfo) === null || _k === void 0 ? void 0 : _k.strengthMax)) {
                textColor = "#c66";
            }
            else if (((_l = this.loraInfo) === null || _l === void 0 ? void 0 : _l.strengthMin) != null &&
                this.value.strength < ((_m = this.loraInfo) === null || _m === void 0 ? void 0 : _m.strengthMin)) {
                textColor = "#c66";
            }
            const [leftArrow, text, rightArrow] = drawNumberWidgetPart(ctx, {
                posX: rposX,
                posY,
                height,
                value: (_o = this.value.strength) !== null && _o !== void 0 ? _o : 1,
                direction: -1,
                textColor,
            });
            this.hitAreas.strengthDec.bounds = leftArrow;
            this.hitAreas.strengthVal.bounds = text;
            this.hitAreas.strengthInc.bounds = rightArrow;
            this.hitAreas.strengthAny.bounds = [
                leftArrow[0],
                rightArrow[0] + rightArrow[1] - leftArrow[0],
            ];
            rposX = leftArrow[0] - innerMargin;
        }
        const infoIconSize = height * 0.66;
        const infoWidth = infoIconSize + innerMargin + innerMargin;
        if (this.hitAreas["info"]) {
            rposX -= innerMargin;
            drawInfoIcon(ctx, rposX - infoIconSize, posY + (height - infoIconSize) / 2, infoIconSize);
            this.hitAreas.info.bounds = [rposX - infoIconSize, infoWidth];
            rposX = rposX - infoIconSize - innerMargin;
        }
        const loraWidth = rposX - posX;
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        const loraLabel = String(((_p = this.value) === null || _p === void 0 ? void 0 : _p.lora) || "None");
        ctx.fillText(fitString(ctx, loraLabel, loraWidth), posX, midY);
        this.hitAreas.lora.bounds = [posX, loraWidth];
        posX += loraWidth + innerMargin;
        ctx.globalAlpha = app.canvas.editor_alpha;
        ctx.restore();
    }
    serializeValue(serializedNode, widgetIndex) {
        var _b;
        const v = { ...this.value };
        if (!this.showModelAndClip) {
            delete v.strengthTwo;
        }
        else {
            this.value.strengthTwo = (_b = this.value.strengthTwo) !== null && _b !== void 0 ? _b : 1;
            v.strengthTwo = this.value.strengthTwo;
        }
        return v;
    }
    onToggleDown(event, pos, node) {
        this.value.on = !this.value.on;
        this.cancelMouseDown();
        return true;
    }
    onInfoDown(event, pos, node) {
        this.showLoraInfoDialog();
    }
    onLoraDown(event, pos, node) {
        showLoraChooser(event, (value) => {
            if (typeof value === "string") {
                this.value.lora = value;
                this.loraInfo = null;
            }
            node.setDirtyCanvas(true, true);
        });
        this.cancelMouseDown();
    }
    onStrengthDecDown(event, pos, node) {
        this.stepStrength(-1, false);
    }
    onStrengthIncDown(event, pos, node) {
        this.stepStrength(1, false);
    }
    onStrengthTwoDecDown(event, pos, node) {
        this.stepStrength(-1, true);
    }
    onStrengthTwoIncDown(event, pos, node) {
        this.stepStrength(1, true);
    }
    onStrengthAnyMove(event, pos, node) {
        this.doOnStrengthAnyMove(event, false);
    }
    onStrengthTwoAnyMove(event, pos, node) {
        this.doOnStrengthAnyMove(event, true);
    }
    doOnStrengthAnyMove(event, isTwo = false) {
        var _b;
        if (event.deltaX) {
            let prop = isTwo ? "strengthTwo" : "strength";
            this.haveMouseMovedStrength = true;
            this.value[prop] = ((_b = this.value[prop]) !== null && _b !== void 0 ? _b : 1) + event.deltaX * 0.05;
        }
    }
    onStrengthValUp(event, pos, node) {
        this.doOnStrengthValUp(event, false);
    }
    onStrengthTwoValUp(event, pos, node) {
        this.doOnStrengthValUp(event, true);
    }
    doOnStrengthValUp(event, isTwo = false) {
        if (this.haveMouseMovedStrength)
            return;
        let prop = isTwo ? "strengthTwo" : "strength";
        const canvas = app.canvas;
        canvas.prompt("Value", this.value[prop], (v) => (this.value[prop] = Number(v)), event);
    }
    onMouseUp(event, pos, node) {
        super.onMouseUp(event, pos, node);
        this.haveMouseMovedStrength = false;
    }
    stepStrength(direction, isTwo = false) {
        var _b;
        let step = 0.05;
        let prop = isTwo ? "strengthTwo" : "strength";
        let strength = ((_b = this.value[prop]) !== null && _b !== void 0 ? _b : 1) + step * direction;
        this.value[prop] = Math.round(strength * 100) / 100;
    }
}

app.registerExtension({
    name: "AE.LorasLoader",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "AE.LorasLoader") {
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (info) {
                const r = onConfigure ? onConfigure.apply(this, info) : undefined;

                var _b;
                while ((_b = this.widgets) === null || _b === void 0 ? void 0 : _b.length)
                    removeWidget(this, 0);

                this.addLoraWidget = null;
                this._tempWidth = this.size[0];
                this._tempHeight = this.size[1];

                for (const widgetValue of info.widgets_values || []) {
                    if ((widgetValue === null || widgetValue === void 0 ? void 0 : widgetValue.lora) !== undefined) {
                        const widget = addNewLoraWidget(this);
                        widget.value = { ...widgetValue };
                    }
                }

                addNonLoraWidgets(this);
                this.size[0] = this._tempWidth;
                this.size[1] = Math.max(this._tempHeight, this.computeSize()[1]);
            };

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this) : undefined;

                this.loraWidgetsCounter = 0;
                this.properties[PROP_LABEL_SHOW_STRENGTHS] = PROP_VALUE_SHOW_STRENGTHS_SINGLE;

                aeApi.getLoras();
                
                addNonLoraWidgets(this);
            };

            nodeType.prototype.getSlotInPosition = function (canvasX, canvasY) {
                var _b;
                
                let lastWidget = null;
                for (const widget of this.widgets) {
                    if (!widget.last_y)
                        return;
                    if (canvasY > this.pos[1] + widget.last_y) {
                        lastWidget = widget;
                        continue;
                    }
                    break;
                }

                if ((_b = lastWidget === null || lastWidget === void 0 ? void 0 : lastWidget.name) === null || _b === void 0 ? void 0 : _b.startsWith("lora_")) {
                    return { widget: lastWidget, output: { type: "LORA WIDGET" } };
                }
                
                return null;
            };

            const getSlotMenuOptions = nodeType.prototype.getSlotMenuOptions;
            nodeType.prototype.getSlotMenuOptions = function (slot) {
                var _b, _c, _d, _e, _f, _g;
                if ((_c = (_b = slot === null || slot === void 0 ? void 0 : slot.widget) === null || _b === void 0 ? void 0 : _b.name) === null || _c === void 0 ? void 0 : _c.startsWith("lora_")) {
                    const widget = slot.widget;
                    const index = this.widgets.indexOf(widget);
                    const canMoveUp = !!((_e = (_d = this.widgets[index - 1]) === null || _d === void 0 ? void 0 : _d.name) === null || _e === void 0 ? void 0 : _e.startsWith("lora_"));
                    const canMoveDown = !!((_g = (_f = this.widgets[index + 1]) === null || _f === void 0 ? void 0 : _f.name) === null || _g === void 0 ? void 0 : _g.startsWith("lora_"));
                    const menuItems = [
                        {
                            content: `Move Up`,
                            disabled: !canMoveUp,
                            callback: () => {
                                moveArrayItem(this.widgets, widget, index - 1);
                            },
                        },
                        {
                            content: `Move Down`,
                            disabled: !canMoveDown,
                            callback: () => {
                                moveArrayItem(this.widgets, widget, index + 1);
                            },
                        },
                        {
                            content: `Remove`,
                            callback: () => {
                                removeArrayItem(this.widgets, widget);
                            },
                        },
                    ];

                    let canvas = app.canvas;
                    new LiteGraph.ContextMenu(menuItems, { title: "LORA WIDGET", event: app.lastAdjustedMouseEvent }, canvas.getCanvasWindow());

                    return null;
                }
                return getSlotMenuOptions ? getSlotMenuOptions.apply(this, slot) : undefined;
            };
        }
    },
});