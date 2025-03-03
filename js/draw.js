export function isLowQuality() {
    var _a;
    const canvas = app.canvas;
    return (((_a = canvas.ds) === null || _a === void 0 ? void 0 : _a.scale) || 1) <= 0.5;
}

function binarySearch(max, getValue, match) {
    let min = 0;
    while (min <= max) {
        let guess = Math.floor((min + max) / 2);
        const compareVal = getValue(guess);
        if (compareVal === match)
            return guess;
        if (compareVal < match)
            min = guess + 1;
        else
            max = guess - 1;
    }
    return max;
}

export function fitString(ctx, str, maxWidth) {
    let width = ctx.measureText(str).width;
    const ellipsis = "...";
    const ellipsisWidth = measureText(ctx, ellipsis);
    if (width <= maxWidth || width <= ellipsisWidth) {
        return str;
    }
    const index = binarySearch(str.length, (guess) => measureText(ctx, str.substring(0, guess)), maxWidth - ellipsisWidth);
    return str.substring(0, index) + ellipsis;
}

export function measureText(ctx, str) {
    return ctx.measureText(str).width;
}

export function drawRoundedRectangle(ctx, options) {
    const lowQuality = isLowQuality();
    options = { ...options };
    ctx.strokeStyle = options.colorStroke || LiteGraph.WIDGET_OUTLINE_COLOR;
    ctx.fillStyle = options.colorBackground || LiteGraph.WIDGET_BGCOLOR;
    ctx.beginPath();
    ctx.roundRect(options.posX, options.posY, options.width, options.height, lowQuality ? [0] : options.borderRadius ? [options.borderRadius] : [options.height * 0.5]);
    ctx.fill();
    !lowQuality && ctx.stroke();
}

export function drawTogglePart(ctx, options) {
    const lowQuality = isLowQuality();
    ctx.save();
    const { posX, posY, height, value } = options;
    const toggleRadius = height * 0.36;
    const toggleBgWidth = height * 1.5;
    if (!lowQuality) {
        ctx.beginPath();
        ctx.roundRect(posX + 4, posY + 4, toggleBgWidth - 8, height - 8, [height * 0.5]);
        ctx.globalAlpha = app.canvas.editor_alpha * 0.25;
        ctx.fillStyle = "rgba(255,255,255,0.45)";
        ctx.fill();
        ctx.globalAlpha = app.canvas.editor_alpha;
    }
    ctx.fillStyle = value === true ? "#89B" : "#888";
    const toggleX = lowQuality || value === false
        ? posX + height * 0.5
        : value === true
            ? posX + height
            : posX + height * 0.75;
    ctx.beginPath();
    ctx.arc(toggleX, posY + height * 0.5, toggleRadius, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    return [posX, toggleBgWidth];
}

export function drawNumberWidgetPart(ctx, options) {
    const arrowWidth = 9;
    const arrowHeight = 10;
    const innerMargin = 3;
    const numberWidth = 32;
    const xBoundsArrowLess = [0, 0];
    const xBoundsNumber = [0, 0];
    const xBoundsArrowMore = [0, 0];
    ctx.save();
    let posX = options.posX;
    const { posY, height, value, textColor } = options;
    const midY = posY + height / 2;
    if (options.direction === -1) {
        posX = posX - arrowWidth - innerMargin - numberWidth - innerMargin - arrowWidth;
    }
    ctx.fill(new Path2D(`M ${posX} ${midY} l ${arrowWidth} ${arrowHeight / 2} l 0 -${arrowHeight} L ${posX} ${midY} z`));
    xBoundsArrowLess[0] = posX;
    xBoundsArrowLess[1] = arrowWidth;
    posX += arrowWidth + innerMargin;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const oldTextcolor = ctx.fillStyle;
    if (textColor) {
        ctx.fillStyle = textColor;
    }
    ctx.fillText(fitString(ctx, value.toFixed(2), numberWidth), posX + numberWidth / 2, midY);
    ctx.fillStyle = oldTextcolor;
    xBoundsNumber[0] = posX;
    xBoundsNumber[1] = numberWidth;
    posX += numberWidth + innerMargin;
    ctx.fill(new Path2D(`M ${posX} ${midY - arrowHeight / 2} l ${arrowWidth} ${arrowHeight / 2} l -${arrowWidth} ${arrowHeight / 2} v -${arrowHeight} z`));
    xBoundsArrowMore[0] = posX;
    xBoundsArrowMore[1] = arrowWidth;
    ctx.restore();
    return [xBoundsArrowLess, xBoundsNumber, xBoundsArrowMore];
}
drawNumberWidgetPart.WIDTH_TOTAL = 9 + 3 + 32 + 3 + 9;

export function drawNodeWidget(ctx, options) {
    const lowQuality = isLowQuality();
    const data = {
        width: options.width,
        height: options.height,
        posY: options.posY,
        lowQuality,
        margin: 15,
        colorOutline: LiteGraph.WIDGET_OUTLINE_COLOR,
        colorBackground: LiteGraph.WIDGET_BGCOLOR,
        colorText: LiteGraph.WIDGET_TEXT_COLOR,
        colorTextSecondary: LiteGraph.WIDGET_SECONDARY_TEXT_COLOR,
    };
    ctx.strokeStyle = options.colorStroke || data.colorOutline;
    ctx.fillStyle = options.colorBackground || data.colorBackground;
    ctx.beginPath();
    ctx.roundRect(data.margin, data.posY, data.width - data.margin * 2, data.height, lowQuality ? [0] : options.borderRadius ? [options.borderRadius] : [options.height * 0.5]);
    ctx.fill();
    if (!lowQuality) {
        ctx.stroke();
    }
    return data;
}

export function addMenuItemOnExtraMenuOptions(node, config, menuOptions, after = "Shape") {
    let idx = menuOptions
        .slice()
        .reverse()
        .findIndex((option) => option === null || option === void 0 ? void 0 : option.isAe);
    if (idx == -1) {
        idx = menuOptions.findIndex((option) => { var _a; return (_a = option === null || option === void 0 ? void 0 : option.content) === null || _a === void 0 ? void 0 : _a.includes(after); }) + 1;
        if (!idx) {
            idx = menuOptions.length - 1;
        }
        menuOptions.splice(idx, 0, null);
        idx++;
    }
    else {
        idx = menuOptions.length - idx;
    }
    const subMenuOptions = typeof config.subMenuOptions === "function"
        ? config.subMenuOptions(node)
        : config.subMenuOptions;
    menuOptions.splice(idx, 0, {
        content: typeof config.name == "function" ? config.name(node) : config.name,
        has_submenu: !!(subMenuOptions === null || subMenuOptions === void 0 ? void 0 : subMenuOptions.length),
        isAe: true,
        callback: (value, _options, event, parentMenu, _node) => {
            if (!!(subMenuOptions === null || subMenuOptions === void 0 ? void 0 : subMenuOptions.length)) {
                new LiteGraph.ContextMenu(subMenuOptions.map((option) => (option ? { content: option } : null)), {
                    event,
                    parentMenu,
                    callback: (subValue, _options, _event, _parentMenu, _node) => {
                        if (config.property) {
                            node.properties = node.properties || {};
                            node.properties[config.property] = config.prepareValue
                                ? config.prepareValue(subValue.content || '', node)
                                : subValue.content || '';
                        }
                        config.callback && config.callback(node, subValue === null || subValue === void 0 ? void 0 : subValue.content);
                    },
                });
                return;
            }
            if (config.property) {
                node.properties = node.properties || {};
                node.properties[config.property] = config.prepareValue
                    ? config.prepareValue(node.properties[config.property], node)
                    : !node.properties[config.property];
            }
            config.callback && config.callback(node, value === null || value === void 0 ? void 0 : value.content);
        },
    });
}

export function addMenuItem(node, _app, config, after = "Shape") {
    const oldGetExtraMenuOptions = node.prototype.getExtraMenuOptions;
    node.prototype.getExtraMenuOptions = function (canvas, menuOptions) {
        oldGetExtraMenuOptions && oldGetExtraMenuOptions.apply(this, [canvas, menuOptions]);
        addMenuItemOnExtraMenuOptions(this, config, menuOptions, after);
    };
}