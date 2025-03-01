import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "AE.Lists",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "AE.IntList" || nodeData.name === "AE.FloatList" || nodeData.name === "AE.StringList") {
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (info) {
                const r = onConfigure ? onConfigure.apply(this, info) : undefined;

                if (this.updateNumbers)
                    this.updateNumbers();
            };

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this) : undefined;
                const input = this.widgets.find((w) => w.name === "list").element;

                input.style.border = "0.5px solid " + LiteGraph.WIDGET_OUTLINE_COLOR;
                input.style.borderRadius = "4px";
                input.style.paddingLeft = "24px";

                const numbers = document.createElement('textarea');

                numbers.classList = input.classList;

                numbers.style.border = "0.5px solid rgba(255, 0, 0, 0)";
                numbers.style.borderRadius = "4px";
                numbers.style.paddingLeft = "4px";
                numbers.style.width = "20px";
                numbers.style.textAlign = "right";
                numbers.style.overflow = 'hidden';

                numbers.style.background = "rgba(255, 0, 0, 0)";
                numbers.style.pointerEvents = "none";
                numbers.style.opacity = "0.4";

                function updateLineNumbers() {
                    const maxLines = 99;
                    const lines = input.value.split('\n');

                    if (lines.length > maxLines) {
                        input.value = lines.slice(0, maxLines).join('\n');
                    }

                    if (nodeData.name === "AE.IntList") {
                        const filteredValue = input.value.split('\n').map(line => {
                            return line.replace(/(?!^-)-|[^-\d]/g, '');
                        }).join('\n');

                        if (input.value !== filteredValue) {
                            input.value = filteredValue;
                        }
                    } else if (nodeData.name === "AE.FloatList") {
                        const filteredValue = input.value.split('\n').map(line => {
                            return line.replace(/(?!^-)-|[^\d.-]|(?<=\..*)\./g, '');
                        }).join('\n');

                        if (input.value !== filteredValue) {
                            input.value = filteredValue;
                        }
                    }

                    numbers.value = Array(input.value.split('\n').length)
                        .fill(0).map((_, i) => `${i + 1}`).join('\r\n');
                }

                input.addEventListener('input', updateLineNumbers);

                input.addEventListener('scroll', () => {
                    numbers.scrollTop = input.scrollTop;
                });

                input.parentNode.insertBefore(numbers, input);

                this.numbers = numbers;
                this.updateNumbers = updateLineNumbers;

                const observer = new MutationObserver((mutationsList) => {
                    for (let mutation of mutationsList) {
                        if (mutation.type === 'attributes') {
                            if (mutation.attributeName === 'style') {
                                numbers.style.transformOrigin = input.style.transformOrigin;
                                numbers.style.transform = input.style.transform;
                                numbers.style.left = input.style.left;
                                numbers.style.top = input.style.top;
                                //numbers.style.width = input.style.width;
                                numbers.style.height = input.style.height;
                                numbers.style.position = input.style.position;
                                numbers.style.zIndex = parseInt(input.style.zIndex) + 1;
                                numbers.style.willChange = input.style.willChange;
                                numbers.style.display = input.style.display;
                            }
                        }
                    }
                });

                observer.observe(input, {
                    attributes: true
                });
            };

            const onRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function () {
                const r = onRemoved ? onRemoved.apply(this) : undefined;

                this.numbers.remove();
            };
        }
    },
});