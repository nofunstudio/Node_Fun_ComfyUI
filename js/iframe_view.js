// iframe_widget.js
import { app } from "../../scripts/app.js";
import { $el } from "../../scripts/ui.js";

// Global store for texture data, keyed by unique_id
window.TEXTURE_STORE = window.TEXTURE_STORE || {};

// Function to create the iframe widget
async function widgetIframe(node, nodeData, inputData, app) {
	// Locate our parameter widgets
	const urlWidget = node.widgets.find((w) => w.name === "url");
	const widthWidget = node.widgets.find((w) => w.name === "width");
	const heightWidget = node.widgets.find((w) => w.name === "height");
	const stateWidget = node.widgets.find((w) => w.name === "scene_state");

	console.log("[widgetIframe] Initializing iframe widget for node:", node);
	node.isCaptureComplete = false;
	// Create iframe wrapper with some styling
	const iframeWrapper = $el("div.iframeWrapper", {
		style: {
			width: "100%",
			height: "100%",
			overflow: "hidden",
			border: "1px solid var(--border-color)",
			borderRadius: "4px",
		},
	});

	// Create the iframe element
	const iframe = $el("iframe", {
		style: {
			width: widthWidget.value + "px",
			height: heightWidget.value + "px",
			border: "none",
		},
		sandbox: "allow-scripts allow-same-origin allow-forms",
		loading: "lazy",
	});

	iframeWrapper.appendChild(iframe);

	// Set initial URL if provided
	if (urlWidget.value) {
		iframe.src = urlWidget.value;
	}

	// Ensure we have a "frame_count" widget to specify animation frame count
	let frameCountWidget = node.widgets.find((w) => w.name === "frame_count");
	if (!frameCountWidget) {
		frameCountWidget = {
			name: "frame_count",
			value: "4", // default number of frames as a string
			isHidden: false, // visible so user can change it
			callback: function () {
				// Reset frame buffer when the frame_count changes.
				node.frameBuffer = [];
				node.isCaptureComplete = false;
				console.log("isCaptureComplete callback:", node.isCaptureComplete);
			},
		};
		node.widgets.push(frameCountWidget);
	}

	// Message handler for communication from the iframe
	function messageHandler(event) {
		// Ensure the message comes from our iframe
		if (event.source !== iframe.contentWindow) return;

		console.log("[messageHandler] Received message:", event.data);
		const { type, data } = event.data;
		switch (type) {
			case "IFRAME_READY":
				console.log("[messageHandler] IFRAME_READY received");
				sendParamsToIframe();
				if (iframe.contentWindow)
					// iframe.contentWindow.postMessage({ type: "CAPTURE_TEXTURES" }, "*");
					break;
			case "TEXTURE_OUTPUT":
				console.log(
					"[messageHandler] TEXTURE_OUTPUT received with data:",
					data
				);
				handleTextureOutput(data);
				break;
			case "SCENE_STATE_UPDATED":
				if (data.state) {
					stateWidget.value = JSON.stringify(data.state, null, 2);
					console.log(
						"[messageHandler] Updated scene state:",
						stateWidget.value
					);
				}
				break;
			default:
				console.warn("[messageHandler] Unhandled message type:", type);
		}
	}

	// Attach the message handler
	console.log("[widgetIframe] Attaching message event listener for iframe.");
	window.addEventListener("message", messageHandler, false);

	// Function to send parameters (width, height, scene state) to the iframe
	function sendParamsToIframe() {
		let sceneState;
		try {
			sceneState = JSON.parse(stateWidget.value || "{}");
		} catch (e) {
			console.error("Error parsing scene state, using default:", e);
			sceneState = {};
		}
		const params = {
			type: "PARAMS_UPDATE",
			data: {
				width: parseInt(widthWidget.value, 10) || 512,
				height: parseInt(heightWidget.value, 10) || 512,
				sceneState,
				unique_id: node.unique_id,
			},
		};

		console.log(
			"[sendParamsToIframe] Sending params at",
			Date.now(),
			":",
			params
		);
		if (iframe.contentWindow) {
			// If you know the target origin, replace '*' with that string
			iframe.contentWindow.postMessage(params, "*");
		}
	}

	// Updated handleTextureOutput: Capture frames until target is reached, then aggregate them.
	function handleTextureOutput(data) {
		console.log(
			"isCaptureComplete handleTextureOutput:",
			node.isCaptureComplete
		);
		if (node.isCaptureComplete) return;
		// Validate that we received all expected textures.
		if (!data || !data.color || !data.depth || !data.normal || !data.canny) {
			console.warn(
				"[handleTextureOutput] Incomplete texture data received; skipping frame."
			);
			return;
		}

		// Ensure the frame buffer exists.
		if (!node.frameBuffer) node.frameBuffer = [];

		// Append the complete frame data (with base64 textures) into the buffer.
		node.frameBuffer.push({
			timestamp: Date.now(),
			textures: {
				color: data.color,
				depth: data.depth,
				normal: data.normal,
				canny: data.canny,
			},
		});
		console.log(
			"[handleTextureOutput] Captured frame",
			node.frameBuffer.length
		);

		// Use the cached target frame count value.
		let targetFrameCount = node.targetFrameCount || 4;
		console.log("[handleTextureOutput] Target frame count:", targetFrameCount);

		// If we haven't reached the target, request the next capture.
		if (node.frameBuffer.length < targetFrameCount) {
			if (iframe && iframe.contentWindow) {
				console.log("[handleTextureOutput] Requesting next frame capture...");
				if (node.frameBuffer.length === 0) {
					// iframe.contentWindow.postMessage({ type: "CAPTURE_TEXTURES" }, "*");
					// console.log("[handleTextureOutput] starting capture...");
				}
			} else {
				console.warn(
					"[handleTextureOutput] No iframe contentWindow available."
				);
			}
		} else {
			// Update the hidden widget with the animation frames.
			node.setHiddenWidgetValue("animationFrames", node.frameBuffer);

			// Force a flush of hidden widget values by invoking onSerialize immediately.
			if (typeof node.onSerialize === "function") {
				const serializedHiddenData = {};
				node.onSerialize(serializedHiddenData);
				console.log(
					"[handleTextureOutput] Forced reserialization:",
					serializedHiddenData
				);
			}

			iframe.contentWindow.postMessage({ type: "STOP_CAPTURE" }, "*");
			node.isCaptureComplete = true;
			console.log("isCaptureComplete StopCapture:", node.isCaptureComplete);
			console.log(
				"[handleTextureOutput] Completed capturing",
				targetFrameCount,
				"frames. Capture complete flag set."
			);
		}
	}

	// Create the DOM widget for this node and register cleanup
	const widget = node.addDOMWidget("iframe", "custom_widget", iframeWrapper, {
		getValue() {
			return {
				url: urlWidget.value,
				width: widthWidget.value,
				height: heightWidget.value,
				scene_state: stateWidget.value,
			};
		},
		setValue(v) {
			if (v.url) {
				iframe.src = v.url;
				urlWidget.value = v.url;
			}
		},
	});

	// Attach a reference to the iframe on the widget so we can access it later
	widget.iframe = iframe;
	console.log("[widgetIframe] Widget created and registered.");

	// Handle resizing: update both node size and iframe dimensions, then re-send params
	node.onResize = function () {
		const width = parseInt(widthWidget.value, 10) || 512;
		const height = parseInt(heightWidget.value, 10) || 512;
		node.size = [width, Math.max(height + 40, 140)];
		iframe.style.width = width + "px";
		iframe.style.height = height - 40 + "px";
		console.log("[onResize] Node resized to new dimensions:", width, height);
		sendParamsToIframe();
	};

	// When the URL widget changes, update the iframe's src
	urlWidget.callback = function () {
		console.log("[URL Widget] Updating iframe src to:", this.value);
		iframe.src = this.value;
	};

	// When any of the parameter widgets change, update the iframe with the new parameters
	widthWidget.callback =
		heightWidget.callback =
		stateWidget.callback =
			function () {
				sendParamsToIframe();
			};

	// Modified setHiddenWidgetValue to only handle "unique_id" and "animationFrames"
	node.setHiddenWidgetValue = function (name, value) {
		if (name !== "unique_id" && name !== "animationFrames") return;
		console.log(
			"[setHiddenWidgetValue] Setting hidden widget for",
			name,
			"at",
			Date.now()
		);
		let widgetEntry = this.widgets.find((w) => w.name === name);
		if (widgetEntry) {
			widgetEntry.value = value;
			console.log(`[setHiddenWidgetValue] Updated widget '${name}' to:`, value);
		} else {
			widgetEntry = { name, value, isHidden: true };
			this.widgets.push(widgetEntry);
			console.log(
				`[setHiddenWidgetValue] Created and set widget '${name}' to:`,
				value
			);
		}
		if (this.unique_id) {
			window.TEXTURE_STORE = window.TEXTURE_STORE || {};
			window.TEXTURE_STORE[this.unique_id] =
				window.TEXTURE_STORE[this.unique_id] || {};
			window.TEXTURE_STORE[this.unique_id][name] = value;
			console.log(
				`[setHiddenWidgetValue] Global texture store for unique_id '${this.unique_id}':`,
				window.TEXTURE_STORE[this.unique_id]
			);
		}
	};

	// When creating the widget
	node.cleanup = () => {
		console.log(
			"[cleanup] Cleanup triggered for node:",
			node.unique_id,
			"at",
			Date.now()
		);
		window.removeEventListener("message", messageHandler, false);
		if (node.unique_id && window.TEXTURE_STORE[node.unique_id]) {
			delete window.TEXTURE_STORE[node.unique_id];
		}
		console.log("[cleanup] Cleaned up resources for node:", node.unique_id);
	};

	// Updated onSerialize: Only serialize the animationFrames hidden widget.
	node.onSerialize = (nodeData) => {
		nodeData.hidden = nodeData.hidden || {};
		const animWidget = node.widgets.find((w) => w.name === "animationFrames");
		nodeData.hidden["animationFrames"] =
			animWidget && animWidget.value ? animWidget.value : [];
		console.log("[onSerialize] Serialized hidden data:", nodeData.hidden);
	};

	// Updated triggerCapture function with guard logic to avoid recapturing frames
	node.triggerCapture = function () {
		// Check if frames are already captured. If so, do not trigger a new capture.
		if (
			node.isCaptureComplete &&
			node.frameBuffer &&
			node.frameBuffer.length > 0
		) {
			console.log(
				"[triggerCapture] Frames already captured; skipping recapture."
			);
			return;
		}

		// Reset state for a new capture.
		node.frameBuffer = [];
		node.isCaptureComplete = false;
		node.setHiddenWidgetValue("animationFrames", []);
		node.targetFrameCount =
			parseInt(node.widgets.find((w) => w.name === "frame_count")?.value, 10) ||
			4;
		console.log(
			"[triggerCapture] Reset capture state. Target frame count:",
			node.targetFrameCount
		);

		if (iframe.contentWindow)
			iframe.contentWindow.postMessage({ type: "CAPTURE_TEXTURES" }, "*");
		console.log(" starting capture...");

		// Start polling for capture completion.
		const checkCaptureInterval = setInterval(() => {
			if (node.isCaptureComplete) {
				clearInterval(checkCaptureInterval);
				// Force a re‑serialization of hidden widget values.
				if (typeof node.onSerialize === "function") {
					const serializedHiddenData = {};
					node.onSerialize(serializedHiddenData);
					console.log(
						"[triggerCapture] Forced reserialization:",
						serializedHiddenData
					);
				}
				console.log(
					"[triggerCapture] Capture complete, queuing prompt using updated hidden data."
				);
				app.queuePrompt();

				// After queuing the prompt, schedule clearing the captured frames
				// so that the next capture cycle can run fresh.
				setTimeout(() => {
					console.log(
						"[triggerCapture] Clearing capture frames for new cycle."
					);
					node.frameBuffer = [];
					node.isCaptureComplete = false;
					node.setHiddenWidgetValue("animationFrames", []);
				}, 50);
			}
		}, 500);
	};

	return widget;
}

// Register the extension and add extra menu options for the IframeView node
app.registerExtension({
	name: "Iframe View",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeType.comfyClass === "IframeView") {
			// Add extra menu options to refresh the iframe and trigger texture capture
			nodeType.prototype.getExtraMenuOptions = function (_, options) {
				options.unshift(
					{
						content: "🔄 Refresh Iframe",
						callback: () => {
							const domWidget = this.widgets.find((w) => w.name === "iframe");
							if (domWidget && domWidget.iframe) {
								const url = this.widgets.find((w) => w.name === "url").value;
								console.log("[Extra Menu] Refreshing iframe with url:", url);
								domWidget.iframe.src = url;
							}
						},
					},
					{
						content: "📸 Capture Textures",
						callback: () => {
							const domWidget = this.widgets.find((w) => w.name === "iframe");
							if (domWidget?.iframe?.contentWindow) {
								console.log("[Extra Menu] Triggering texture capture.");
								domWidget.iframe.contentWindow.postMessage(
									{ type: "CAPTURE_TEXTURES" },
									"*"
								);
							}
						},
					}
				);
			};

			const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = async function () {
				// Create and initialize the Iframe widget.
				const widget = await widgetIframe(this, nodeData, nodeData, app);
				this.onResize();

				// Use node.id as the unique id to ensure consistency between JS and Python.
				this.unique_id = this.id;
				console.log(
					"[widgetIframe] Using node id as unique id:",
					this.unique_id
				);

				// Update or create the hidden unique id widget to be consistent in the global texture store.
				let uniqueWidget = this.widgets.find((w) => w.name === "unique_id");
				if (uniqueWidget) uniqueWidget.value = this.unique_id;
				else {
					this.widgets.push({
						name: "unique_id",
						value: this.unique_id,
						isHidden: true,
					});
				}

				return originalOnNodeCreated
					? originalOnNodeCreated.apply(this, arguments)
					: undefined;
			};

			// Updated onProcess method that waits for capture to finish if needed.
			nodeType.prototype.onProcess = async function () {
				console.log("[onProcess] Node processing triggered.");
				// Only trigger a capture if frames are not already captured.
				if (
					!this.isCaptureComplete &&
					typeof this.triggerCapture === "function"
				) {
					console.log("[onProcess] Capture not complete; triggering capture.");
					this.triggerCapture();

					// Optionally, wait until capture completes (up to a maximum wait time).
					const maxWaitTime = 10000; // e.g. 10 seconds.
					const pollInterval = 100;
					let waited = 0;
					while (!this.isCaptureComplete && waited < maxWaitTime) {
						await new Promise((resolve) => setTimeout(resolve, pollInterval));
						waited += pollInterval;
					}
					if (!this.isCaptureComplete) {
						console.warn(
							"[onProcess] Capture did not complete within wait time; proceeding with current data."
						);
					} else {
						console.log(
							"[onProcess] Capture complete; proceeding with updated hidden data."
						);
					}
				} else {
					console.log(
						"[onProcess] Capture already complete; using existing data."
					);
				}
			};
		}
	},
});

function generateFallbackUniqueId() {
	// Only used if no unique id is provided
	return Math.random().toString(36).substr(2, 9);
}

export function onNodeCreated(node) {
	// Try to read the unique id from node.settings or node.hiddenInputs
	let providedUniqueId = undefined;
	if (node.settings && node.settings.unique_id) {
		providedUniqueId = node.settings.unique_id;
	} else if (node.hiddenInputs && node.hiddenInputs.unique_id) {
		providedUniqueId = node.hiddenInputs.unique_id;
	}

	// Force it to a string so that Python (which does str(unique_id)) can match correctly.
	if (providedUniqueId) {
		node.unique_id = String(providedUniqueId);
	} else {
		node.unique_id = generateFallbackUniqueId();
	}

	console.log("[widgetIframe] onNodeCreated: Using unique id:", node.unique_id);

	// Update or create a hidden widget for unique id to ensure consistency.
	let uniqueIdWidget = node.widgets.find((w) => w.name === "unique_id");
	if (uniqueIdWidget) {
		uniqueIdWidget.value = node.unique_id;
		console.log(
			"[widgetIframe] Updated hidden widget 'unique_id' to:",
			node.unique_id
		);
	} else {
		node.widgets.push({
			name: "unique_id",
			value: node.unique_id,
			isHidden: true,
		});
		console.log(
			"[widgetIframe] Created hidden widget 'unique_id' with:",
			node.unique_id
		);
	}
}

export { widgetIframe };

// At the end of your file, or in a separate initialization block,
// override the global queuePrompt to trigger capture on IframeView nodes.

const originalQueuePrompt = app.queuePrompt;
app.queuePrompt = function (...args) {
	console.log("[QueuePrompt] Triggering capture for IframeView nodes.");
	// Call the original queuePrompt functionality.
	const result = originalQueuePrompt.apply(this, args);
	// Find all nodes in the graph and trigger capture on those with comfyClass "IframeView".
	if (app.graph && app.graph._nodes_by_id) {
		Object.values(app.graph._nodes_by_id).forEach((node) => {
			if (
				node.comfyClass === "IframeView" &&
				typeof node.triggerCapture === "function"
			) {
				console.log(`[QueuePrompt] Triggering capture on node ${node.id}`);
				node.triggerCapture();
			}
		});
	}
	return result;
};
