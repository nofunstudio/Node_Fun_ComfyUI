// iframe_widget.js
import { app } from "../../scripts/app.js";
import { $el } from "../../scripts/ui.js";

// Global store for texture data, keyed by unique_id
window.TEXTURE_STORE = window.TEXTURE_STORE || {};

// Function to create the iframe widget
async function widgetIframe(node, nodeData, inputData, app) {
  // Create iframe wrapper with some styling
  const iframeWrapper = $el("div.iframeWrapper", {
    style: {
      width: "100%",
      height: "100%",
      overflow: "hidden",
      border: "1px solid var(--border-color)",
      borderRadius: "4px"
    }
  });

  // Create the iframe element
  const iframe = $el("iframe", {
    style: {
      width: "100%",
      height: "100%",
      border: "none"
    },
    sandbox: "allow-scripts allow-same-origin allow-forms",
    loading: "lazy"
  });

  iframeWrapper.appendChild(iframe);

  // Locate our parameter widgets
  const urlWidget = node.widgets.find((w) => w.name === "url");
  const widthWidget = node.widgets.find((w) => w.name === "width");
  const heightWidget = node.widgets.find((w) => w.name === "height");
  const stateWidget = node.widgets.find((w) => w.name === "scene_state");

  // Set initial URL if provided
  if (urlWidget.value) {
    iframe.src = urlWidget.value;
  }

  // Message handler for communication from the iframe
  function messageHandler(event) {
    // Ensure the message comes from our iframe
    if (event.source !== iframe.contentWindow) return;
    
    console.log("[messageHandler] Received message:", event.data);
    
    const { type, data } = event.data;
    switch (type) {
      case 'IFRAME_READY':
        sendParamsToIframe();
        break;
      case 'TEXTURE_OUTPUT':
        console.log("[messageHandler] TEXTURE_OUTPUT received with data:", data);
        handleTextureOutput(data);
        break;
      case 'SCENE_STATE_UPDATED':
        if (data.state) {
          stateWidget.value = JSON.stringify(data.state, null, 2);
        }
        break;
      default:
        console.warn("[messageHandler] Unhandled message type:", type);
    }
  }

  // Attach the message handler
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
      type: 'PARAMS_UPDATE',
      data: {
        width: parseInt(widthWidget.value, 10) || 512,
        height: parseInt(heightWidget.value, 10) || 512,
        sceneState,
      }
    };

    if (iframe.contentWindow) {
      // If you know the target origin, replace '*' with that string
      iframe.contentWindow.postMessage(params, '*');
    }
  }

  // Function to handle texture output coming from the iframe
  async function handleTextureOutput(data) {
    // Make sure the node has a valid unique_id
    if (!node.unique_id) {
      console.warn("[handleTextureOutput] Node unique_id is not set, cannot update texture store.");
      return;
    }
    // For each expected texture key, update the hidden widget and the global texture store.
    const textureKeys = ["color", "canny", "depth", "normal"];
    textureKeys.forEach((key) => {
      if (data[key]) {
        node.setHiddenWidgetValue(key, {
          data: data[key].data,
          width: data[key].width,
          height: data[key].height,
          format: data[key].format,
          type: data[key].type,
          shape: data[key].shape
        });
      }
    });

    // Optionally clear heavy data from the global store if it is no longer needed
    if (node.unique_id && window.TEXTURE_STORE[node.unique_id]) {
        // You can conditionally delete or nullify large arrays after processing.
        delete window.TEXTURE_STORE[node.unique_id];
      }
    // Delay the prompt so that the global store has updated
    setTimeout(() => {
      console.log("[handleTextureOutput] Final global texture store for unique_id", node.unique_id, window.TEXTURE_STORE[node.unique_id]);
      app.queuePrompt();
    }, 50);
  }

  // Create the DOM widget for this node and register cleanup
  const widget = node.addDOMWidget("iframe", "custom_widget", iframeWrapper, {
    getValue() {
      return {
        url: urlWidget.value,
        width: widthWidget.value,
        height: heightWidget.value,
        scene_state: stateWidget.value
      };
    },
    setValue(v) {
      if (v.url) {
        iframe.src = v.url;
        urlWidget.value = v.url;
      }
    }
  });

  // Attach a reference to the iframe on the widget so we can access it later
  widget.iframe = iframe;

  // Handle resizing: update both node size and iframe dimensions, then re-send params
  node.onResize = function () {
    const width = parseInt(widthWidget.value, 10) || 512;
    const height = parseInt(heightWidget.value, 10) || 512;
    node.size = [width, Math.max(height + 40, 140)];
    iframe.style.width = width + "px";
    iframe.style.height = (height - 40) + "px";
    sendParamsToIframe();
  };

  // When the URL widget changes, update the iframe's src
  urlWidget.callback = function () {
    iframe.src = this.value;
  };

  // When any of the parameter widgets change, update the iframe with the new parameters
  widthWidget.callback = heightWidget.callback = stateWidget.callback = function () {
    sendParamsToIframe();
  };

  // Override the default setHiddenWidgetValue to also update our global TEXTURE_STORE.
  node.setHiddenWidgetValue = function (name, value) {
    let widgetEntry = this.widgets.find(w => w.name === name);
    if (widgetEntry) {
      widgetEntry.value = value;
      console.log(`[setHiddenWidgetValue] Updated widget '${name}' to:`, value);
    } else {
      // Create the widget if it does not exist.
      widgetEntry = { name, value, isHidden: true };
      this.widgets.push(widgetEntry);
      console.log(`[setHiddenWidgetValue] Created and set widget '${name}' to:`, value);
    }
    if (this.unique_id) {
      window.TEXTURE_STORE = window.TEXTURE_STORE || {};
      window.TEXTURE_STORE[this.unique_id] = window.TEXTURE_STORE[this.unique_id] || {};
      window.TEXTURE_STORE[this.unique_id][name] = value;
      console.log(`[setHiddenWidgetValue] Global texture store for unique_id '${this.unique_id}':`, window.TEXTURE_STORE[this.unique_id]);
    }
  };

  // When creating the widget
node.cleanup = () => {
    window.removeEventListener("message", messageHandler, false);
    if (node.unique_id && window.TEXTURE_STORE[node.unique_id]) {
      delete window.TEXTURE_STORE[node.unique_id];
    }
    console.log("[cleanup] Cleaned up resources for node:", node.unique_id);
  };

  // Updated onSerialize: Strip out the heavy "data" field to avoid localStorage quota problems.
  node.onSerialize = (nodeData) => {
    const textureKeys = ["color", "canny", "depth", "normal"];
    nodeData.hidden = nodeData.hidden || {};
    textureKeys.forEach((key) => {
      const widgetEntry = node.widgets.find(w => w.name === key);
      if (widgetEntry && widgetEntry.value) {
        // If the widget's value includes a "data" field, remove it from the serialization.
        if ("data" in widgetEntry.value) {
          const { data, ...metadata } = widgetEntry.value;
          nodeData.hidden[key] = metadata;
        } else {
          nodeData.hidden[key] = widgetEntry.value;
        }
      }
    });
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
              const domWidget = this.widgets.find(w => w.name === "iframe");
              if (domWidget && domWidget.iframe) {
                const url = this.widgets.find(w => w.name === "url").value;
                domWidget.iframe.src = url;
              }
            }
          },
          {
            content: "📸 Capture Textures",
            callback: () => {
              const domWidget = this.widgets.find(w => w.name === "iframe");
              if (domWidget?.iframe?.contentWindow) {
                domWidget.iframe.contentWindow.postMessage(
                  { type: 'CAPTURE_TEXTURES' },
                  '*'
                );
              }
            }
          }
        );
      };

      const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = async function () {
        // If nodeData.hidden definitions exist, add any missing widgets.
        if (nodeData && nodeData.hidden) {
          Object.keys(nodeData.hidden).forEach((key) => {
            if (!this.widgets.find(widget => widget.name === key)) {
              this.widgets.push({ name: key, value: nodeData.hidden[key] || null, isHidden: true });
              console.log(`[onNodeCreated] Hidden widget '${key}' added.`);
            }
          });
        }
        const r = originalOnNodeCreated ? originalOnNodeCreated.apply(this, arguments) : undefined;
        const widget = await widgetIframe(this, nodeData, nodeData, app);
        this.onResize();

        // Synchronize the node's unique_id with the one passed from Python.
        if (nodeData && nodeData.hidden && nodeData.hidden.unique_id) {
          this.unique_id = nodeData.hidden.unique_id;
          console.log(`[onNodeCreated] Using provided unique_id from nodeData: ${this.unique_id}`);
          let uniqueWidget = this.widgets.find(w => w.name === "unique_id");
          if (uniqueWidget) {
            uniqueWidget.value = this.unique_id;
          } else {
            this.widgets.push({ name: "unique_id", value: this.unique_id, isHidden: true });
          }
        } else {
          // Fallback: generate a unique id.
          this.unique_id = Math.random().toString(36).substr(2, 9);
          console.log(`[onNodeCreated] Generated fallback unique_id: ${this.unique_id}`);
          let uniqueWidget = this.widgets.find(w => w.name === "unique_id");
          if (uniqueWidget) uniqueWidget.value = this.unique_id;
          else this.widgets.push({ name: "unique_id", value: this.unique_id, isHidden: true });
        }
        return r;
      };
    }
  }
});

export { widgetIframe };

