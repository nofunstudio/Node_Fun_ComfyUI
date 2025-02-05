# Define a global counter variable that persists across executions.
global_counter = 0

class DynamicQueueCounter:
    # This flag informs ComfyUI that this node should always run.
    ALWAYS_RUN = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Widget shows start as 0, but we subtract 1 internally.
                "start": ("FLOAT", {"default": 0.0}),
                # Widget shows stop as 3, but we subtract 1.5 internally.
                "stop": ("FLOAT", {"default": 3.0}),
                "step": ("FLOAT", {"default": 1.0}),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT", "STRING")
    RETURN_NAMES = ("FLOAT", "INT", "Info")
    FUNCTION = "node"
    CATEGORY = "custom.number"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always return NaN so that ComfyUI re-runs this node each prompt.
        return float("NaN")

    def node(self, start, stop, step):
        # Adjust widget values internally.
        internal_start = float(start) - 1.0
        internal_stop = float(stop) - 1.5

        # Initialize the counter on first run.
        if not hasattr(self, "counter"):
            self.counter = internal_start
            print("SimpleNumberCounter: Initialized counter =", self.counter)

        current = self.counter

        # If no step, do not change the counter.
        if step == 0:
            info = f"Output: {current}. Step is 0, counter remains unchanged."
            return float(current), int(current), info

        # Determine the counting mode based on internal_start vs internal_stop.
        if internal_start < internal_stop:  # Increment mode
            next_value = self.counter + step
            if next_value >= internal_stop:
                output = internal_stop
                # Reset to the internal start after outputting the stop value.
                self.counter = internal_start
            else:
                output = self.counter
                self.counter = next_value
        else:  # Decrement mode
            next_value = self.counter - step
            if next_value <= internal_stop:
                output = internal_stop
                self.counter = internal_start
            else:
                output = self.counter
                self.counter = next_value

        info = f"Output: {output}. Next will be {self.counter}."
        return float(output), int(output), info