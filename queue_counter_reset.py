# Define a global counter variable that persists across executions.
global_counter = 0

class DynamicQueueCounter:
    # This flag informs ComfyUI that this node should always run.
    ALWAYS_RUN = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Widget shows start as 0
                "start": ("FLOAT", {"default": 0.0}),
                # Widget shows stop as 8
                "stop": ("FLOAT", {"default": 8.0}),
                "step": ("FLOAT", {"default": 1.0}),
                # Toggle to reset counter to start
                "reset": ("BOOLEAN", {"default": False}),
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

    def node(self, start, stop, step, reset):
        display_offset = 1.0
        # Adjust widget values for internal calculation.
        # (The displayed value is normally calculated as internal_value + display_offset.)
        internal_start = float(start) - display_offset
        internal_stop = float(stop) - display_offset

        # Initialize counter if this is the first run
        if not hasattr(self, "counter"):
            self.counter = internal_start
            self.last_reset = reset
            print("DynamicQueueCounter: Initialized counter =", self.counter)
        else:
            # Check if user toggled the reset button
            if reset != self.last_reset:
                # Reset toggle state changed, reset counter to start
                self.counter = internal_start
                self.last_reset = reset
                print(f"DynamicQueueCounter: Reset toggled - counter reset to {self.counter} (display: {internal_start + display_offset})")

        if step == 0:
            display_current = self.counter + display_offset
            info = f"Output: {display_current}. Step is 0, counter remains unchanged."
            return float(display_current), int(display_current), info

        # Increment mode.
        if internal_start < internal_stop:
            next_value = self.counter + step
            if next_value >= internal_stop:
                # In reset branch, output the internal_stop directly (without adding offset)
                # so that with start=0 and stop=8 the final displayed number is 7.
                output = internal_stop
                display_output = output  # <-- FIX: remove extra display_offset addition.
                display_reset = internal_start + display_offset
                info = (
                    f"Output: {display_output}. Reached or exceeded stop; "
                    f"counter will reset to {display_reset} on next call."
                )
                self.counter = internal_start
            else:
                output = self.counter
                self.counter = next_value
                display_output = output + display_offset
                info = (
                    f"Output: {display_output}. Next will be {self.counter + display_offset}."
                )
        # Decrement mode remains unchanged.
        else:
            next_value = self.counter - step
            if next_value <= internal_stop:
                output = internal_stop
                display_output = output + display_offset
                display_reset = internal_start + display_offset
                info = (
                    f"Output: {display_output}. Reached or exceeded stop; "
                    f"counter will reset to {display_reset} on next call."
                )
                self.counter = internal_start
            else:
                output = self.counter
                self.counter = next_value
                display_output = output + display_offset
                info = (
                    f"Output: {display_output}. Next will be {self.counter + display_offset}."
                )

        return float(display_output), int(display_output), info