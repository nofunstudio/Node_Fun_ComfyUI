class StringLower:
    CATEGORY = "Text"   # shows up under "Text" in the node picker

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING",),   # one STRING input
            }
        }

    RETURN_TYPES = ("STRING",)  # one STRING output (fixed syntax error)
    FUNCTION = "to_lower"         # name of the method to call

    def to_lower(self, text):
        # do the work and return a tuple matching RETURN_TYPES
        return (text.lower(),) 