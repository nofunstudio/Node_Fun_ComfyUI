class IndexedStringSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {"default": 0, "min": 0, "max": 20}),
                "names": (
                    "STRING",
                    {
                        "default": "Fluffy, Pastel, Paint, Option4, Option5, Option6, Option7, Option8, Option9, Option10, Option11, Option12, Option13, Option14, Option15, Option16, Option17, Option18, Option19, Option20",
                        "multiline": True,
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "select_name"

    def select_name(self, index, names):
        # Split the string by commas and remove extra whitespace
        names_list = [n.strip() for n in names.split(",") if n.strip()]
       
        return (names_list[index],)


NODE_CLASS_MAPPINGS = {
    "IndexedStringSelector": IndexedStringSelector
}