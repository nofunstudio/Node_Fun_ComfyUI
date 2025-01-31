class WebViewer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {
                    "multiline": False,
                    "default": "https://nofun.io"
                }),
                "width": ("INT", {
                    "default": 800,
                    "min": 100,
                    "max": 2000,
                    "step": 10
                }),
                "height": ("INT", {
                    "default": 600,
                    "min": 100,
                    "max": 2000,
                    "step": 10
                })
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "view_webpage"
    CATEGORY = "Web"

    def view_webpage(self, url, width, height):
        import webbrowser
        import tempfile
        import os

        html_content = f"""
        <html>
            <body style="margin:0;padding:0;">
                <iframe src="{url}" width="{width}" height="{height}" frameborder="0"></iframe>
            </body>
        </html>
        """
        
        # Create a temporary HTML file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w') as f:
            f.write(html_content)
            temp_path = f.name

        # Open the temporary file in the default browser
        webbrowser.open('file://' + os.path.abspath(temp_path))
        
        return ()

# Register the node
NODE_CLASS_MAPPINGS = {
    "WebViewer": WebViewer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WebViewer": "Web Viewer"
} 