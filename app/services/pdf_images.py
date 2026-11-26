"""Describes the pictures inside a PDF with the language model, so their content can be searched."""
import base64
import io

import openai
from langchain_community.document_loaders.parsers import LLMImageBlobParser
from langchain_core.messages import HumanMessage


class PdfImageParser(LLMImageBlobParser):
    """
    LangChain's parser with two fixes:

    - It saves each picture as PNG but labels it JPEG. Claude refuses an image whose label does
      not match, which failed every PDF containing a picture. Here the label says PNG.
    - A picture the model refuses (too large, too small, unreadable) is left out, instead of
      failing the whole document. Other errors, such as an outage, are still raised.
    """

    def _analyze_image(self, img) -> str:
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        message = HumanMessage(content=[
            {"type": "text", "text": self.prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
        ])
        try:
            return self.model.invoke([message]).content
        except openai.BadRequestError as error:
            print(f"⚠ Warning: The model refused a picture ({img.width}x{img.height}), "
                  f"so it was left out: {str(error)[:300]}")
            return ""
