from pathlib import Path
from typing import Optional, Union

import markdown
from pydantic import BaseModel, field_validator
from weasyprint import HTML


class MarkdownContent(BaseModel):
    content: str
    css: Optional[str] = None


class FileHandler(BaseModel):
    output_dir: Union[Path | str]

    @field_validator("output_dir", mode="before")
    @classmethod
    def validate_output_dir(cls, value):
        if isinstance(value, str):
            value = Path(value)
        if not value.is_dir():
            value.mkdir(parents=True, exist_ok=True)
        return value

    def save_file(self, filename: str, content: Union[bytes | str]) -> Path:
        file_path = self.output_dir / filename
        if isinstance(content, str):
            with open(file_path, "w") as file:
                file.write(content)
            return file_path
        elif isinstance(content, bytes):
            with open(file_path, "wb") as file:
                file.write(content)
            return file_path


class MarkdownConverter(BaseModel):
    file_handler: FileHandler

    def to_html(self, markdown_content: MarkdownContent) -> str:
        html_content = markdown.markdown(markdown_content.content)
        if markdown_content.css:
            html_content = f"<style>{markdown_content.css}</style>\n{html_content}"
        return html_content

    def to_pdf(self, html_content: str, pdf_filename: str) -> Path:
        html = HTML(string=html_content)
        pdf_content = html.write_pdf()
        return self.file_handler.save_file(pdf_filename, pdf_content)


# # Example Usage
# def convert_and_save(
#         markdown_content: MarkdownContent,
#         output_directory: str,
#         pdf_filename: str,
#         docx_filename: str
#         ):
#     file_manager = FileHandler(output_dir=output_directory)
#     converter = MarkdownConverter(file_handler=file_manager)
#
#     # Convert Markdown to HTML and PDF
#     html_content = converter.to_html(markdown_content)
#     html_file_path = file_manager.save_file('example.html', html_content)
#     print(f"HTML file saved at: {html_file_path}")
#     pdf_file_path = converter.to_pdf(html_content, pdf_filename)
#     print(f"PDF file saved at: {pdf_file_path}")
#
#
# # Running the example
# markdown_content = MarkdownContent(
#     content="# Hello, World!\nThis is a markdown example.", css="body { font-family: Arial; }"
#     )
# convert_and_save(markdown_content, 'output', 'example.pdf', 'example.docx')
