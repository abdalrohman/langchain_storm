import pytest

from langchain_storm.utilities.markdown_converter import (
    FileHandler,
    MarkdownContent,
    MarkdownConverter,
)


@pytest.fixture
def file_handler(tmp_path):
    return FileHandler(output_dir=tmp_path)


@pytest.fixture
def markdown_converter(file_handler):
    return MarkdownConverter(file_handler=file_handler)


def test_validate_output_dir_existing(file_handler):
    path = file_handler.output_dir
    assert path.is_dir()


def test_save_file_str_content(file_handler):
    file_path = file_handler.save_file("test.txt", "Test content")
    assert file_path.exists()
    with open(file_path, "r") as file:
        assert file.read() == "Test content"


def test_save_file_bytes_content(file_handler):
    file_path = file_handler.save_file("test.bin", b"Test content")
    assert file_path.exists()
    with open(file_path, "rb") as file:
        assert file.read() == b"Test content"


def test_to_html(markdown_converter):
    markdown_content = MarkdownContent(
        content="**bold**", css="body { font-size: 14px; }"
    )
    html_content = markdown_converter.to_html(markdown_content)
    assert "<strong>bold</strong>" in html_content
    assert "<style>body { font-size: 14px; }</style>" in html_content


@pytest.mark.parametrize(
    "html_content, pdf_filename",
    [
        ("<h1>Title</h1>", "test.pdf"),
        # Add more test cases if necessary
    ],
)
def test_to_pdf(markdown_converter, html_content, pdf_filename):
    file_path = markdown_converter.to_pdf(html_content, pdf_filename)
    assert file_path.exists()
    # Additional assertions can be added to check the content of the PDF if needed
