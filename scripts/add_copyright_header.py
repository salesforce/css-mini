from __future__ import annotations
import argparse
import sys
from typing import Sequence


class CopyrightHeader:
    def __init__(self, text: str) -> None:
        self.text = text
        self.commented_text = "\n".join(
            f"# {line}" if line else "#" for line in text.splitlines()
        )

    def is_correct(self, content: str) -> bool:
        return content.startswith(self.commented_text)

    def prepend(self, content: str) -> str:
        return self.commented_text + "\n" + content

    @staticmethod
    def starts_with_copyright(line: str) -> bool:
        return line.startswith("# Copyright")

    @classmethod
    def from_source(cls) -> CopyrightHeader:
        with open(".github/COPYRIGHT_HEADER", encoding="utf-8") as fd:
            return cls(fd.read())


def check_missing_copyright_header(
    header: CopyrightHeader,
    content: str,
    path: str
) -> bool:
    lines = content.splitlines()
    if len(lines) < 2:
        return True
    if header.starts_with_copyright(lines[0]):
        if header.is_correct(content):
            return False
        print(f"Malformed header in {path}")
        sys.exit(1)
    return True



def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)

    header = CopyrightHeader.from_source()

    for path in args.paths:
        with open(path, encoding="utf-8") as fd:
            content = fd.read()
        if check_missing_copyright_header(header, content, path):
            if args.replace:
                new_content = header.prepend(content)
                with open(path, "w", encoding="utf-8") as fd:
                    fd.write(new_content)
            else:
                print(f"Missing header in {path}")
                sys.exit(1)



if __name__ == "__main__":
    main()