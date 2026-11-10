import glob
import re
import os
import sys

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
md_files = glob.glob(f"{repo_root}/**/*.md", recursive=True)
md_files = [f for f in md_files if ".pytest_cache" not in f]

print(f"Scanning {len(md_files)} markdown files in {repo_root}:\n")

all_good = True
for path in md_files:
    rel_path = os.path.relpath(path, repo_root)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        lines = content.splitlines()

    # 1. Check for disallowed macros
    ops = re.findall(r'\\operatorname\b', content)

    # 2. Check for raw ^*
    raw_stars = re.findall(r'\^\*', content)

    # 3. Check for double dollar count parity
    double_dollars = content.count("$$")

    # 4. Check for nested > $$
    quote_math = re.findall(r'>\s*\$\$', content)

    # 5. Check display math blocks formatting
    display_blocks = re.findall(r'\$\$(.*?)\$\$', content, re.DOTALL)

    # 6. Check blank lines around $$
    blank_line_issues = 0
    in_math = False
    for idx, line in enumerate(lines):
        line_num = idx + 1
        stripped = line.strip()
        if stripped == '$$':
            if not in_math:
                in_math = True
                if idx > 0 and lines[idx - 1].strip() != '':
                    blank_line_issues += 1
            else:
                in_math = False
                if idx + 1 < len(lines) and lines[idx + 1].strip() != '':
                    blank_line_issues += 1

    print(f"File: {rel_path}")
    print(f"  - \\operatorname occurrences: {len(ops)}")
    print(f"  - Raw ^* occurrences: {len(raw_stars)}")
    print(f"  - Display blocks ($$): {len(display_blocks)} (pairs: {double_dollars // 2})")
    print(f"  - Nested > $$ occurrences: {len(quote_math)}")
    print(f"  - Blank line issues around $$: {blank_line_issues}")

    if ops or raw_stars or (double_dollars % 2 != 0) or quote_math or blank_line_issues:
        all_good = False
        print(f"    --> ISSUES DETECTED IN {rel_path}!")
    else:
        print(f"    --> Status: CLEAN")
    print()

if not all_good:
    print("Syntax verification FAILED.")
    sys.exit(1)
else:
    print("ALL MARKDOWN FILES PASSED SYNTAX & MATHJAX COMPATIBILITY CHECKS!")
    sys.exit(0)
