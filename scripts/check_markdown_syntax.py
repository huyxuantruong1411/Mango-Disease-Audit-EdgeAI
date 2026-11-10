import glob
import re
import os

repo_root = os.path.abspath("MANGO/Mango-Disease-Audit-EdgeAI")
md_files = glob.glob(f"{repo_root}/**/*.md", recursive=True)

print(f"Scanning {len(md_files)} markdown files in {repo_root}:\n")

all_good = True
for path in md_files:
    rel_path = os.path.relpath(path, repo_root)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Check for raw ^*
    star_matches = re.findall(r"(\w\^[\*])", content)

    # Check for double dollar $$
    double_dollars = content.count("$$")

    # Check for inline single dollar $
    # (Subtract double dollars)
    clean_no_dd = content.replace("$$", "")
    single_dollars = clean_no_dd.count("$")

    # Check for % inside math
    math_percents = re.findall(r"(\$[^$\n]*[^\\]%[^$\n]*\$)", content)

    print(f"File: {rel_path}")
    print(f"  - ^* occurrences: {len(star_matches)}")
    print(f"  - $$ count: {double_dollars} (properly paired: {double_dollars % 2 == 0})")
    print(f"  - $ count: {single_dollars} (properly paired: {single_dollars % 2 == 0})")
    if math_percents:
        print(f"  - WARNING: {len(math_percents)} unescaped % in math: {math_percents[:3]}")
    print()

    if len(star_matches) > 0 or double_dollars % 2 != 0:
        all_good = False

print(f"Markdown syntax check result: {'ALL PASSED' if all_good else 'ISSUES FOUND'}")
