"""First Proof LaTeX formatting contract helpers."""
from __future__ import annotations

import re


DEFAULT_FIRSTPROOF_PAGE_LIMIT = 12

_DOCUMENTCLASS_RE = re.compile(
    r"\\documentclass(?:\[(?P<options>[^\]]*)\])?\{(?P<class>[^}]*)\}",
    re.MULTILINE,
)

_FORBIDDEN_USEPACKAGES = (
    "geometry",
    "setspace",
    "doublespacing",
    "savetrees",
    "a4wide",
    "anysize",
    "typearea",
    "vmargin",
    "extsizes",
)

_USEPACKAGE_RE = re.compile(
    r"\\usepackage\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}",
)

_FORBIDDEN_FORMATTING_COMMANDS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\\(?:newgeometry|restoregeometry|geometry)\s*\{[^}]*\}\s*"), "geometry command"),
    (re.compile(r"\\(?:onehalfspacing|doublespacing|singlespacing)\b\s*"), "spacing command"),
    (re.compile(r"\\(?:setstretch|linespread|spacing)\s*\{[^}]*\}\s*"), "spacing command"),
    (re.compile(r"\\(?:tiny|scriptsize|footnotesize|small|large|Large|LARGE|huge|Huge)\b\s*"), "font-size command"),
    (re.compile(r"\\fontsize\s*\{[^}]*\}\s*\{[^}]*\}\s*\\selectfont\s*"), "font-size command"),
    (
        re.compile(
            r"\\(?:renewcommand|def)\s*\{?\\baselinestretch\}?\s*(?:\[[^\]]*\])?\s*\{[^}]*\}\s*"
        ),
        "baselinestretch command",
    ),
    (
        re.compile(
            r"\\setlength\s*\{\s*\\(?:textwidth|textheight|oddsidemargin|evensidemargin|topmargin|headheight|headsep|footskip|baselineskip)\s*\}\s*\{[^}]*\}\s*"
        ),
        "manual page-layout length",
    ),
    (
        re.compile(
            r"\\(?:addtolength)\s*\{\s*\\(?:textwidth|textheight|oddsidemargin|evensidemargin|topmargin|headheight|headsep|footskip|baselineskip)\s*\}\s*\{[^}]*\}\s*"
        ),
        "manual page-layout length",
    ),
)


def render_firstproof_latex_contract(page_limit: int) -> str:
    """Human-facing formatting contract for Author/Critic prompts."""
    return f"""\
First Proof LaTeX submission contract:
- `answer.tex` must be a complete standalone LaTeX document.
- Use exactly `\\documentclass[12pt]{{article}}`.
- The compiled PDF must be at most {page_limit} pages.
- The `fullpage` package is permitted.
- Do not use any other margin/layout changes: no `geometry`, `a4wide`,
  `typearea`, `anysize`, manual `\\textwidth`, `\\textheight`,
  `\\oddsidemargin`, `\\evensidemargin`, `\\topmargin`, etc.
- Do not change line spacing: no `setspace`, `\\linespread`,
  `\\baselinestretch`, `\\onehalfspacing`, or `\\doublespacing`.
- Do not change font size inside the document: no `\\small`,
  `\\footnotesize`, `\\scriptsize`, or `\\fontsize`.
- Run `pdflatex` and fix compile errors before claiming readiness.
"""


def normalize_firstproof_latex(tex: str, *, removals: list[str] | None = None) -> str:
    """Return a complete 12pt article with forbidden formatting stripped."""
    return ensure_complete_latex(tex, removals=removals)


def strip_forbidden_packages(tex: str) -> tuple[str, list[str]]:
    """Remove forbidden formatting packages while preserving harmless packages.

    ``fullpage`` is intentionally not forbidden: First Proof permits it.
    If a forbidden package shares an option list with harmless packages,
    the harmless packages are re-emitted without the shared options.
    """
    removals: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        full = match.group(0)
        opts_match = re.search(r"\\usepackage\s*(\[[^\]]*\])?", full)
        opts_text = opts_match.group(1) if opts_match and opts_match.group(1) else ""
        has_opts = bool(opts_text)
        names = [n.strip() for n in match.group(1).split(",") if n.strip()]
        dropped = [n for n in names if n in _FORBIDDEN_USEPACKAGES]
        if not dropped and not (has_opts and "fullpage" in names):
            return full
        keep = [n for n in names if n not in _FORBIDDEN_USEPACKAGES]
        if has_opts:
            if "fullpage" in keep:
                removals.append("removed options from permitted \\usepackage{fullpage}")
            if not dropped:
                return f"\\usepackage{{{','.join(keep)}}}" if keep else ""
            if keep:
                removals.append(
                    f"removed \\usepackage{opts_text}{{{','.join(dropped)}}} "
                    f"and re-emitted survivors without options: "
                    f"\\usepackage{{{','.join(keep)}}}"
                )
                return f"\\usepackage{{{','.join(keep)}}}"
            removals.append(
                f"removed \\usepackage{opts_text}{{{','.join(dropped)}}}"
            )
            return ""
        removals.append(f"removed \\usepackage{{{','.join(dropped)}}}")
        if not keep:
            return ""
        return f"\\usepackage{{{','.join(keep)}}}"

    cleaned = _USEPACKAGE_RE.sub(_replace, tex)
    cleaned = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", cleaned)
    return cleaned, removals


def strip_forbidden_formatting_commands(tex: str) -> tuple[str, list[str]]:
    removals: list[str] = []
    cleaned = tex
    for pattern, label in _FORBIDDEN_FORMATTING_COMMANDS:
        cleaned, count = pattern.subn("", cleaned)
        if count:
            removals.append(f"removed {count} forbidden {label}(s)")
    cleaned = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", cleaned)
    return cleaned, removals


def ensure_complete_latex(tex: str, *, removals: list[str] | None = None) -> str:
    stripped = tex.strip()
    if not stripped:
        stripped = "No solution text was produced."
    has_docclass = r"\documentclass" in stripped
    has_begin = r"\begin{document}" in stripped
    has_end = r"\end{document}" in stripped
    if has_docclass and has_begin:
        out = stripped
        if not has_end:
            out += "\n\\end{document}"
        return sanitize_complete_document(out, removals=removals) + "\n"
    if has_docclass and not has_begin:
        out = stripped + "\n\\begin{document}\n\\end{document}"
        return sanitize_complete_document(out, removals=removals) + "\n"

    sanitised, package_removals = strip_forbidden_packages(stripped)
    sanitised, command_removals = strip_forbidden_formatting_commands(sanitised)
    if removals is not None:
        removals.extend(package_removals)
        removals.extend(command_removals)
    if r"\begin{document}" in sanitised:
        sanitised = re.sub(r"\\begin\{document\}", "", sanitised, count=1)
        if removals is not None:
            removals.append("removed unmatched \\begin{document} from partial input")
    if r"\end{document}" in sanitised:
        sanitised = re.sub(r"\\end\{document\}", "", sanitised, count=1)
        if removals is not None:
            removals.append("removed unmatched \\end{document} from partial input")
    sanitised = sanitised.strip() or "No solution text was produced."
    return (
        "\\documentclass[12pt]{article}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage{amsmath,amssymb,amsthm}\n"
        "\\begin{document}\n"
        f"{sanitised}\n"
        "\\end{document}\n"
    )


def sanitize_complete_document(tex: str, *, removals: list[str] | None = None) -> str:
    normalized = normalize_documentclass(tex, removals=removals)
    cleaned, package_removals = strip_forbidden_packages(normalized)
    cleaned, command_removals = strip_forbidden_formatting_commands(cleaned)
    if removals is not None:
        removals.extend(package_removals)
        removals.extend(command_removals)
    return cleaned


def normalize_documentclass(tex: str, *, removals: list[str] | None = None) -> str:
    """Normalize the submitted document class to exactly 12pt article."""
    match = _DOCUMENTCLASS_RE.search(tex)
    if not match:
        return tex
    raw_options = match.group("options") or ""
    parsed = [part.strip() for part in raw_options.split(",") if part.strip()]
    original_class = match.group("class")
    if removals is not None:
        if original_class and original_class != "article":
            removals.append(f"rewrote document class {original_class!r} -> 'article'")
        dropped_options = [opt for opt in parsed if opt != "12pt"]
        if dropped_options:
            removals.append(
                "removed nonstandard \\documentclass option(s) "
                + ",".join(dropped_options)
            )
        if "12pt" not in parsed:
            removals.append("set \\documentclass option to 12pt")
    replacement = "\\documentclass[12pt]{article}"
    return _DOCUMENTCLASS_RE.sub(lambda _match: replacement, tex, count=1)
