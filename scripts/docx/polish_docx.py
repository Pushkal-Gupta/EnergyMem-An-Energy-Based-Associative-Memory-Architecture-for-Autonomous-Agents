"""Fix the layout defects pandoc's docx writer leaves behind, in place.

Five of them, four visible on the first page or in any wide table:

  * no page size or margins at all in sectPr, so Word falls back to whatever the
    local default template says and the text block is a different width on every
    machine;
  * the author block is left-aligned under a centred title;
  * body paragraphs are ragged-right, where the paper is justified;
  * every table carries an equal-width column grid with no layout rule, so a
    thirteen-column table gets the same column widths as a three-column one and
    the cells wrap into ribbons;
  * the png content type is never declared, which makes the package invalid and
    lets a strict reader drop both figures.

Run after fix_omml.py. Idempotent.
"""
import re, sys, zipfile

LETTER = '<w:pgSz w:h="15840" w:w="12240" />'
MARGINS = ('<w:pgMar w:bottom="1440" w:footer="720" w:gutter="0" w:header="720" '
           'w:left="1440" w:right="1440" w:top="1440" />')


def style_pPr_insert(xml, style_id, element):
    """Add `element` to the end of a style's pPr, creating the pPr if needed."""
    m = re.search(r'(<w:style [^>]*w:styleId="%s".*?</w:style>)' % style_id, xml, re.S)
    if not m:
        return xml, False
    block = m.group(1)
    if element in block:
        return xml, False
    empty = re.search(r"<w:pPr\s*/>", block)
    if "<w:pPr>" in block:
        new = block.replace("</w:pPr>", element + "</w:pPr>", 1)
    elif empty:                 # pandoc writes an empty <w:pPr /> in some styles
        new = block.replace(empty.group(0), "<w:pPr>" + element + "</w:pPr>", 1)
    else:                       # pPr goes after qFormat and before rPr
        anchor = "<w:qFormat />" if "<w:qFormat />" in block else "<w:name"
        if anchor == "<w:name":
            return xml, False
        new = block.replace(anchor, anchor + "<w:pPr>" + element + "</w:pPr>", 1)
    return xml.replace(block, new, 1), True


def main(path):
    with zipfile.ZipFile(path) as z:
        parts = {n: z.read(n) for n in z.namelist()}
    report = []

    # 0. pandoc writes the figures into word/media but never declares the png
    #    content type, so the package is strictly invalid and a strict reader can
    #    drop both figures. Word is lenient; Google Docs and LibreOffice need not be.
    ct = parts["[Content_Types].xml"].decode("utf-8")
    for ext, mime in (("png", "image/png"), ("jpeg", "image/jpeg"), ("jpg", "image/jpeg")):
        have = any(n.lower().endswith("." + ext) for n in parts)
        if have and 'Extension="%s"' % ext not in ct:
            ct = ct.replace("<Default Extension=\"xml\"",
                            '<Default Extension="%s" ContentType="%s" /><Default Extension="xml"'
                            % (ext, mime), 1)
            report.append("declared %s content type" % ext)
    parts["[Content_Types].xml"] = ct.encode("utf-8")

    doc = parts["word/document.xml"].decode("utf-8")

    # 1. page geometry: US Letter, one-inch margins, as the PDF has.
    if "<w:pgSz" not in doc:
        doc = doc.replace("</w:sectPr>", LETTER + MARGINS + "</w:sectPr>", 1)
        report.append("page size and margins set to Letter / 1 in")

    # 1b. table alt text. pandoc copies the caption into w:tblCaption verbatim,
    #     LaTeX and all, so a screen reader reads "\\beta, k and \\tau all selected".
    #     The caption is a visible paragraph above the table anyway; make the alt
    #     text plain.
    SYM = {r"\\beta": "\u03b2", r"\\tau": "\u03c4", r"\\alpha": "\u03b1",
           r"\\eta": "\u03b7", r"\\Delta": "\u0394", r"\\to": "\u2192",
           r"\\dagger": "\u2020", r"\\times": "\u00d7", r"\\pm": "\u00b1"}
    def plain(m):
        v = m.group(1)
        for k, r in SYM.items():
            v = re.sub(k + r"\b", r, v)
        v = re.sub(r"\\(?:textbf|textsc|emph|texttt|textit|mathrm|text)\{", "", v)
        v = v.replace("$", "").replace("{", "").replace("}", "")
        v = re.sub(r"\\[A-Za-z]+", "", v)
        v = re.sub(r"\\([%&_#])", r"\1", v)
        return 'w:tblCaption w:val="%s"' % re.sub(r"\s{2,}", " ", v).strip()
    n_before = len(re.findall(r'w:tblCaption w:val="[^"]*[\\$]', doc))
    doc = re.sub(r'w:tblCaption w:val="([^"]*)"', plain, doc)
    if n_before:
        report.append("%d table alt texts made plain" % n_before)

    # 2. tables fill the text column and Word sizes the columns by content.
    n_tbl = doc.count("<w:tblW ")
    doc = re.sub(r'<w:tblW w:type="auto" w:w="0" />',
                 '<w:tblW w:type="pct" w:w="5000" />', doc)
    before = doc.count("<w:tblLayout")
    doc = doc.replace("<w:tblLook", '<w:tblLayout w:type="autofit" /><w:tblLook')
    report.append("%d tables set to full width with content-sized columns"
                  % (doc.count("<w:tblLayout") - before or n_tbl))
    parts["word/document.xml"] = doc.encode("utf-8")

    # 3. styles: centre the byline, justify body text, centre figures, and set
    #    tables in 9 pt so the wide ones fit the column.
    st = parts["word/styles.xml"].decode("utf-8")
    for sid in ("Author", "CaptionedFigure", "Figure"):
        st, done = style_pPr_insert(st, sid, '<w:jc w:val="center" />')
        if done:
            report.append("%s centred" % sid)
    for sid in ("BodyText", "FirstParagraph", "Compact", "Bibliography"):
        st, done = style_pPr_insert(st, sid, '<w:jc w:val="both" />')
        if done:
            report.append("%s justified" % sid)
    # the byline is a six-line block; 12 pt (pandoc's default for Author, inherited
    #    from Title) makes it compete with the title, so set it at 10 pt.
    m = re.search(r'(<w:style [^>]*w:styleId="Author"[^>]*>.*?</w:style>)', st, re.S)
    if m and '<w:sz w:val="20"' not in m.group(1):
        st = st.replace(m.group(1),
                        re.sub(r'<w:sz(Cs)? w:val="\d+" />', '', m.group(1))
                        .replace("<w:rPr>", '<w:rPr><w:sz w:val="20" />'
                                            '<w:szCs w:val="20" />', 1), 1)
        report.append("byline set to 10 pt")

    # pandoc's defaults are presentation-sized: a 28 pt title and a 20 pt section
    # heading on a 12 pt body. Scale them to manuscript proportions, keeping the
    # hierarchy intact.
    for sid, half_pt in (("Title", 40), ("Heading1", 30), ("Heading2", 26),
                         ("Heading3", 24), ("Heading4", 24)):
        m = re.search(r'(<w:style [^>]*w:styleId="%s"[^>]*>.*?</w:style>)' % sid, st, re.S)
        if not m or 'w:sz w:val="%d"' % half_pt in m.group(1):
            continue
        block = m.group(1)
        if re.search(r'<w:sz w:val="\d+" />', block):
            new = re.sub(r'<w:sz w:val="\d+" />', '<w:sz w:val="%d" />' % half_pt, block)
            new = re.sub(r'<w:szCs w:val="\d+" />', '<w:szCs w:val="%d" />' % half_pt, new)
        elif "<w:rPr>" in block:
            new = block.replace("<w:rPr>", '<w:rPr><w:sz w:val="%d" />'
                                           '<w:szCs w:val="%d" />' % (half_pt, half_pt), 1)
        else:
            continue
        st = st.replace(block, new, 1)
        report.append("%s at %g pt" % (sid, half_pt / 2))

    m = re.search(r'(<w:style w:default="1" w:styleId="Table" w:type="table">.*?</w:style>)',
                  st, re.S)
    if m and "<w:rPr>" not in m.group(1):
        st = st.replace(m.group(1),
                        m.group(1).replace("<w:tblPr>",
                                           '<w:rPr><w:sz w:val="18" />'
                                           '<w:szCs w:val="18" /></w:rPr><w:tblPr>', 1), 1)
        report.append("table text set to 9 pt")
    parts["word/styles.xml"] = st.encode("utf-8")

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)
    print("polish_docx: " + "; ".join(report) if report else "polish_docx: nothing to do")


if __name__ == "__main__":
    main(sys.argv[1])
