#!/usr/bin/env python3
"""Download an original PDF, preserve a note scaffold, and register it once."""
import argparse
from datetime import date, datetime, timedelta, timezone
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from urllib.parse import quote, urlparse

import requests
from pypdf import PdfReader
from pypdf.errors import PdfReadError


ID_PATTERN = r'(?:\d{4}\.\d{4,5}|[a-zA-Z][a-zA-Z.\-]+/\d{7})(?:v\d+)?'
USER_AGENT = 'Paper-Reading/paper-intake (personal research library)'


def identify_arxiv(value):
    value = value.strip()
    if re.fullmatch(ID_PATTERN, value):
        return value
    parsed = urlparse(value)
    if parsed.hostname not in {'arxiv.org', 'www.arxiv.org', 'export.arxiv.org',
                               'alphaxiv.org', 'www.alphaxiv.org'}:
        return None
    match = re.fullmatch(r'/(?:abs|pdf)/(' + ID_PATTERN + r')(?:\.pdf)?/?', parsed.path)
    return match.group(1) if match else None


class MetadataParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.meta = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'meta' and attrs.get('name') and 'content' in attrs:
            self.meta.setdefault(attrs['name'].lower(), []).append(attrs['content'])


def get_metadata(paper_id):
    with requests.get(f'https://arxiv.org/abs/{paper_id}',
                      headers={'User-Agent': USER_AGENT}, timeout=(15, 90)) as response:
        response.raise_for_status()
        html = response.text
    parser = MetadataParser()
    parser.feed(html)
    meta = parser.meta
    title = ' '.join(meta.get('citation_title', [''])[0].split())
    if not title:
        raise ValueError('The arXiv page does not contain a paper title.')
    base_id = re.sub(r'v\d+$', '', paper_id)
    if not re.search(r'v\d+$', paper_id):
        versions = re.findall(re.escape(base_id) + r'v(\d+)\b', html)
        if versions:
            paper_id = f'{base_id}v{max(map(int, versions))}'
    return {
        'title': title,
        'authors': '、'.join(
            ' '.join(reversed([part.strip() for part in name.split(',', 1)]))
            if ',' in name else name for name in meta.get('citation_author', [])
        ) or '待核实',
        'published': meta.get('citation_date', ['待核实'])[0].replace('/', '-'),
        'arxiv_id': paper_id,
        'base_id': base_id,
        'source': f'https://www.alphaxiv.org/abs/{base_id}',
        'pdf_urls': [f'https://arxiv.org/pdf/{paper_id}',
                     f'https://pdfs.assets.alphaxiv.org/{paper_id}.pdf'],
    }


def normalize(value):
    return ''.join(c for c in value.casefold() if c.isalnum())


def check_pdf(path, title):
    with path.open('rb') as stream:
        if not stream.read(5) == b'%PDF-':
            raise ValueError(f'Not a PDF: {path.name}')
        reader = PdfReader(stream)
        if reader.is_encrypted or not reader.pages:
            raise ValueError(f'Unreadable or empty PDF: {path.name}')
        first_pages = '\n'.join(p.extract_text() or '' for p in reader.pages[:2])
        key = normalize(title)[:55]
        if key and key not in normalize(first_pages):
            raise ValueError(f'PDF title does not match the requested paper: {path.name}')
        return len(reader.pages)


def download_pdf(urls, target, title):
    errors = []
    for url in dict.fromkeys(urls):
        part = None
        try:
            fd, filename = tempfile.mkstemp(prefix=target.stem + '-', suffix='.part', dir=target.parent)
            part = Path(filename)
            with os.fdopen(fd, 'wb') as output:
                with requests.get(url, headers={'User-Agent': USER_AGENT},
                                  stream=True, timeout=(15, 120)) as response:
                    response.raise_for_status()
                    for chunk in response.iter_content(128 * 1024):
                        output.write(chunk)
            pages = check_pdf(part, title)
            if target.exists():
                raise FileExistsError(f'PDF appeared during download; refusing to replace {target}')
            part.replace(target)
            return pages
        except (requests.RequestException, OSError, ValueError, PdfReadError) as exc:
            errors.append(f'{url}: {exc}')
        finally:
            if part is not None:
                part.unlink(missing_ok=True)
    raise ValueError('Could not download a verified original PDF.\n' + '\n'.join(errors))


def relative_link(label, source_file, target):
    relative = os.path.relpath(target, source_file.parent).replace('\\', '/')
    return f'[{label}]({quote(relative, safe="/._-")})'


def find_existing_note(repo, metadata):
    matches = []
    for path in (repo / 'papernotes').rglob('*.md'):
        text = path.read_text(encoding='utf-8-sig')
        same_id = metadata.get('base_id') and re.search(
            r'(?:arxiv\.org|alphaxiv\.org)/(?:abs|pdf)/' + re.escape(metadata['base_id']) + r'(?:v\d+)?(?=[\s).#/?<>]|$)', text)
        headings = re.findall(r'^#\s+(.+?)\s*$', text, re.M)
        same_title = any(normalize(h) == normalize(metadata['title']) for h in headings)
        if same_id or same_title:
            matches.append(path)
    if len(matches) > 1:
        raise ValueError('Multiple existing notes match this paper: ' + ', '.join(str(p) for p in matches))
    return matches[0] if matches else None


def note_text(metadata, args, note, pdf, added_date):
    summary = args.summary or '一两句话简介（待填写）'
    lines = [
        f'<!-- {summary} -->',
        f'<!-- 作者：{metadata["authors"]}；机构：{args.institutions or "待核实"}；发表会议：{args.venue or "待核实"}；发表时间：{metadata["published"]}；md撰写日期：{added_date} -->',
        f'# {metadata["title"]}', '',
    ]
    sources = [f'[论文来源]({metadata["source"]})']
    if metadata.get('arxiv_id'):
        paper_id = metadata['arxiv_id']
        version = re.search(r'v\d+$', paper_id)
        sources = [f'[alphaXiv]({metadata["source"]})',
                   f'[arXiv{(" " + version.group()) if version else ""}](https://arxiv.org/abs/{paper_id})']
    for label, value in [('Code', args.code), ('Project', args.project)]:
        if value:
            sources.append(f'[{label}]({value})')
    lines += [' / '.join(sources), '', relative_link('原文 PDF', note, pdf), '',
              '## 问题', '', '', '## 方法', '', '', '## 实验', '', '',
              '# 附录', '', '# Noun explanation && Extensive knowledge', '',
              '# 思考？', '', '', '问题：', '认知增量：', '方法：', 'gap：', '']
    return '\n'.join(lines)


def update_index(index, metadata, note, pdf, month_dir):
    raw = index.read_bytes()
    text = raw.decode('utf-8-sig')
    note_relative = os.path.relpath(note, index.parent).replace('\\', '/')
    pdf_relative = os.path.relpath(pdf, index.parent).replace('\\', '/')
    note_present = quote(note_relative, safe='/._-') in text or note_relative in text
    pdf_present = quote(pdf_relative, safe='/._-') in text or pdf_relative in text
    existing_entry = None
    if metadata.get('base_id'):
        existing_entry = re.search(
            r'(?:arxiv\.org|alphaxiv\.org)/(?:abs|pdf)/' + re.escape(metadata['base_id']) + r'(?:v\d+)?(?=[\s).#/?<>]|$)', text)
    if not existing_entry:
        existing_entry = re.search(r'^\[' + re.escape(metadata['title']) + r'\]\([^)]+\)', text, re.M)
    if note_present and pdf_present:
        return False
    newline = '\r\n' if raw.count(b'\r\n') > raw.count(b'\n') / 2 else '\n'
    year, month = month_dir.split('_')
    heading = f'# {year}.{int(month)}'
    record = (f'[{metadata["title"]}]({metadata["source"]}){newline}{newline}'
              f'{relative_link("笔记", index, note)} / {relative_link("原文 PDF", index, pdf)}')
    heading_match = re.search(r'^' + re.escape(heading) + r'[ \t]*\r?$', text, re.M)
    def gap(value):
        if value.endswith(newline * 2):
            return ''
        return newline if value.endswith(newline) else newline * 2

    if existing_entry:
        # Older reading-list entries may have only a paper URL; add the missing local links.
        end = text.find('\n', existing_entry.end())
        end = len(text) if end < 0 else end + 1
        links = []
        if not note_present:
            links.append(relative_link('笔记', index, note))
        if not pdf_present:
            links.append(relative_link('原文 PDF', index, pdf))
        updated = text[:end] + gap(text[:end]) + ' / '.join(links) + newline * 2 + text[end:]
    elif heading_match:
        next_heading = re.search(r'^# ', text[heading_match.end():], re.M)
        insertion = heading_match.end() + next_heading.start() if next_heading else len(text)
        prefix, suffix = text[:insertion], text[insertion:]
        prefix += gap(prefix)
        updated = prefix + record + newline * 2 + suffix
    else:
        updated = text + gap(text)
        updated += heading + newline * 2 + record + newline
    updated = updated.rstrip('\r\n') + newline
    if raw.startswith(b'\xef\xbb\xbf'):
        updated = '\ufeff' + updated
    fd, filename = tempfile.mkstemp(prefix='.papers-', suffix='.part', dir=index.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(updated.encode('utf-8'))
        os.replace(filename, index)
    finally:
        Path(filename).unlink(missing_ok=True)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paper', help='arXiv/alphaXiv URL, arXiv ID, or a verified paper URL')
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument('--date', type=date.fromisoformat,
                        default=datetime.now(timezone(timedelta(hours=8))).date())
    parser.add_argument('--month-dir', help='Optional YY_M collection directory')
    for option in ['name', 'summary', 'institutions', 'venue', 'code', 'project',
                   'title', 'authors', 'published', 'pdf-url']:
        parser.add_argument('--' + option)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    index = repo / 'papers/papers.md'
    if not index.is_file() or not (repo / 'papernotes').is_dir():
        parser.error('--repo must contain papers/papers.md and papernotes/')
    month_dir = args.month_dir or f'{args.date:%y}_{args.date.month}'
    if not re.fullmatch(r'\d{2}_(?:[1-9]|1[0-2])', month_dir):
        parser.error('--month-dir must use YY_M, with month 1 through 12')
    paper_id = identify_arxiv(args.paper)
    if paper_id:
        metadata = get_metadata(paper_id)
    else:
        if not args.title or not args.pdf_url:
            parser.error('For non-arXiv papers, supply verified --title and --pdf-url')
        metadata = {'title': args.title, 'authors': args.authors or '待核实',
                    'published': args.published or '待核实', 'source': args.paper,
                    'pdf_urls': [args.pdf_url]}
    existing = find_existing_note(repo, metadata)
    name = args.name or ''.join(re.findall(r'[A-Za-z0-9]+', metadata['title']))[:90]
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', name) or re.fullmatch(
        r'(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?', name, re.I
    ) or name.endswith('.'):
        parser.error('Use a short ASCII filename stem without path separators or Windows reserved names')
    note = existing or repo / 'papernotes' / month_dir / f'{name}.md'
    if not existing and note.exists():
        raise FileExistsError(f'Another note already uses this name: {note}')
    # A repeated intake reuses the original location instead of creating another month copy.
    pdf = repo / 'papers' / note.relative_to(repo / 'papernotes').with_suffix('.pdf')
    if existing and re.fullmatch(r'\d{2}_\d{1,2}', note.parent.name):
        month_dir = note.parent.name
    report = {'title': metadata['title'], 'arxiv_id': metadata.get('arxiv_id'),
              'authors': metadata['authors'], 'published': metadata['published'],
              'note': str(note), 'pdf': str(pdf), 'existing_note': bool(existing)}
    if args.dry_run:
        print(json.dumps({'dry_run': True, **report}, ensure_ascii=False, indent=2))
        return
    pdf.parent.mkdir(parents=True, exist_ok=True)
    report['pages'] = check_pdf(pdf, metadata['title']) if pdf.exists() else download_pdf(
        metadata['pdf_urls'], pdf, metadata['title'])
    if not existing:
        note.parent.mkdir(parents=True, exist_ok=True)
        with note.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(note_text(metadata, args, note, pdf, args.date.isoformat()))
    report['index_updated'] = update_index(index, metadata, note, pdf, month_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'paper-intake: {exc}', file=sys.stderr)
        sys.exit(1)
