import sys
import json
import re

def tex_escape(text):
    """
        :param text: a plain text message
        :return: the message escaped to appear correctly in LaTeX
				from https://stackoverflow.com/questions/16259923/how-can-i-escape-latex-special-characters-inside-django-templates
    """
    conv = {
        #'&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
        '\\': r'\textbackslash{}',
        '<': r'\textless{}',
        '>': r'\textgreater{}',
    }
    regex = re.compile('|'.join(re.escape(str(key)) for key in sorted(conv.keys(), key = lambda item: - len(item))))
    return regex.sub(lambda match: conv[match.group()], text)

with open(sys.argv[1]) as fin:
  records = json.load(fin)
  for record in records:
    if isinstance(record, dict):
      for key in record:
        print(key, record[key])
    else:
      print(tex_escape(record))
    print()
    print()
    print()
    input()

