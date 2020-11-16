# Convert Latin1 encoded file (e.g. German and Dutch CoNLL) to UTF-8
iconv -f ISO8859-1 -t UTF8 $1 > $1
