ORIGINAL=$1
TAGSET06=$2
OUTPUT=$3

paste $ORIGINAL $TAGSET06 | awk '{if($2=="") printf("\n"); else printf("%s %s %s\n", $1, $2, $9); }' > $OUTPUT
