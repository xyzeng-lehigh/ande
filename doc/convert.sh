#!/bin/bash
#Arguments -r                  : build with non-bib cross reference (3 passes)
#          -f                  : build with bib cross references (2+2+2 passes)
#          -n                  : number of latex builds without bibtex (2 passes)
#          -t                  : tex file name (default pap)

for f in *.toc *.lot *.lof *.aux
do
  if [ -f "$f" ]
  then
    rm $f
  fi
done

NT_BEF=3
NT_BIB=0
NT_AFT=0
TEXFILE=ande_manual

while getopts "rfs:t:" OPTION
do
  case $OPTION in
    r) NT_BEF=3
       ;;
    f) NT_BIB=2
       NT_AFT=2
       ;;
    n) NT_BEF=$OPTARG
       ;;
    t) TEXFILE=$OPTARG
       ;;
    ?) usage
       exit
       ;;
  esac
done

if [ ! -f "$TEXFILE.tex" ]
then
  echo "Error: $TEXFILE.tex does not exist!"
  exit
fi

for (( n=1; n<=$NT_BEF; n++ ))
do
  pdflatex $TEXFILE
done

for (( n=1; n<=$NT_BIB; n++ ))
do
  bibtex $TEXFILE
done

for (( n=1; n<=$NT_AFT; n++ ))
do
  pdflatex $TEXFILE
done

