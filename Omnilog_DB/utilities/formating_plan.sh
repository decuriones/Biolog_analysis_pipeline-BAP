#!/bin/bash
cd /home/elouanln/projects/def-jcomte/elouanln/Sandbox/Code/Phenotyping/Omnilog/analysis_pipeline/Omnilog_DB

awk '
  /^[A-H](1[0-2]|[1-9])$/ {
      if (coord != "") print coord "\t" buf
      coord = $0; buf = ""; next
  }
  NF {                                  # ignore les lignes vides
      if (buf ~ /-$/) buf = buf $0
      else buf = (buf == "" ? $0 : buf " " $0)
  }
  END { if (coord != "") print coord "\t" buf }
' PM3B_plan.txt | tr ' ' '_'| sed "s/\t/\'\:\'/g" | sed "s/^/\'/g" | sed "s/$/'/g" | sed "s/$/,/g" | tr '\n' ' ' | sed "s/^/{/g" | sed "s/$/}/g" > PM3B_plan.json
