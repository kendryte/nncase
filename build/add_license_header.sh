#!/bin/bash  
for f in $(find ../src/ -name '*.cpp' -or -name '*.h')
do
  echo "Processing $f" 
  if ! grep -q Copyright $f
  then
    cat license_header.txt $f >$f.new && mv $f.new $f
  fi
done
