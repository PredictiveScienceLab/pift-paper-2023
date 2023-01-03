# A script that runs everything about example 2.
#
# Author:
#   Ilias Bilionis
#
# Date:
#   1/2/2023
#
if [ ! -f ./example02 ]; then
  echo "You haven't compiled the code or you are not running this script from the examples directory."
  exit 1
fi

if [ -d example02_resultsa ]; then
  rm -rf example02_resultsa
fi

mkdir example02_resultsa

id=0
for n in 10 20 40 80
do
for sigma in 0.0001 0.01
do
for gamma in 0.0 0.2 0.4 0.6 0.8 1.0
do
  cmd="./example02 $gamma example02.yml $n $sigma $id"
  echo $cmd
  eval $cmd
  if [ $? -ne 0 ]; then
    echo "*** FAILED ***"
    exit 2
  fi
  prefix=`printf "example02_gamma=%1.2e_n=%d_sigma=%1.2e_%d" $gamma $n $sigma $id`
  mv ${prefix}*.csv example02_resultsa/
  if [ $? -ne 0 ]; then
    echo "*** FAILED ***"
    exit 2
  fi
done
done
done
